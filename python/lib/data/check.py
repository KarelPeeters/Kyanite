from typing import List

from lib.data.file import DataFile
from lib.data.group import DataGroup
from lib.data.position import Position
from lib.data.sampler import PositionSampler


def check_data_file(file: DataFile):
    print(f"Checking {file.info.bin_path}")
    print(f"  Simulations: {len(file.simulations)}")
    print(f"  Positions: {len(file.positions)}")
    print(f"  Includes final: {file.info.includes_final_positions}")

    count_covered = 0
    positions: List[Position] = list(file.positions)

    for sim in file.simulations:
        print(f"  checking {sim.index + 1}/{len(file.simulations)}")

        matching_positions = [p for p in positions if p.simulation.index == sim.index]
        count_covered += len(matching_positions)

        actual_indices = [p.file_pi for p in matching_positions]
        assert list(sim.file_pis) == actual_indices

        for p in matching_positions:
            assert p.simulation == sim

        assert matching_positions[-1].is_final_position == file.info.includes_final_positions

    if len(file.simulations) >= 50:
        sim_count = len(file.simulations)

        middle_slice = file.simulations[sim_count // 3: 2 * sim_count // 3]

        assert file.positions[middle_slice[0].start_file_pi].move_index == 0
        assert file.positions[middle_slice[-1].end_file_pi - 1].is_final_position == file.info.includes_final_positions
        assert file.positions[middle_slice[-1].end_file_pi].move_index == 0

        sim_slices = [
            file.simulations[sim_count // 5:2 * sim_count // 5],
            file.simulations[3 * sim_count // 5:4 * sim_count // 5]
        ]
        group = DataGroup(file.info.game, sim_slices)

        allowed_indices = set()
        for s in sim_slices:
            for p in s.positions:
                allowed_indices.add(p.file_pi)

        sampler = PositionSampler(group, 16, None, True, 1)
        for _ in range(32):
            batch = sampler.next_batch()

            for file_pi in batch.file_pi:
                assert int(file_pi) in allowed_indices
        sampler.close()

        sampler = PositionSampler(group, 16, 6, True, 1)
        for i in range(32):
            unrolled_batch = sampler.next_unrolled_batch()

            last_file_pis = [-1] * len(unrolled_batch)

            for batch in unrolled_batch.positions:

                for ni, file_pi in enumerate(batch.file_pi):
                    file_pi = int(file_pi)

                    if last_file_pis[ni] != -1:
                        assert int(file_pi) == -1 or file_pi == last_file_pis[ni] + 1
                    last_file_pis[ni] = file_pi
        sampler.close()

    assert count_covered == len(file.positions)
