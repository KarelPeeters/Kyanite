from typing import List

from lib.data.file import DataFile
from lib.data.position import Position


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

        actual_indices = [p.pi for p in matching_positions]
        assert list(sim.position_ids) == actual_indices

        for p in matching_positions:
            assert p.simulation == sim

        assert matching_positions[-1].is_final_position == file.info.includes_final_positions

    assert count_covered == len(file.positions)
