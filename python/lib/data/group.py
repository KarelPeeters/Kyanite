from typing import List

import numpy as np

from lib.data.file import FileSimulationsView, DataFile
from lib.data.position import Position, Simulation
from lib.games import Game


class DataGroup:
    def __init__(self, game: Game, sim_views: List[FileSimulationsView]):
        assert isinstance(sim_views, list)

        self.game = game
        self._sim_views = sim_views

        self._cum_len_sim = np.cumsum([len(s) for s in sim_views], )
        self._cum_len_position = np.cumsum([len(s.positions) for s in sim_views])

        self.simulations = GroupSimulationsView(self)
        self.positions = GroupPositionsView(self)

    def with_new_handles(self) -> 'DataGroup':
        return DataGroup(
            game=self.game,
            sim_views=[s.with_new_handles() for s in self._sim_views]
        )

    def close(self):
        for s in self._sim_views:
            s.file.close()

    @staticmethod
    def from_files(game: Game, files: List[DataFile], range_min: float = 0, range_max: float = 1) -> 'DataGroup':
        def pick_slice(f: DataFile):
            min = int(range_min * len(f.simulations))
            max = int(range_max * len(f.simulations))

            return f.simulations[min:max]

        views = [pick_slice(f) for f in files]
        return DataGroup(game, views)


class GroupSimulationsView:
    def __init__(self, group: DataGroup):
        self.group = group

    def __len__(self):
        return len_from_cum_sums(self.group._cum_len_sim)

    def __getitem__(self, index: int) -> Simulation:
        oi, ii = split_index(index, self.group._cum_len_sim)
        return self.group._sim_views[oi][ii]


class GroupPositionsView:
    def __init__(self, group: DataGroup):
        self.group = group

    def __len__(self):
        return len_from_cum_sums(self.group._cum_len_position)

    def __getitem__(self, index: int) -> Position:
        oi, ii = split_index(index, self.group._cum_len_position)
        return self.group._sim_views[oi].positions[ii]


def len_from_cum_sums(cum_sums: np.array) -> int:
    if len(cum_sums) == 0:
        return 0
    else:
        return cum_sums[-1]


def split_index(i: int, cum_sums: np.array) -> (int, int):
    length = len_from_cum_sums(cum_sums)

    if not (0 <= i < length):
        raise IndexError(f"Index {i} out of bounds for length {cum_sums[-1]}")

    outer = np.searchsorted(cum_sums, i, "right")
    if outer == 0:
        inner = i
    else:
        inner = i - cum_sums[outer - 1]

    return outer, inner
