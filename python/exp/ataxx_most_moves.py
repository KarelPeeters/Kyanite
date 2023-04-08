import itertools

from ortools.sat.python import cp_model


def singles(size, x, y):
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            rx = x + dx
            ry = y + dy
            if (dx != 0 or dy != 0) and 0 <= rx < size and 0 <= ry < size:
                yield rx, ry


def doubles(size, x, y):
    for dy in [-2, -1, 0, 1, 2]:
        for dx in [-2, -1, 0, 1, 2]:
            rx = x + dx
            ry = y + dy
            if (abs(dx) == 2 or abs(dy) == 2) and 0 <= rx < size and 0 <= ry < size:
                yield rx, ry


def max_moves_board_for_size(size: int):
    print(f"Size: {size}")
    model = cp_model.CpModel()
    vars = {}

    def new_bool(name: str):
        result = model.NewBoolVar(name)
        vars[name] = result
        return result

    def new_and(name: str, terms):
        result = new_bool(name)
        model.AddMinEquality(result, terms)
        return result

    def new_or(name: str, terms):
        result = new_bool(name)
        model.AddMaxEquality(result, terms)
        return result

    tiles = [[new_bool(f"tile_{x}_{y}") for y in range(size)] for x in range(size)]
    opponents = [[new_bool(f"opponent_{x}_{y}") for y in range(size)] for x in range(size)]

    for y in range(size):
        for x in range(size):
            if x > size / 2 or y > size / 2 or y > x:
                # disable non-triangle opponent tiles
                model.Add(opponents[x][y] == 0)
            else:
                # ensure no overlap
                model.AddAtMostOne(tiles[x][y], opponents[x][y])

    # ensure at least one opponent tile
    model.AddAtLeastOne(*(o for row in opponents for o in row))

    single_moves = []
    double_moves = []

    for y in range(size):
        for x in range(size):
            tile = tiles[x][y]
            opponent = opponents[x][y]

            # single move
            single_source = new_or(f"single_source_{x}_{y}", [tiles[rx][ry] for rx, ry in singles(size, x, y)])
            single = new_and(f"single_{x}_{y}", [single_source, 1 - tile, 1 - opponent])
            single_moves.append(single)

            # double moves
            for rx, ry in doubles(size, x, y):
                double = new_and(f"double_{x}_{y}_{rx}_{ry}", [tiles[rx][ry], 1 - tile, 1 - opponent])
                double_moves.append(double)

    model.Maximize(sum(single_moves) + sum(double_moves))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("  solution:")
        print(f"    status: {status}")
        for y in range(size):
            print("   ", end="")
            for x in range(size):
                if solver.Value(tiles[x][y]):
                    symbol = "x"
                elif solver.Value(opponents[x][y]):
                    symbol = "o"
                else:
                    symbol = "."
                print(f" {symbol}", end="")
            print()
        print(f"    moves: {solver.Value(sum(single_moves) + sum(double_moves))}")
        print(f"    singles: {solver.Value(sum(single_moves))}")
        print(f"    doubles: {solver.Value(sum(double_moves))}")
    else:
        print('  No solution found.')


def main():
    for size in itertools.count(2):
        max_moves_board_for_size(size)


if __name__ == '__main__':
    main()
