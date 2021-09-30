import asyncio

import numpy as np


async def append_data_to(array: np.array, index: int, data: int):
    await asyncio.sleep(1)
    array[index] = data


async def append_all(array: np.array):
    await asyncio.wait([
        asyncio.create_task(append_data_to(array, i, i)) for i in range(len(array))
    ])


def main():
    array = np.zeros(20)
    asyncio.run(append_all(array))
    print(array)


if __name__ == '__main__':
    main()
