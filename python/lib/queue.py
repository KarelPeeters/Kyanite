from collections import deque
from threading import Condition
from typing import Generic, TypeVar


class CQueueClosed(Exception):
    pass


T = TypeVar("T")


class CQueue(Generic[T]):
    def __init__(self, capacity: int):
        self.cond = Condition()
        self.closed = False

        self.capacity = capacity
        self.queue = deque()

    def check_open(self):
        if self.closed:
            raise CQueueClosed()

    def push_blocking(self, item: T):
        with self.cond:
            while True:
                self.check_open()
                if len(self.queue) < self.capacity:
                    self.queue.append(item)
                    self.cond.notify_all()
                    return
                self.cond.wait()

    def pop_blocking(self) -> T:
        with self.cond:
            while True:
                self.check_open()
                if len(self.queue):
                    result = self.queue.pop()
                    self.cond.notify_all()
                    return result
                self.cond.wait()

    def close(self):
        self.closed = True
        with self.cond:
            self.cond.notify_all()
