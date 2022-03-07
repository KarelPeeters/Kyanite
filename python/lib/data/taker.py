class Taker:
    def __init__(self, inner):
        self.inner = inner
        self.next = 0

    def take(self, n: int):
        self.next += n
        return self.inner[self.next - n:self.next]

    def finish(self):
        assert self.next == len(self.inner), f"Only read {self.next}/{len(self.inner)} bytes"
