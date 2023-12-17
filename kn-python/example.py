import kyanite
import numpy as np
import os
import time
import unittest
from onnx.backend.base import Backend, BackendRep
from onnx.backend.test import BackendTest


def simple():
    path = r"C:\Documents\Programming\STTT\kZero\data\networks\chess_16x128_gen3634.onnx"
    with open(path, "rb") as f:
        onnx_bytes = f.read()
    graph = kyanite.Graph.from_bytes(onnx_bytes).optimize()

    input = np.zeros((1024, 21, 8, 8), dtype=np.float32)

    batch_size = graph.infer_batch_size([input])
    print(f"Batch size: {batch_size}")

    prepared = graph.prepare("cuda", batch_size)

    for _ in range(10):
        # TODO use real inputs
        start = time.time()
        prepared.eval([input])
        print("Time: ", time.time() - start)


class KyaniteBackendRep(BackendRep):
    def __init__(self, graph: kyanite.Graph, device):
        self.graph = graph
        self.device = device
        # TODO cache prepared model, reuse if batch size matches

    def run(self, inputs):
        batch_size = self.graph.infer_batch_size(inputs)
        prepared = self.graph.prepare(self.device, batch_size)
        return prepared.eval(inputs)


class KyaniteBackend(Backend):
    @classmethod
    def prepare(cls, model, device: str = "cpu") -> KyaniteBackendRep:
        onnx_bytes = model.SerializeToString
        graph = kyanite.Graph.from_bytes(onnx_bytes()).optimize()
        return KyaniteBackendRep(graph, device)


class DummyBackend(Backend):
    next: int = 0

    @classmethod
    def prepare(cls, model, device: str = "cpu") -> None:
        os.makedirs("models", exist_ok=True)
        with open(f"models/{DummyBackend.next}.onnx", "wb") as f:
            f.write(model.SerializeToString())
        DummyBackend.next += 1
        return None


def backend():
    # backend_test = BackendTest(KyaniteBackend, "KyaniteBanckendTest")
    backend_test = BackendTest(DummyBackend, "DummyBackendTest")
    globals().update(backend_test.enable_report().test_cases)
    unittest.main()


def main():
    # simple()
    backend()


if __name__ == "__main__":
    main(),
