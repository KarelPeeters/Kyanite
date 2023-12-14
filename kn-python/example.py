import kyanite
import time

path = r"C:\Documents\Programming\STTT\kZero\data\networks\chess_16x128_gen3634.onnx"
with open(path, "rb") as f:
    onnx_bytes = f.read()
graph = kyanite.Graph.from_bytes(onnx_bytes)

rt = kyanite.Runtime("cuda")
batch_size = 1024
token = rt.prepare(graph, batch_size)

for _ in range(10):
    start = time.time()
    rt.eval(token)
    print("Time: ", time.time() - start)
