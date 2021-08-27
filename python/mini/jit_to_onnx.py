import glob
import os

import torch.jit

from train import save_onnx
from util import DEVICE

pattern = "../../data/derp/retrain_other/training/**/*.pt"

for path_pt in glob.glob(pattern):
    assert path_pt.endswith(".pt"), path_pt

    path_onnx = os.path.splitext(path_pt)[0] + ".onnx"

    print(f"Converting {path_pt} to {path_onnx}")
    model = torch.jit.load(path_pt, map_location=DEVICE)
    save_onnx(model, path_onnx)
