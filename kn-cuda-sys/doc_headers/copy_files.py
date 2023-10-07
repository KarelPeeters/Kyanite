import subprocess
import re
import os
import shutil

# make sure the working dir is right
assert os.path.exists("Cargo.toml")

while True:
    result = subprocess.run(["cargo", "doc", "--no-deps", "--features", "docsrs"], capture_output=True)
    err = result.stderr.decode("utf-8")

    if m := re.search(r"fatal error: '(.*)' file not found", err):
        name = m.group(1)
        print(f"Missing file: {name}")

        cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include"
        include_root = "kn-cuda-sys/doc_headers/cuda_include"

        source = os.path.join(cuda_root, name)
        dest = os.path.join(include_root, name)

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(source, dest)
    else:
        break

print("No missing files!")
