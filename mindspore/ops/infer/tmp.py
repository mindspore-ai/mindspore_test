import os

files = os.listdir("/home/panzh/mindspore2/mindspore/ops/infer/")
files = [file for file in files if ".h" in file or ".cc" in file]

print(files)