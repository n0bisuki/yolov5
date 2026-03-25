from ultralytics import YOLO
from pathlib import Path
import pathlib

pathlib.PosixPath = pathlib.WindowsPath

model = YOLO("ai.pt")
# model = YOLO("ai.pt", task="detect")
model.export(format="engine", dynamic=True)  # fixed-size engine


# Run inference
model = YOLO("ai.engine", task="detect")
results = model.predict(r"H:\cheat\scripting\T-1.jpg", verbose=True)
results.show()
