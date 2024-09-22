# License Plate Recognition Project


## Dataset

The dataset can be accessed and downloaded from the following link:

[License Plate Recognition Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/2)

## How to Download the Dataset

To download the dataset in YOLOv8 format, use the following Python code:

```python
from roboflow import Roboflow

# Initialize Roboflow with API key
rf = Roboflow(api_key="vE7RwiS9tLY3EpjkMbJ6")

# Access the project and its version
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(2)

# Download the dataset in YOLOv8 format
dataset = version.download("yolov8")
