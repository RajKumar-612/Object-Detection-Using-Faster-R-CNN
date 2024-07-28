# Object Detection Using Faster R-CNN

This project implements an object detection system using a pre-trained Faster R-CNN model with ResNet-50 backbone. The system processes images from the VOC2007 dataset, detects objects, and evaluates the performance using metrics such as Intersection over Union (IoU), Precision, and Recall.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Functions](#functions)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Dependencies](#dependencies)

## Introduction

This project uses a Faster R-CNN model to detect objects in images from the VOC2007 dataset. The model is pre-trained on the COCO dataset and fine-tuned for detection tasks. The system applies Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes and evaluates the detections based on IoU, Precision, and Recall.

## Setup

1. Clone the repository and navigate to the project directory:

   ```sh
   git clone <repository-url>
   cd <project-directory>
   ```

2. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Download and extract the VOC2007 dataset:

   ```sh
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   tar -xvf VOCtrainval_06-Nov-2007.tar
   ```

## Usage

1. Run the script to process the images and evaluate the model:

   ```sh
   python object_detection.py
   ```

2. The script will process the first 100 images, display the first 5 images with bounding boxes, and print overall evaluation metrics.

## Functions

### `parse_annotation(xml_file)`

Parses an XML file to extract bounding box coordinates and labels.

### `calculate_iou(box_a, box_b)`

Calculates Intersection over Union (IoU) between two bounding boxes.

### `load_coco_classes(filename='coco_classes.txt')`

Loads COCO class names from a file.

## Evaluation Metrics

The system calculates the following metrics to evaluate the model's performance:

- **Average IoU**: Average Intersection over Union for true positives.
- **Precision**: Ratio of true positives to the sum of true positives and false positives.
- **Recall**: Ratio of true positives to the sum of true positives and false negatives.

## Results

The script displays the first 5 images with bounding boxes and prints overall evaluation metrics:

- **Average IoU**:
- **Precision**:
- **Recall**:

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- PIL
- matplotlib
- xml.etree.ElementTree
- os

Install dependencies using:

```sh
pip install -r requirements.txt
```
