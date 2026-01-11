Object Detection using Faster R-CNN (PASCAL VOC 2012)

##  Overview

This project implements an object detection system using Faster R-CNN trained on a subset of the PASCAL VOC 2012 dataset.
The model detects and localizes multiple objects in an image by predicting bounding boxes and class labels.

The implementation uses PyTorch, TorchVision, and Albumentations, and is trained and evaluated on Google Colab GPU.

##  Model Architecture

- Detector: Faster R-CNN
- Backbone: ResNet-50 + Feature Pyramid Network (FPN)
- Pretrained Weights: ImageNet (IMAGENET1K_V1)
- ROI Method: ROI Align

### Losses
- Classification loss
- Bounding box regression loss
- RPN objectness loss
- RPN box regression loss

##  Dataset

### Dataset Used

- PASCAL VOC 2012
- XML annotations parsed manually
- Images without target classes are filtered out

### Selected Classes
```
VOC_CLASSES = {
    "person": 1,
    "car": 2,
    "dog": 3,
    "cat": 4,
    "bottle": 5
}
```

Total classes = 5 foreground + 1 background

##  Data Processing

### Annotation Handling

For each image:

- Bounding boxes extracted from XML
- Labels mapped to class IDs
- Invalid samples safely skipped

Each sample returns:

```
{
  "boxes": Tensor[N, 4],
  "labels": Tensor[N],
  "area": Tensor[N],
  "iscrowd": Tensor[N]
}
```

##  Data Augmentation

Implemented using Albumentations with bounding-box safety.

### Training Augmentations

- Horizontal flip
- Random brightness & contrast
- Shift, scale, rotate
- Normalization
- Conversion to PyTorch tensors

### Validation Augmentations

- Resize
- Normalize
- Tensor conversion

##  Training Configuration

Parameter	Value

- Optimizer	SGD
- Learning Rate	0.0005
- Momentum	0.9
- Weight Decay	0.0005
- Batch Size	4
- Epochs	10
- LR Scheduler	StepLR
- Device	GPU (Colab)

##  Evaluation Metrics

Evaluation is done using TorchMetrics.

Metrics Used

- mAP@0.5 (IoU = 0.5)
- mAP@0.5:0.95 (COCO-style average)

##  Results

Metric	Value

- mAP@0.5	0.55
- mAP@0.5:0.95	0.24

### Observations

- Strong performance on dominant classes (person, car)
- Lower mAP@0.5:0.95 due to strict localization thresholds
- Minority class confusion observed (e.g., dog vs person in close proximity)

##  Training Time

- ~1 hour for 5 epochs on Colab GPU

Longer training improves accuracy but is compute-limited

##  Challenges Faced

- Class imbalance (person class dominates)
- Limited compute resources
- Long training times for Faster R-CNN
- Confusion between visually similar objects

##  Future Improvements

- Increase training epochs
- Class-balanced sampling
- Fine-tune anchor sizes
- Use COCO-pretrained Faster R-CNN
- Experiment with lighter backbones (MobileNet, ResNet-18)

##  Project Structure

```
├── demo_video
├── sample_input
├── Faster_R_CNN_Objects_Detection.ipynb
├── fasterrcnn_checkpoint.pth
├── requirements.txt
└── README.md
```

##  Requirements

- torch
- torchvision
- albumentations
- opencv-python
- matplotlib
- torchmetrics
- tqdm

##  Submission Includes

- Jupyter Notebook
- Training & validation loss plots
- mAP evaluation results
- Inference visualizations
- Screen recording of predictions

##  Conclusion

This project demonstrates a complete end-to-end object detection pipeline using Faster R-CNN on the PASCAL VOC dataset. Despite limited training time and class imbalance, the model achieves reasonable accuracy and produces visually consistent detections, making it suitable for academic and learning purposes.