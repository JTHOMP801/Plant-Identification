# Pothos Plant Condition Classifier (Transfer Learning)

This project uses convolutional neural networks (CNNs) with **transfer learning** to classify pothos plant images into **four** grouped condition categories. The dataset was **manually collected and labeled**, then expanded through preprocessing and augmentation. See My report pdf for a much more indepth analysis of this project's processes, metrics and more.

## Classes (4)
Related issues were grouped into broader categories to make labeling consistent:
- **Healthy**
- **Leaf Spot** (viral, bacterial, or nutrient-deficiency symptom patterns)
- **Pests** (mealybugs, spider mites, aphids, scale insects, fungus gnats)
- **Root Conditions** (root rot / overwatering / edema; overwatering-related)

## Dataset
- **Initial (manual):** 457 images  
  - 150 healthy, 120 leaf spot, 86 pests, 101 root conditions
- **After preprocessing + augmentation:** 2,112 images  
  - 665 healthy, 559 leaf spot, 412 pests, 476 root conditions
- **Input size:** 224×224
- **Split:** 70% train / 15% validation / 15% test (per class, randomized with a fixed seed)

## Preprocessing
- Removed duplicate images using hashing
- Augmentation: generated 4 variants per image using
  - random rotation (±20°)
  - random crop/zoom (80–100%)
  - contrast adjustment (±20%)
  - brightness adjustment (±20%)
- Cropping + resizing pipeline:
  - locate the plant region via contours, crop with padding
  - resize and pad to 224×224 while preserving aspect ratio

## Models (Transfer Learning)
Compared three pretrained backbones:
- **ResNet50**
- **EfficientNetV2B0**
- **MobileNetV2**

Training flow:
- Freeze backbone → train a new classification head
- Unfreeze part of the backbone → fine-tune
- Used callbacks such as EarlyStopping and ModelCheckpoint (and learning-rate scheduling)

Implementation note:
- An early ResNet50 run produced very low accuracy due to incorrect input preprocessing (`rescale=1./255`). Switching to ResNet50’s `preprocess_input` fixed performance.

## Results (Test Set)
| Model | Test Accuracy | Test Loss |
|------|---------------:|----------:|
| **ResNet50** | **94.44%** | 0.0938 |
| EfficientNetV2B0 | 93.43% | 0.2541 |
| MobileNetV2 | 92.42% | 0.2127 |

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV (image preprocessing)
- NumPy
- scikit-learn (metrics + confusion matrix)
- Matplotlib (plots)


└── README.md

