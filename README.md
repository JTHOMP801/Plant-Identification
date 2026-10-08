# Pothos Plant Condition Classifier (Transfer Learning)

This project uses convolutional neural networks (CNNs) with transfer learning to
classify pothos plant images into four grouped condition categories. The dataset
was manually collected and labeled, then expanded through preprocessing and
augmentation.

## Classes (4)

Related issues were grouped into broader categories to make labeling consistent:

- **Healthy**
- **Leaf Spot** (viral, bacterial, or nutrient-deficiency symptom patterns)
- **Pests** (mealybugs, spider mites, aphids, scale insects, fungus gnats)
- **Root Conditions** (root rot / overwatering / edema)

## Dataset

424 manually collected and labeled images, after duplicate removal:

| class | images |
|---|---|
| Healthy | 133 |
| Leaf Spot | 112 |
| Root Conditions | 96 |
| Pests | 83 |

Split 70/15/15, then augmented. Only the training set is augmented, so the final
counts are not a uniform multiple of the originals:

| split | Healthy | Leaf Spot | Pests | Root | total |
|---|---|---|---|---|---|
| train | 465 | 390 | 287 | 332 | **1,474** |
| val | 19 | 16 | 12 | 14 | **61** |
| test | 21 | 18 | 13 | 15 | **67** |

Input size: 224×224

## Pipeline

```
deduplicated originals (424)
  └─ split 70/15/15 at the photo level, fixed seed
       └─ augment TRAIN ONLY — 5× per photo
            └─ contour crop + letterbox resize to 224×224
```

### Preprocessing

- Removed duplicate images using hashing
- Augmentation generates 4 variants per training image, plus the original:
  - horizontal flip
  - random rotation (±20°)
  - random crop/zoom (80–100%)
  - combined brightness and contrast jitter (±20% each)
- Cropping + resizing:
  - locate the plant region via contours, crop with 20% padding
  - resize and pad to 224×224 while preserving aspect ratio
- All output written as JPEG regardless of input format

### Why augmentation comes after the split

An earlier version of this pipeline augmented before splitting. Augmentation
writes 5 files per source photo, so shuffling the resulting flat file list
scattered copies of the same photo across train, validation and test — the model
was being evaluated on near-duplicates of its own training data. That version
reported **94.4% test accuracy, which was not a real number.**

Splitting on the originals first makes the leak structurally impossible. A
verification cell in `MakePlantDataset.ipynb` confirms it by stripping the
`_orig` / `_augN` suffixes from every filename and checking for any stem shared
between train and val/test. It reports zero on the current dataset.

A second bug surfaced during the fix. Keras `flow_from_directory` silently skips
any file whose extension is outside
`('png','jpg','jpeg','bmp','ppm','tif','tiff')`. Roughly 43% of the crawled
images were `.webp` and were never loaded into training at all — the generators
were reading 844/33/37 images instead of 1474/61/67. The crop step now writes
`.jpg` unconditionally.

## Models (Transfer Learning)

Compared three pretrained backbones: **ResNet50**, **EfficientNetV2B0**,
**MobileNetV2**.

Training flow:
1. Freeze backbone -> train a new classification head
2. Unfreeze part of the backbone -> fine-tune
3. Callbacks: EarlyStopping, ModelCheckpoint, learning-rate scheduling

**Implementation note:** an early ResNet50 run produced very low accuracy due to
incorrect input preprocessing (`rescale=1./255`). Switching to ResNet50's own
`preprocess_input` fixed it. Each backbone now uses its matching preprocessing
function, selected automatically by the generator factory.

## Results (Test Set)

| Model | Test Accuracy | Macro F1 | Test Loss | Params |
|---|---|---|---|---|
| ResNet50 | 0.746 | 0.728 | 1.144 | 24.1M |
| EfficientNetV2B0 | 0.702 | 0.683 | 0.830 | 6.25M |
| MobileNetV2 | 0.642 | 0.634 | 0.888 | 2.59M |

**Read these with the test set size in mind.** With 67 test images, a single
image is worth 1.5 accuracy points and the 95% confidence intervals span about
20 points (ResNet50: [0.63, 0.84]). The three backbones are separated by 3 and 7
images respectively — not a statistically meaningful margin. ResNet50 leads on
accuracy but carries the worst test loss, meaning it is more confidently wrong
than EfficientNetV2B0 at four times the parameter count.

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV (image preprocessing)
- NumPy / pandas
- scikit-learn (metrics + confusion matrix)
- Matplotlib (plots)
