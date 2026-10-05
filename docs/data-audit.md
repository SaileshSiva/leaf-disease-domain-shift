# Data audit

What is actually in the two datasets, checked file by file before any of it is used. Run on 5 October 2026.

Every number here comes from `scripts/audit_data.py`. Running it again reproduces the tables. It opens and fully decodes every image, takes each file's real format, size and colour mode from the image itself rather than from its name, and fingerprints each file (MD5) so identical copies can be found. A list with one row per file and a list of every duplicate group are written to `data/audit/`. That folder is not committed, so the script has to be rerun to get them back.

## Summary

- **Nothing is corrupt.** All 56,877 files open and decode.
- **The two datasets look completely different, even before a model sees them.** Every PlantVillage image is a 256×256 RGB photo. PlantDoc images range from 69 pixels to 6,000 pixels on a side, in 1,378 different sizes, and a few are CMYK, greyscale or have transparency. This is the lab-to-field gap showing up in the file statistics.
- **PlantDoc's labels are noisier than its folder names suggest.** The same photo appears under two different disease labels nine times. Five of those are potato early blight against potato late blight, which are classes this project needs.
- **PlantDoc's own train/test split leaks.** Eleven photos appear in both `train/` and `test/`.
- **PlantVillage has 21 photos saved twice** under different file names. If they are not removed, a copy could land in training and the other in the in-domain test set.
- **No photo appears in both datasets,** so the lab and field data are genuinely separate.

## Overview

| Dataset | Split | Classes | Files | Corrupt | Size (MB) |
|---|---|---|---|---|---|
| PlantVillage | all | 38 | 54,305 | 0 | 811 |
| PlantDoc | train | 28 | 2,336 | 0 | 879 |
| PlantDoc | test | 27 | 236 | 0 | 70 |

My counts differ slightly from the published figures: 54,305 against the 54,306 in Mohanty et al. (2016), and 2,572 against the 2,598 in Singh et al. (2020). I report my own counts and cite the published ones separately.

A PlantDoc image averages about 370 KB on disk, against about 15 KB for a PlantVillage image, because its photos are larger, higher-resolution JPEGs from the web.

## Images per class

### PlantVillage

PlantVillage ships as one folder per class with no split, so there is a single count per class.

| Class | Images |
|---|---|
| Apple___Apple_scab | 630 |
| Apple___Black_rot | 621 |
| Apple___Cedar_apple_rust | 275 |
| Apple___healthy | 1,645 |
| Blueberry___healthy | 1,502 |
| Cherry_(including_sour)___Powdery_mildew | 1,052 |
| Cherry_(including_sour)___healthy | 854 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 513 |
| Corn_(maize)___Common_rust_ | 1,192 |
| Corn_(maize)___Northern_Leaf_Blight | 985 |
| Corn_(maize)___healthy | 1,162 |
| Grape___Black_rot | 1,180 |
| Grape___Esca_(Black_Measles) | 1,383 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1,076 |
| Grape___healthy | 423 |
| Orange___Haunglongbing_(Citrus_greening) | 5,507 |
| Peach___Bacterial_spot | 2,297 |
| Peach___healthy | 360 |
| Pepper,_bell___Bacterial_spot | 997 |
| Pepper,_bell___healthy | 1,478 |
| **Potato___Early_blight** | **1,000** |
| **Potato___Late_blight** | **1,000** |
| **Potato___healthy** | **152** |
| Raspberry___healthy | 371 |
| Soybean___healthy | 5,090 |
| Squash___Powdery_mildew | 1,835 |
| Strawberry___Leaf_scorch | 1,109 |
| Strawberry___healthy | 456 |
| **Tomato___Bacterial_spot** | **2,127** |
| **Tomato___Early_blight** | **1,000** |
| **Tomato___Late_blight** | **1,909** |
| **Tomato___Leaf_Mold** | **952** |
| **Tomato___Septoria_leaf_spot** | **1,771** |
| **Tomato___Spider_mites Two-spotted_spider_mite** | **1,676** |
| **Tomato___Target_Spot** | **1,404** |
| **Tomato___Tomato_Yellow_Leaf_Curl_Virus** | **5,357** |
| **Tomato___Tomato_mosaic_virus** | **373** |
| **Tomato___healthy** | **1,591** |

Tomato and potato classes, the ones in scope, are in bold. PlantVillage has 10 tomato and 3 potato classes. They are unevenly sized: Yellow Leaf Curl Virus has 5,357 images and potato healthy has 152, a ratio of about 35 to 1. The split will need to be stratified, and macro F1 matters alongside accuracy.

### PlantDoc

PlantDoc comes already split into `train/` and `test/`.

| Class | Train | Test | Total |
|---|---|---|---|
| Apple Scab Leaf | 83 | 10 | 93 |
| Apple leaf | 82 | 9 | 91 |
| Apple rust leaf | 78 | 10 | 88 |
| Bell_pepper leaf | 53 | 8 | 61 |
| Bell_pepper leaf spot | 62 | 9 | 71 |
| Blueberry leaf | 104 | 11 | 115 |
| Cherry leaf | 47 | 10 | 57 |
| Corn Gray leaf spot | 64 | 4 | 68 |
| Corn leaf blight | 179 | 12 | 191 |
| Corn rust leaf | 106 | 10 | 116 |
| Peach leaf | 102 | 9 | 111 |
| **Potato leaf early blight** | **108** | **8** | **116** |
| **Potato leaf late blight** | **97** | **8** | **105** |
| Raspberry leaf | 112 | 7 | 119 |
| Soyabean leaf | 57 | 8 | 65 |
| Squash Powdery mildew leaf | 124 | 6 | 130 |
| Strawberry leaf | 88 | 8 | 96 |
| **Tomato Early blight leaf** | **79** | **9** | **88** |
| **Tomato Septoria leaf spot** | **140** | **11** | **151** |
| **Tomato leaf** | **55** | **8** | **63** |
| **Tomato leaf bacterial spot** | **101** | **9** | **110** |
| **Tomato leaf late blight** | **101** | **10** | **111** |
| **Tomato leaf mosaic virus** | **44** | **10** | **54** |
| **Tomato leaf yellow virus** | **70** | **6** | **76** |
| **Tomato mold leaf** | **85** | **6** | **91** |
| **Tomato two spotted spider mites leaf** | **2** | **0** | **2** |
| grape leaf | 57 | 12 | 69 |
| grape leaf black rot | 56 | 8 | 64 |

Several things stand out for the tomato and potato classes:

- **Tomato spider mites has only 2 images, both in train,** which is why `test/` has 27 classes and `train/` has 28. Two images cannot support a field test.
- **PlantDoc has no healthy potato class.** PlantVillage's healthy potato class therefore has nothing to be tested against in the field.
- **PlantDoc has no Target Spot class.** PlantVillage's Tomato Target Spot also has no field counterpart.
- **The test split is very small:** 6 to 11 images per tomato or potato class. A single misclassified image moves a class's accuracy by 9 to 17 percentage points.
- "Tomato leaf" is PlantDoc's name for a healthy tomato leaf, and should map to PlantVillage's `Tomato___healthy`.

These are inputs to the class mapping on Day 03, not decisions. The mapping is in `class-mapping.md`.

## File formats

| Dataset | Split | JPEG | MPO | PNG |
|---|---|---|---|---|
| PlantVillage | all | 54,304 | 0 | 1 |
| PlantDoc | train | 2,329 | 2 | 5 |
| PlantDoc | test | 236 | 0 | 0 |

Almost everything is JPEG. MPO is a JPEG variant written by some phone cameras. Pillow reads it as an ordinary JPEG.

Four PlantDoc files are named `.png.jpg` but are really PNGs, probably renamed when they were scraped:

- `train/Corn Gray leaf spot/2f73110f80014a25a53f9551c94bf164.png.jpg`
- `train/Corn rust leaf/SouthernRustLeaf.png.jpg`
- `train/Corn rust leaf/ppth-friskop-1-corn-rust.png.jpg`
- `train/Corn rust leaf/southernrust1.png.jpg`

They all decode correctly, and none are tomato or potato. No action needed.

## Resolutions

| Dataset | Split | Width (min / median / max) | Height (min / median / max) | Distinct sizes | Most common size |
|---|---|---|---|---|---|
| PlantVillage | all | 256 / 256 / 256 | 256 / 256 / 256 | 1 | 256×256 (100%) |
| PlantDoc | train | 115 / 800 / 6,000 | 85 / 665 / 6,000 | 1,378 | 1600×1200 (3%) |
| PlantDoc | test | 115 / 800 / 6,000 | 69 / 678 / 5,312 | 194 | 1024×768 (3%) |

PlantVillage is perfectly uniform: every image is 256×256. PlantDoc has no typical size at all, since even its most common size covers only 3% of images. PlantDoc images also come in many shapes; the median image is wider than it is tall.

This matters for preprocessing. Resizing to 224 pixels slightly shrinks PlantVillage images. For PlantDoc it means heavy downscaling of most images, upscaling of the smallest ones, and a choice about how to handle the aspect ratio (crop or pad). A model trained only on PlantVillage has never seen any of that variation.

## Colour modes

| Dataset | Split | RGB | RGBA | CMYK | Greyscale (L) |
|---|---|---|---|---|---|
| PlantVillage | all | 54,304 | 1 | 0 | 0 |
| PlantDoc | train | 2,325 | 5 | 5 | 1 |
| PlantDoc | test | 236 | 0 | 0 | 0 |

Twelve images are not plain RGB. The model expects three colour channels, so preprocessing must convert every image to RGB when it is loaded. Without that step, a CMYK image would arrive with four channels and the greyscale one with one.

## Corrupt files

None. Every file in both datasets opened and decoded fully.

## Duplicates

These are exact duplicates only: files whose bytes are identical. A resized or re-compressed copy of the same photo would not be caught, so the true number of duplicates is probably higher, especially in PlantDoc.

| Kind | Groups | Extra copies |
|---|---|---|
| All duplicate groups | 33 | 33 |
| Within one class only | 21 | 21 |
| Across classes (same photo, different labels) | 9 | 9 |
| Across PlantDoc train and test | 11 | 11 |
| Across PlantVillage and PlantDoc | 0 | 0 |

A group can fall under more than one kind. Most of the cross-class duplicates also cross the train/test boundary. Every group is a pair, two copies of one photo.

### PlantVillage: the same photo saved twice

All 21 within-class duplicates are in PlantVillage, in three classes:

| Class | Duplicate pairs |
|---|---|
| Tomato___Late_blight | 8 |
| Apple___healthy | 7 |
| Tomato___healthy | 6 |

In every pair, the two files have different random prefixes but the same original name, for example `…___GH_HL Leaf 389.JPG` appearing twice. It looks like the same photo was added to the dataset twice. On its own this is harmless. The risk is in splitting: a random split could put one copy in training and the other in the in-domain test set, and the model would be tested on a photo it has already seen. Fourteen of the pairs are in tomato classes. Duplicates must be removed before the split is made.

### PlantDoc: the same photo with two labels, or in both splits

| Photo | First copy | Second copy |
|---|---|---|
| `18028_1.jpg` / `24064_1.jpg` | train · Potato early blight | train · Potato late blight |
| `irish-blight-symptoms-on-potato-leaves-atmf8b.jpg` | train · Potato early blight | test · Potato late blight |
| `backus-056-potato-blight.jpg` | train · Potato late blight | test · Potato early blight |
| `5816740026_d42ef24413_Phytophthora-Infestans.jpg` | train · Potato early blight | test · Potato late blight |
| `1421_0.jpeg?itok=FMtmgePj.jpg` | train · Potato early blight | test · Potato late blight |
| `tomato_V8.jpg` | train · Tomato Septoria leaf spot | test · Tomato bacterial spot |
| `IMG_42231.jpg` | train · Corn leaf blight | test · Corn gray leaf spot |
| `2015070295153021.jpg` | train · Corn gray leaf spot | test · Corn leaf blight |
| `corn-gray-leaf-spot-f4.jpg` | train · Corn gray leaf spot | test · Corn leaf blight |
| `early-blight-septoria-ls-fig-3.jpg` | train · Tomato Septoria leaf spot | test · Tomato Septoria leaf spot |
| `tylcv-seminar-1-638.jpg` | train · Tomato yellow virus | test · Tomato yellow virus |
| `blueberry-leaves-normal-above-and-iron-deficient-below-bgahf8.jpg` | train · Blueberry leaf | test · Blueberry leaf |

The first nine rows are the **same photo carrying two different labels**, so at most one of them can be right. Five of those are potato early blight against late blight, which are two of the classes I need. Two of them give themselves away by name: `irish-blight-…` and `…Phytophthora-Infestans` refer to late blight (*Phytophthora infestans* is the late blight pathogen), yet both sit in the early blight folder in `train/`. So the early blight folder contains at least some late blight. This is direct evidence that PlantDoc's labels need the hand check planned for Day 04.

The last three rows are the same photo with the same label in both `train/` and `test/`. That is leakage. A model fine-tuned on PlantDoc's `train/` would be scored on photos it trained on.

Eleven of PlantDoc's 236 test images (4.7%) are copies of training images. Together with the small test set, this is a strong reason not to use PlantDoc's own split. Instead, pool `train/` and `test/`, remove duplicates and resolve conflicting labels, then make my own split. That decision belongs to Day 05 and will be recorded in the decision log.

## What this means for the next steps

| Step | What the audit says |
|---|---|
| Class mapping (Day 03) | Tomato spider mites (2 images), healthy potato (no PlantDoc class) and Target Spot (no PlantDoc class) cannot be tested in the field. "Tomato leaf" is the healthy tomato class. |
| Label check (Day 04) | Start with potato early and late blight. Five photos already carry both labels, and at least two are named after late blight but filed as early blight. |
| Splits (Day 05) | Remove exact duplicates in both datasets before splitting. Pool PlantDoc's train and test folders instead of using its split. Stratify by class, because class sizes vary 35-fold. |
| Preprocessing (Day 06) | Convert every image to RGB. Decide between cropping and padding for PlantDoc's non-square images. |
