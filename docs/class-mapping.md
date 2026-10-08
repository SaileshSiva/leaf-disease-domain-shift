# Class mapping

Which PlantVillage classes have a counterpart in PlantDoc, how many images each pair has to work with, and which classes the project uses. Built on 8 October 2026 from the counts in [data-audit.md](data-audit.md).

Every number here comes from `scripts/class_overlap.py`, which reads `data/audit/images.csv` written by `scripts/audit_data.py`. The PlantVillage-to-PlantDoc pairing is defined once, in that script's `PAIRS` dictionary, so later scripts use the same mapping as this document. The inclusion rule is applied there too, as `MIN_FIELD_IMAGES`.

## Overlap by crop

Submission 01 restricted the project to tomato and potato because they were expected to be the crops with the most disease classes in both datasets. The audit confirms this. Tomato has 9 PlantVillage classes with a PlantDoc counterpart, and no other crop has more than 3. Tomato and potato together give 11 pairs.

| Crop | PlantVillage classes | With a PlantDoc pair |
|---|---|---|
| Tomato | 10 | 9 |
| Apple | 4 | 3 |
| Corn (maize) | 4 | 3 |
| Grape | 4 | 2 |
| Pepper, bell | 2 | 2 |
| Potato | 3 | 2 |
| Blueberry | 1 | 1 |
| Cherry | 2 | 1 |
| Peach | 2 | 1 |
| Raspberry | 1 | 1 |
| Soybean | 1 | 1 |
| Squash | 1 | 1 |
| Strawberry | 2 | 1 |
| Orange | 1 | 0 |

## Tomato and potato class pairs

PlantDoc counts pool its `train/` and `test/` folders, since its own split is too small and leaks (see the audit). Each count is given three ways:

- **raw:** files on disk.
- **unique:** distinct photos, with exact duplicates inside the class counted once.
- **unconflicted:** unique photos that do not also appear under a different PlantDoc label.

The unconflicted count is the most conservative estimate of what each class can contribute before the Day 04 label check.

| PlantVillage class | PV unique | PlantDoc class | PD raw | PD unique | PD unconflicted | Kept |
|---|---|---|---|---|---|---|
| Tomato___Septoria_leaf_spot | 1,771 | Tomato Septoria leaf spot | 151 | 150 | 149 | yes |
| Potato___Early_blight | 1,000 | Potato leaf early blight | 116 | 116 | 111 | yes |
| Tomato___Late_blight | 1,901 | Tomato leaf late blight | 111 | 111 | 111 | yes |
| Tomato___Bacterial_spot | 2,127 | Tomato leaf bacterial spot | 110 | 110 | 109 | yes |
| Potato___Late_blight | 1,000 | Potato leaf late blight | 105 | 105 | 100 | yes |
| Tomato___Leaf_Mold | 952 | Tomato mold leaf | 91 | 91 | 91 | yes |
| Tomato___Early_blight | 1,000 | Tomato Early blight leaf | 88 | 88 | 88 | yes |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5,357 | Tomato leaf yellow virus | 76 | 75 | 75 | yes |
| Tomato___healthy | 1,585 | Tomato leaf | 63 | 63 | 63 | yes |
| Tomato___Tomato_mosaic_virus | 373 | Tomato leaf mosaic virus | 54 | 54 | 54 | yes |
| Tomato___Spider_mites Two-spotted_spider_mite | 1,676 | Tomato two spotted spider mites leaf | 2 | 2 | 2 | no |
| Potato___healthy | 152 | — | 0 | 0 | 0 | no |
| Tomato___Target_Spot | 1,404 | — | 0 | 0 | 0 | no |

### Notes on the pairings

- **Healthy leaves.** PlantDoc names a healthy leaf by its crop alone, so "Tomato leaf" pairs with `Tomato___healthy`. PlantDoc has no "Potato leaf" class, so `Potato___healthy` has no pair.
- **"Tomato leaf yellow virus"** is paired with Tomato Yellow Leaf Curl Virus. It is the only yellowing virus among PlantVillage's tomato classes, and one of its files is named `tylcv-seminar-1-638.jpg` (TYLCV).
- **The potato blight counts drop by 5 each** once conflicted photos are removed. These are the five photos filed as both early and late blight. They are the first thing the Day 04 label check has to resolve.

## Inclusion rule

> A tomato or potato class is included if it exists in both datasets and PlantDoc has **at least 50 unconflicted photos** for it, after pooling `train/` and `test/`.

**Why "exists in both datasets".** The project's headline number is the gap between a class's score on held-out PlantVillage images and its score on PlantDoc field photos. A class with no PlantDoc images can be trained but never tested in the field, so it cannot contribute to that number. Keeping such a class would also mean the in-domain score covers more classes than the field score, and the gap would stop comparing like with like.

**Why 50.** Every included class's PlantDoc photos are split two ways: an untouched field test set, and a small slice for the field fine-tuning experiment. At 50 photos, a class can keep about 40 for testing, so one misclassified image moves that class's field accuracy by about 2.5 percentage points rather than 10 or more. That leaves about 10 for fine-tuning. The exact split proportions are decided on Day 05.

**The result does not depend on the exact threshold.** The smallest included class has 54 photos and the largest excluded one has 2, so any threshold from 3 to 54 selects the same 10 classes.

**Why "unconflicted".** Photos that PlantDoc files under two different labels are not counted, because at most one of those labels is right. The Day 04 label check can move counts either way. It may give some conflicted photos back to one class, and it may exclude photos found to be mislabelled. The class closest to the threshold is mosaic virus, at 54, so it is the one to recheck against the rule after Day 04.

## Class set

**10 classes: 8 tomato and 2 potato.** This is within the "ten to twelve" range given in Submission 01, which also said the class count would be reduced rather than the datasets changed if the overlap was smaller than expected.

| # | Class (PlantVillage folder name) | PlantDoc class | PD unconflicted |
|---|---|---|---|
| 1 | Tomato___Bacterial_spot | Tomato leaf bacterial spot | 109 |
| 2 | Tomato___Early_blight | Tomato Early blight leaf | 88 |
| 3 | Tomato___Late_blight | Tomato leaf late blight | 111 |
| 4 | Tomato___Leaf_Mold | Tomato mold leaf | 91 |
| 5 | Tomato___Septoria_leaf_spot | Tomato Septoria leaf spot | 149 |
| 6 | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato leaf yellow virus | 75 |
| 7 | Tomato___Tomato_mosaic_virus | Tomato leaf mosaic virus | 54 |
| 8 | Tomato___healthy | Tomato leaf | 63 |
| 9 | Potato___Early_blight | Potato leaf early blight | 111 |
| 10 | Potato___Late_blight | Potato leaf late blight | 100 |

The PlantVillage folder name is used as each class's identifier throughout the code. PlantDoc folders are translated to it through `PAIRS`.

## Excluded classes

| Class | PV unique | PD unconflicted | Reason |
|---|---|---|---|
| Tomato___Spider_mites Two-spotted_spider_mite | 1,676 | 2 | Two field photos cannot support a field test. One misclassification would move its accuracy by 50 points. |
| Tomato___Target_Spot | 1,404 | 0 | PlantDoc has no Target Spot class. |
| Potato___healthy | 152 | 0 | PlantDoc has no healthy potato class. |

All three are excluded because they cannot be measured in the field, not because they are unimportant.

**Excluding healthy potato has a deployment cost.** In practice, most leaves a farmer photographs are healthy. A model with no healthy potato class will label every healthy potato leaf as early or late blight. The project still tests healthy against diseased through `Tomato___healthy`, which stays in. Two other ways of keeping healthy potato were considered and rejected:

- **Train it on PlantVillage only.** It could never be scored in the field, and the in-domain and field scores would cover different class sets.
- **Take field images from another source.** Submission 01 rules out changing the datasets, and the new images would be unaudited and arrive mid-project.

This is one case of a wider limitation. The classifier is closed-set: any leaf outside these 10 classes, including the three excluded here, is forced into one of them. The report states this, and suggests field images of healthy potato and an "unknown" rejection step as future work. The inference application will say that it recognises only these 10 classes.

