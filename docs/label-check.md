# PlantDoc label check

A by-eye check of the PlantDoc photos for the 10 classes in [class-mapping.md](class-mapping.md), done because PlantDoc's labels were scraped from the web rather than assigned by experts. PlantDoc is the field test set, so a wrong label there would be counted as a model error and would distort the lab-to-field gap, which is the project's headline number.

**Status:** complete. Potato checked on 18 October 2026 and revised on 22 November; all 8 tomato classes checked on 22 November 2026. Every photo in all 10 classes was checked, 963 in total.

The excluded files are listed one per row in [label-exclusions.csv](label-exclusions.csv). That file, not this document, is what the split script reads.

## Method

- **Every photo, not a sample.** The potato classes have 221 distinct photos between them, few enough to check them all.
- **Contact sheets.** `scripts/contact_sheets.py` tiles each PlantDoc class into numbered grids of 20 and writes an `index.csv` that maps each number to its file. PlantDoc's `train/` and `test/` folders are pooled and each distinct photo (by MD5) appears once. References such as EB#14 (early blight) and LB#62 (late blight) below are those numbers. They are stored in the CSV's `sheet_number` column, and rerunning the script reproduces them.
- **Near-duplicates.** The audit's MD5 check only finds byte-identical copies. A one-off perceptual hash (difference hash, 64 bits, at most 10 bits apart) was also run across the two potato classes to catch the same photo resized or re-saved. It found 3 photos filed under both labels that MD5 had missed. The hash is not yet part of the pipeline. Grouping near-duplicates belongs in the split script (Day 05).
- **Filenames and captions count as evidence.** Many PlantDoc files keep the name the source website gave them, and some carry a caption in the image itself. When the filename or caption names a different disease or crop, that is evidence independent of my own judgement.
- **Visual judgement, with the cues written down.**
  - **Early blight:** small dark-brown spots with concentric "target" rings, often held in by the leaf veins, often with a yellow halo.
  - **Late blight:** large irregular lesions, grey-green and water-soaked turning brown or black, often starting at the leaf tip or edge, with a pale border and sometimes white growth.
  - **Tomato or potato:** tomato leaflets are lobed with toothed edges. Potato leaflets are broader with smooth edges.

  The early and late blight cues apply to tomato as well as potato. The other tomato classes:

  - **Bacterial spot:** many small dark-brown to black spots, greasy or water-soaked at first, sometimes with a yellow halo. No rings and no pale centre. Spots may merge, and their centres may dry and tear out.
  - **Septoria leaf spot:** many small round spots with a grey or tan centre and a dark-brown border. The centre often holds tiny black dots (the fungus's fruiting bodies). Usually starts on the lower leaves.
  - **Leaf mold:** pale-green to yellow patches on the upper surface with soft, diffuse edges. Underneath, an olive-green to grey-brown velvety mould. Not a crisp spot.
  - **Mosaic virus:** a mottled pattern of light and dark green, often with puckered, distorted or narrow "fern-like" leaflets. No discrete lesions.
  - **Yellow leaf curl virus:** small leaflets cupped or curled upwards, with yellow edges, often on a stunted, bushy plant. No lesions.
  - **Healthy (PlantDoc's "Tomato leaf"):** evenly green, with no lesions, mottling, yellowing or curling. Torn edges or insect holes alone do not count as disease.

  **Pairs to watch.** Bacterial spot, Septoria and early blight all make small dark spots on tomato:
  - Septoria has a pale centre.
  - Early blight has rings.
  - Bacterial spot has neither.

  Mosaic virus and yellow leaf curl both distort the leaf. Mosaic mottles the green. Yellow leaf curl cups the leaflet and yellows its edges. When a photo is too small or blurred to separate a pair, it is kept, under the benefit-of-the-doubt rule below.

  Visual calls were made by me at full resolution. Doubtful ones were reviewed twice, and any photo that remained doubtful was given the benefit of the doubt and kept.
- **Each exclusion records its evidence.** The `evidence` column of the CSV says which kind of evidence it rests on: `filename`, `caption`, `annotation`, `duplicate` or `visual`.

## Policy: exclude, do not relabel

A photo found to be wrong is **excluded**, even when the evidence shows what the right label is. It is not moved to the other class.

- The field test set stays "PlantDoc minus documented errors", which is easier to defend than "PlantDoc with labels I assigned".
- Several calls rest on visual judgement alone. That is good enough to doubt a label but weaker grounds for writing a new one into a test set.
- Relabelling would not help where it matters. It would add 24 photos to potato late blight and only 2 to early blight, which is the smaller class after the check.

**The one exception is a photo filed under both labels.** When one of the two labels is judged correct, the copy in the correct folder is kept and the other copy is excluded. Nothing is relabelled, because the photo already sits in the right folder.

## Potato results

| PlantDoc class | Before | Excluded | Kept |
|---|---|---|---|
| Potato leaf early blight | 116 | 54 | **62** |
| Potato leaf late blight | 105 | 17 | **88** |
| **Total** | 221 | 71 | 150 |

"Before" counts distinct photos, with train and test pooled. Both classes stay above the 50-photo inclusion threshold, so the class set does not change.

**Revised 22 November.** Checking the tomato blight classes turned up three photos also filed under potato that the potato pass had kept: EB#26 (= TEB#30), EB#71 (= TEB#56) and LB#31 (= TLB#27). Seen next to their tomato copies, all three show tomato plants, so they are now excluded from potato as `wrong_crop`. The counts here include that change.

### Why photos were excluded

| Reason | Early blight | Late blight | What it means |
|---|---|---|---|
| `wrong_crop` | 20 | 9 | A tomato leaf filed under potato. |
| `other_disease` | 24 | 2 | The photo shows the other blight. |
| `in_other_folder` | 6 | 1 | The same photo is filed under both labels. This is the copy in the wrong folder. |
| `two_diseases` | 2 | 2 | One image shows two diseases. |
| `unusable` | 2 | 3 | Not a usable leaf photo, such as a slide, a video thumbnail, a multi-panel composite or a greyscale scan. |
| **Total** | **54** | **17** | |

### By strength of evidence

| Evidence | Early blight | Late blight |
|---|---|---|
| Filename, caption or annotation | 22 | 1 |
| Duplicate of a photo filed elsewhere | 6 | 2 |
| Visual only | 26 | 14 |

Without the visual-only calls, the classes would keep 88 and 102 photos. The duplicate row needs one caveat. That the photo is duplicated is a fact, but for 4 of the 8 photos filed under both labels, the choice of which copy to keep was itself a visual call (see finding 4). The visual-only calls are listed below so they can be reviewed one by one.

### Findings

1. **47% of the early blight folder is mislabelled or unusable** (54 of 116). It holds at least 24 late blight photos. Fourteen of them are named or captioned as late blight, and the source often says *Phytophthora infestans*. Another 20 photos are tomato leaves. Results reported on PlantDoc's potato early blight class without cleaning are therefore measured partly against wrong labels.
2. **The late blight folder is much cleaner,** at 16% excluded (17 of 105). Most of its errors are tomato leaves.
3. **Wrong crop is the largest single problem across both classes** (29 photos). A search for "early blight" or "late blight" returns tomato as readily as potato, because both crops get both diseases.
4. **8 photos are filed under both labels**, against the 5 that the audit's MD5 check found. The extra three are re-saved copies: EB#16 = LB#9, EB#29 = LB#35 and EB#93 = LB#88. Each was resolved to one label, as follows:

   | Photo | Kept as | On what evidence |
   |---|---|---|
   | EB#15 = LB#1 (`1421_0.jpeg?itok=FMtmgePj.jpg`) | early blight | visual |
   | EB#2 = LB#57 (`backus-056-potato-blight.jpg`) | late blight | visual |
   | EB#16 = LB#9 (`14321627273_58d893c3e2_*.jpg`) | late blight | visual |
   | EB#19 = LB#11 (`18028_1.jpg` / `24064_1.jpg`) | late blight | visual |
   | EB#29 = LB#35 (`Late+blight+on+potato.jpg`) | late blight | filename |
   | EB#87 = LB#7 (`irish-blight-symptoms-on-potato-leaves-atmf8b.jpg`; Irish blight is late blight) | late blight | filename |
   | EB#93 = LB#88 (`…phytophora-infestans…a60hxg/A60HXG.jpg`) | late blight | filename |
   | EB#32 = LB#4 (`5816740026_d42ef24413_Phytophthora-Infestans.jpg`) | excluded from both | tomato leaf |

5. **Near-duplicates remain inside late blight.** LB#33 (`LB_potato_leaf_topside.jpg`) and LB#53 (`PotatoLateBlight2.jpg`) are the same photo and both are kept. The split must place them on the same side, or one will be tested against its own copy.

### Visual-only calls

These are listed so each can be challenged. The CSV gives the filenames.

- **Tomato, not potato:**
  - early blight: EB#20, 21, 26, 27, 28, 32, 43, 48, 50, 60, 71, 77, 79, 82
  - late blight: LB#2, 4, 28, 29, 31, 37, 41, 44, 70
- **Late blight lesions in the early blight folder:** EB#9, 18, 25, 37, 38, 40, 53, 56, 63, 65.
- **Kept after review:**
  - EB#10 is a whole-plant photo, but early blight lesions are visible.
  - EB#111 is consistent with early blight.
  - EB#57 is a potato leaf, with filename `Potato-earlyblight2.jpg`, credited to Utah State University.

## Tomato blight results

Checked on 22 November 2026, every photo, with the same method and policy as potato. TEB and TLB numbers refer to the contact sheets of `Tomato Early blight leaf` and `Tomato leaf late blight`.

| PlantDoc class | Before | Excluded | Kept |
|---|---|---|---|
| Tomato Early blight leaf | 88 | 15 | **73** |
| Tomato leaf late blight | 111 | 8 | **103** |
| **Total** | 199 | 23 | 176 |

Both classes stay well above the 50-photo threshold.

### Why photos were excluded

| Reason | Early blight | Late blight | What it means here |
|---|---|---|---|
| `wrong_crop` | 6 | 3 | A potato leaf filed under tomato. |
| `other_disease` | 2 | 3 | The photo shows the other blight. |
| `in_other_folder` | 4 | 0 | The same photo is filed under another class. This is the copy in the wrong folder. |
| `unusable` | 3 | 2 | A presentation slide, a botanical drawing, or a two-panel composite. |
| **Total** | **15** | **8** | |

### By strength of evidence

| Evidence | Early blight | Late blight |
|---|---|---|
| Filename or caption | 3 | 4 |
| Duplicate of a photo filed elsewhere | 3 | 0 |
| Visual only | 9 | 4 |

Without the visual-only calls, the classes would keep 82 and 107 photos.

### Findings

1. **The tomato blight folders are much cleaner than potato early blight**, at 17% excluded for early blight (15 of 88) and 7% for late blight (8 of 111).
2. **Wrong crop runs both ways.** Potato leaves sit in the tomato folders (9 photos) just as tomato leaves sit in the potato folders. 5 of the 9 were found because the same photo is also filed under a potato class.
3. **Photos are shared across crops and diseases.** A perceptual hash (the same difference hash as for potato, at most 10 bits apart) across all 10 classes found 16 photos in the tomato blight folders that also appear in another class: 13 shared with a potato class, 2 with Septoria, and 1 shared between the two tomato blights. One photo is filed under all three of TLB, potato early blight and potato late blight. Each copy was resolved to one class, as follows:

   | Photo | Kept as | On what evidence |
   |---|---|---|
   | TEB#12 = LB#54 (`Symptom_potato_late_blight.jpg`) | potato late blight | filename |
   | TEB#25 = TEB#86 = EB#27 | tomato early blight | already excluded from potato as tomato |
   | TEB#30 = EB#26 | tomato early blight | visual |
   | TEB#35 = TSP#54 | Septoria | visual |
   | TEB#47 = TSP#79 | Septoria | visual |
   | TEB#56 = EB#71 | tomato early blight | visual |
   | TEB#61 = EB#74 | tomato early blight | already excluded from potato (filename says tomato) |
   | TEB#68 = EB#5 (`earlyblightpotato.jpg`) | potato early blight | filename |
   | TEB#71 = EB#76 (`early-blight-potatoes.jpg`) | potato early blight | filename |
   | TEB#76 = TLB#110 (`tomato_late-blight_04_zoom.jpg`) | tomato late blight | filename |
   | TLB#27 = LB#31 | tomato late blight | visual |
   | TLB#29 = EB#38 | excluded from both | potato leaf, late blight |
   | TLB#44 = EB#43 (`Early-Blight-Leaves-….jpg`) | excluded from both | tomato leaf, early blight |
   | TLB#54 = EB#63 (`backus-045-potato-blight.jpg`) | excluded from both | potato leaf, late blight |
   | TLB#92 = LB#32 (`Jessore-potato-late-blight-….jpg`) | potato late blight | filename |
   | TLB#97 = EB#32 = LB#4 | tomato late blight | already excluded from potato as tomato |

   "Excluded from both" means the photo is wrong in each folder it sits in. The right class exists (for example potato late blight for TLB#29), but under the exclude-do-not-relabel policy it is not moved there. The two Septoria copies are to be confirmed in the Septoria check.

4. **Near-duplicates remain inside each class.** The split must keep each group on one side:
   - early blight: TEB#19 = 62, 25 = 86, 27 = 29, 49 = 81
   - late blight: TLB#2 = 65, 12 = 25, 17 = 39, 51 = 91, 72 = 73 (a crop of the same photo, found by eye; the hash missed it)

### Visual-only calls

- **Potato, not tomato:** TEB#32, 33, 43, 44; TLB#29.
- **Late blight lesions in the early blight folder:** TEB#75 (diffuse blotches, no rings).
- **Early blight lesions in the late blight folder:** TLB#23 (ringed spots with yellow halos).
- **Septoria, not early blight:** TEB#35, 47 (many small scattered spots, no rings).
- **Unusable:** TEB#67, 82; TLB#42, 93.
- **Kept after review:**
  - TLB#90 and 96 have ringed spots that resemble early blight, but both filenames name late blight, and on a second look they are consistent with it.
  - TLB#79 shows no clear lesion at thumbnail size but is consistent with late blight at full resolution.
  - TLB#16 is named `7262-drought.jpg`, but its lesion is consistent with late blight.

## Other tomato results

Checked on 22 November 2026, every photo, with the same method and policy. Prefixes: TBS bacterial spot, TSP Septoria leaf spot, TLM leaf mold, TMV mosaic virus, TYV yellow leaf curl virus, TH healthy (`Tomato leaf`).

| PlantDoc class | Before | Excluded | Kept |
|---|---|---|---|
| Tomato leaf bacterial spot | 110 | 53 | **57** |
| Tomato Septoria leaf spot | 150 | 24 | **126** |
| Tomato mold leaf | 91 | 23 | **68** |
| Tomato leaf mosaic virus | 54 | 13 | **41** |
| Tomato leaf yellow virus | 75 | 15 | **60** |
| Tomato leaf (healthy) | 63 | 18 | **45** |
| **Total** | 543 | 146 | 397 |

**Mosaic virus and healthy tomato fall below the 50-photo threshold** once exclusions are applied. Both classes are kept; the reasoning is in [class-mapping.md](class-mapping.md) and in the decision log.

### Why photos were excluded

| Reason | TBS | TSP | TLM | TMV | TYV | TH |
|---|---|---|---|---|---|---|
| `wrong_crop` | 19 | 7 | 2 | 8 | 0 | 6 |
| `other_disease` | 22 | 4 | 16 | 1 | 8 | 8 |
| `in_other_folder` | 2 | 1 | 0 | 1 | 0 | 0 |
| `two_diseases` | 0 | 3 | 0 | 0 | 0 | 0 |
| `no_symptoms` | 0 | 0 | 1 | 1 | 0 | 0 |
| `unusable` | 10 | 9 | 4 | 2 | 7 | 4 |
| **Total** | **53** | **24** | **23** | **13** | **15** | **18** |

Two reasons are new in this batch or mean something specific here:

- `no_symptoms`: a photo filed under a disease that shows no symptoms of it.
- `other_disease` in the healthy class: the leaf shows symptoms, so it is not healthy. It also covers disorders that are not infections, such as herbicide injury, physiological leaf roll and edema, when the filename names them.

### By strength of evidence

| Evidence | TBS | TSP | TLM | TMV | TYV | TH |
|---|---|---|---|---|---|---|
| Filename or caption | 24 | 12 | 14 | 4 | 3 | 5 |
| Duplicate of a photo filed elsewhere | 0 | 0 | 0 | 1 | 0 | 0 |
| Visual only | 29 | 12 | 9 | 8 | 12 | 13 |

### Findings

1. **Bacterial spot is the noisiest tomato class**, with 48% excluded (53 of 110):
   - 21 photos are named as bacterial *speck* or bacterial *canker*. These are different bacterial diseases, and a web search for "bacterial spot" returns them readily.
   - 19 are other plants. Most are pepper, which bacterial spot also infects. The others are hibiscus, cannabis, begonia, dogwood, a brassica and a cucurbit.
2. **Look-alike diseases share a name.** Leaf mold holds 4 photos of *black* leaf mold (a different fungus), plus gray mold, gray leaf spot, Botrytis and powdery mildew. Septoria holds 2 gray leaf spot photos.
3. **Wrong crop is widespread outside the blights.** Mosaic virus has 8 photos of other plants: dahlia, luffa, tobacco, pelargonium, a cucurbit, an ornamental, pepper and a potato-like leaf. Healthy tomato has 6, mostly pepper.
4. **Slides and composites are common** (36 across the six classes). Web image search returns presentation slides and multi-panel figures as readily as photographs.
5. **Photos shared across classes** were resolved as follows:

   | Photo | Kept as | On what evidence |
   |---|---|---|
   | TBS#8 = TSP#140 (`tomato_V8.jpg`, filed under both labels by PlantDoc) | Septoria | visual |
   | TBS#46 = TSP#57 | Septoria | visual |
   | TBS#60 = TSP#74 (`Tomato bacterial spot closeup…`) | bacterial spot | filename |
   | TEB#35 = TSP#54, TEB#47 = TSP#79 | Septoria | visual (see the blight section) |
   | TSP#85 = potato EB#50 | Septoria | already excluded from potato as a tomato leaf |
   | TMV#22 = TYV#39 | yellow leaf curl | visual |
   | TYV#61 = TLM#91 | excluded from both | four-panel composite |
   | TSP#27 = TSP#118 | excluded from both | not tomato; the slide caption names *Septoria albopunctata*, a blueberry pathogen |
   | TSP#91 = TSP#150 | excluded from both | filename names gray leaf spot |

6. **Near-duplicates remain inside classes.** The split must keep each group on one side:
   - found by the hash: TBS#27 = 28, 67 = 76; TSP#38 = 90, 61 = 145, 132 = 133, 134 = 135; TYV#38 = 55
   - likely crops of the same photo, going by filename (`…-600x272` beside the full image), not yet confirmed: TSP#28/29, 32/33, 35/36, 40/41, 5/42

### Visual-only calls

- **Bacterial spot:**
  - wrong crop: TBS#3, 7, 10, 11, 12, 20, 21, 38, 39, 40, 41, 42, 62, 77, 90, 91, 97
  - unusable: TBS#1, 4, 36, 37, 50, 68, 72, 73, 84, 85
  - Septoria filed under bacterial spot: TBS#8, 46
- **Septoria:**
  - wrong crop: TSP#13, 14, 20, 103
  - unusable: TSP#1, 17, 21, 24, 59, 75, 98, 126
- **Leaf mold:**
  - not leaf mold: TLM#9, 14, 15, 72
  - no symptoms: TLM#81
  - unusable: TLM#3, 32, 33, 91
- **Mosaic virus:**
  - wrong crop: TMV#18, 23, 25, 26, 27, 34
  - no symptoms: TMV#40
  - unusable: TMV#6
- **Yellow leaf curl:**
  - yellowing without leaf curl: TYV#2, 7, 56, 57, 58
  - unusable: TYV#6, 30, 46, 49, 50, 61, 62
- **Healthy:**
  - symptoms present: TH#9, 18, 35, 45
  - wrong crop: TH#4, 21, 26, 27, 41
  - unusable: TH#2, 22, 31, 52
- **Kept after review:**
  - TSP#18 shows early Septoria.
  - TSP#85 is a tomato leaf.
  - TSP#89 is kept despite its filename (early blight with yellow halo), because the lesions are consistent with Septoria.
  - TLM#24 is consistent with leaf mold.
  - TH#28 is kept as tomato.

## Limitations

- The checker (me) is not a plant pathologist. Visual calls follow the textbook cues above, from photos of varying quality. The evidence column separates these calls from the stronger evidence so they can be weighed accordingly.
- Each photo was checked by one person, so no inter-rater agreement can be reported. For the tomato classes, an AI assistant made the first pass and proposed exclusions from the contact sheets and filenames; I reviewed every proposal and every doubtful keep, struck some proposals and added others. The decisions recorded here are mine.
- Only the classes this project uses are checked. The rest of PlantDoc is untouched.
- Excluding photos makes the field test cleaner but also somewhat easier, because some of what was removed is the hardest, most ambiguous material. The report should say so next to the field results.
