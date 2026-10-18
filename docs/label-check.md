# PlantDoc label check

A by-eye check of the PlantDoc photos for the 10 classes in [class-mapping.md](class-mapping.md), done because PlantDoc's labels were scraped from the web rather than assigned by experts. PlantDoc is the field test set, so a wrong label there would be counted as a model error and would distort the lab-to-field gap, which is the project's headline number.

**Status:** potato done (18 October 2026). The 8 tomato classes are still to be checked.

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
| Potato leaf early blight | 116 | 52 | **64** |
| Potato leaf late blight | 105 | 16 | **89** |
| **Total** | 221 | 68 | 153 |

"Before" counts distinct photos, with train and test pooled. Both classes stay above the 50-photo inclusion threshold, so the class set does not change.

### Why photos were excluded

| Reason | Early blight | Late blight | What it means |
|---|---|---|---|
| `wrong_crop` | 18 | 8 | A tomato leaf filed under potato. |
| `other_disease` | 24 | 2 | The photo shows the other blight. |
| `in_other_folder` | 6 | 1 | The same photo is filed under both labels. This is the copy in the wrong folder. |
| `two_diseases` | 2 | 2 | One image shows two diseases. |
| `unusable` | 2 | 3 | Not a usable leaf photo, such as a slide, a video thumbnail, a multi-panel composite or a greyscale scan. |
| **Total** | **52** | **16** | |

### By strength of evidence

| Evidence | Early blight | Late blight |
|---|---|---|
| Filename, caption or annotation | 22 | 1 |
| Duplicate of a photo filed elsewhere | 6 | 2 |
| Visual only | 24 | 13 |

Without the visual-only calls, the classes would keep 88 and 102 photos. The duplicate row needs one caveat. That the photo is duplicated is a fact, but for 4 of the 8 photos filed under both labels, the choice of which copy to keep was itself a visual call (see finding 4). The visual-only calls are listed below so they can be reviewed one by one.

### Findings

1. **45% of the early blight folder is mislabelled or unusable** (52 of 116). It holds at least 24 late blight photos. Fourteen of them are named or captioned as late blight, and the source often says *Phytophthora infestans*. Another 18 photos are tomato leaves. Results reported on PlantDoc's potato early blight class without cleaning are therefore measured partly against wrong labels.
2. **The late blight folder is much cleaner,** at 15% excluded (16 of 105). Most of its errors are tomato leaves.
3. **Wrong crop is the largest single problem across both classes** (26 photos). A search for "early blight" or "late blight" returns tomato as readily as potato, because both crops get both diseases.
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
  - early blight: EB#20, 21, 27, 28, 32, 43, 48, 50, 60, 77, 79, 82
  - late blight: LB#2, 4, 28, 29, 37, 41, 44, 70
- **Late blight lesions in the early blight folder:** EB#9, 18, 25, 37, 38, 40, 53, 56, 63, 65.
- **Kept after review:**
  - EB#10 is a whole-plant photo, but early blight lesions are visible.
  - EB#111 is consistent with early blight.
  - EB#57 is a potato leaf, with filename `Potato-earlyblight2.jpg`, credited to Utah State University.

## Limitations

- The checker (me) is not a plant pathologist. Visual calls follow the textbook cues above, from photos of varying quality. The evidence column separates these calls from the stronger evidence so they can be weighed accordingly.
- Each photo was checked once, by one person, so no inter-rater agreement can be reported.
- Only the classes this project uses are checked. The rest of PlantDoc is untouched.
- Excluding photos makes the field test cleaner but also somewhat easier, because some of what was removed is the hardest, most ambiguous material. The report should say so next to the field results.
