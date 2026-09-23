# Dataset

**Source:** UCI Machine Learning Repository — Human Activity Recognition Using Smartphones
DOI: 10.24432/C54S4K
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

**Frozen version:** the original 2012 release (`UCI HAR Dataset.zip`), not the updated
"Smartphone-Based Recognition of Human Activities and Postural Transitions" dataset.
This choice is frozen for the entire semester per the assignment's one-dataset rule.

## Use case (frozen at R0)

Predict a person's current physical activity (WALKING, WALKING_UPSTAIRS,
WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) from smartphone accelerometer +
gyroscope features. For HMM/CRF work in Part II, activity labels will be treated as an
ordered sequence per subject to exploit temporal continuity.

## Getting the data

1. Download `human+activity+recognition+using+smartphones.zip` from the URL above.
2. Unzip into `data/raw/`, then unzip the nested `UCI HAR Dataset.zip` so that
   `data/raw/UCI HAR Dataset/` contains `train/`, `test/`, `features.txt`,
   `activity_labels.txt`, `README.txt`.
3. `data/raw/` is gitignored (large, redistributable from source) — every teammate/grader
   re-downloads it rather than pulling it from Git history.

## Structure (as provided by UCI)

- `train/X_train.txt`, `train/y_train.txt`, `train/subject_train.txt` — 7352 samples, 21 subjects
- `test/X_test.txt`, `test/y_test.txt`, `test/subject_test.txt` — 2947 samples, 9 subjects
- `features.txt` — names of the 561 precomputed time/frequency-domain features
- `activity_labels.txt` — integer-to-label mapping (1=WALKING ... 6=LAYING)
- `*/Inertial Signals/` — raw 128-reading windows per axis, used for from-scratch
  feature/derivation work rather than the precomputed 561-feature vectors alone

## Split policy (frozen at R0)

The UCI-provided train/test split is already **subject-disjoint** (no subject appears in
both train and test), which satisfies the assignment's group/subject-aware leakage rule.
This split is kept sealed as the final test population; any validation/tuning uses a
subject-aware split carved out of the training subjects only (e.g. GroupKFold on
`subject_train.txt`).

## Primary metric (frozen at R0)

Macro-F1, with accuracy and confusion matrix reported as secondary evidence.

## Random seed (frozen at R0)

`SEED = 2452879` (student ID), used consistently across all experiments this semester.
