# Pets-Face-Recognition

## Download the datasets
List of the datasets:
1. Oxford IIIT Pet (https://www.robots.ox.ac.uk/~vgg/data/pets/)
2. Cat Dataset (https://www.microsoft.com/en-us/research/wp-content/uploads/2008/10/ECCV_CAT_PROC.pdf)
3. Petfinder data

`python download_datasets.py`

The script downloads all the needed datasets to the directory ../pets_datasets

## Checkpoints

`python download_models.py`

## Training body detection (Mask R-CNN)

`python main_detection.py --config configs/to_reproduce/mask/mask_rcnn_config.py`

## Training Head and Landmark Detection (Keypoint R-CNN)

`python main_keypoints.py --config configs/to_reproduce/keypoint/keypoints_config.py`


# Testing detectors

You need to modify prepare_tables.py by providing appropriate preprocessing

`python prepare_tables.py` to get .tsv file for assessment

To test Head or Body detectors use:

`python score_detection.py {path to the .tsv} data_25 {Head|Animal}`

For landmark detection evaluation use:

`python score_landmark.py {path to the .tsv} data_25`


## Training Feature Extractors (FE)

### Prepare the datasets for training Feature Extraction

`python transform_reproduce.py`


### Training FE for Cats
Head-specific model

`python main_keypoints.py --config configs/to_reproduce/cat_fe/cat_fe_head.py`

Body-specific model

`python main_keypoints.py --config configs/to_reproduce/cat_fe/body_cat_fe.py`

### Training FE for Dogs
Head-specific model

`python main_keypoints.py --config configs/to_reproduce/dog_fe/fe_dogs_config.py`

Body-specific model

`python main_keypoints.py --config configs/to_reproduce/dog_fe/body_dog_fe.py`