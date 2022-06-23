# Pets-Face-Recognition

## Download the datasets
List of the datasets:
1. Oxford IIIT Pet (https://www.robots.ox.ac.uk/~vgg/data/pets/)
2. Cat Dataset (https://www.microsoft.com/en-us/research/wp-content/uploads/2008/10/ECCV_CAT_PROC.pdf)
3. Petfinder cats (https://zenodo.org/record/6656292#.Yq66DHZBwuU)
4. Petfinder dogs (https://zenodo.org/record/6660349#.Yq8TJHZBwuU)
5. Labelled data from Kashtanka.pet (https://zenodo.org/record/6664769#.Yq8GuXZBwuU)

`python download_datasets.py`

The script downloads all the needed datasets to the directory ../pets_datasets

## Checkpoints and our configs

To download the checkpoints and configs to use them run:

`python download_models.py`

## Detectors

Body detection and segmentation 

| Dataset | AP50 | AP70 | IoU detection | IoU segmentation | 
| ------- | ---------- | ---------- | ---------- | ---------- |
| Oxford IIIT Pets | 0.999 | 0.999 | 0.975 | 0.946 |
| Labelled kashtanka.pet dogs | 0.966 | 0.916 | 0.866 | N/A | 
| Labelled kashtanka.pet cats | 0.979 | 0.952 | 0.836| N/A |

Head and landmarks detection

| Dataset | AP50 | AP70 | IoU | NME | NME (Median) | NME percentile 0.25 | NME percentile 0.75|
| ------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Cat Dataset | 0.999 | 0.988 | 0.909 | 0.044 | - | - | - |
| Labelled kashtanka.pet dogs | 0.999 | 0.715 | 0.774 | 0.141 | 0.057 | 0.036 | 0.088 | 
| Labelled kashtanka.pet cats | 0.975 | 0.869 | 0.866 | 0.277 | 0.061 | 0.037 | 0.094 |

### Training body detection (Mask R-CNN)

`python main_detection.py --config configs/to_reproduce/mask/mask_rcnn_config.py`

### Training Head and Landmark Detection (Keypoint R-CNN)

`python main_keypoints.py --config configs/to_reproduce/keypoint/keypoints_config.py`


### Testing detectors

You need to modify prepare_tables.py by providing appropriate preprocessing if you want to test your models

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


### Generate .tsv for kashtanka.pet testing

Combination:

`python generate_tsv_to_reproduce1.py`

Only Face-based:

`python generate_tsv_to_reproduce2.py`

Pay attention you can modify the scripts and provide your own checkpoints