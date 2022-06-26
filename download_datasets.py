import sys
from pathlib import Path

from torchvision.datasets.utils import download_and_extract_archive, download_url

oxford_ds = (
    ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
    ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
)

cat_dataset = (
    "https://ia801607.us.archive.org/20/items/CAT_DATASET/CAT_DATASET_01.zip",
    "https://ia801607.us.archive.org/20/items/CAT_DATASET/CAT_DATASET_02.zip",
    "https://ia801607.us.archive.org/20/items/CAT_DATASET/00000003_015.jpg.cat"
)

data_25 = (
    "https://minio.k8s.grechka.family/public-shared-blobs/pet_data_25.tar.gz",
)

kashtanka_test = [
    "https://minio.k8s.grechka.family/public-shared-blobs/_blip_split_v3_public.tar.gz"
]

data_25_labeled = (
    "https://zenodo.org/record/6664769/files/data_25_labeled.zip?download=1",
)

pet_finder_cats = (
    "https://zenodo.org/record/6656292/files/petfinder_extra_cats1.zip?download=1",
    "https://zenodo.org/record/6656292/files/petfinder_extra_cats2.zip?download=1",
    "https://zenodo.org/record/6656292/files/petfinder_extra_cats3.zip?download=1",
    "https://zenodo.org/record/6656292/files/petfinder_extra_cats4.zip?download=1",
    "https://zenodo.org/record/6656292/files/petfinder_extra_cats5.zip?download=1",
    "https://zenodo.org/record/6656292/files/petfinder_extra_cats6.zip?download=1"
)

pet_finder_dogs = (
    "https://zenodo.org/record/6660349/files/petfinder_extra_dogs1.zip?download=1",
    "https://zenodo.org/record/6660349/files/petfinder_extra_dogs2.zip?download=1",
    "https://zenodo.org/record/6660349/files/petfinder_extra_dogs3.zip?download=1",
    "https://zenodo.org/record/6660349/files/petfinder_extra_dogs4.zip?download=1",
    "https://zenodo.org/record/6660349/files/petfinder_extra_dogs5.zip?download=1",
    "https://zenodo.org/record/6660349/files/petfinder_extra_dogs6.zip?download=1",
    "https://zenodo.org/record/6660349/files/petfinder_extra_dogs7.zip?download=1",
    "https://zenodo.org/record/6660349/files/petfinder_extra_dogs8.zip?download=1",
    "https://zenodo.org/record/6660349/files/petfinder_extra_dogs9.zip?download=1",
)


def download_oxford(path):
    if (path / "oxford-iiit-pet").exists():
        print("Skipping Oxford IIIT Pet")
    else:
        print("Downloading Oxford IIIT Pet")
        for url, md5 in oxford_ds:
            download_and_extract_archive(url, download_root=str(path / "oxford-iiit-pet"), md5=md5,
                                         remove_finished=True)


def download_cat_dataset(path):
    if (path / "CAT_DATASET").exists():
        print("Skipping Cat Dataset with landmarks")
    else:
        print("Downloading Cat Dataset with landmarks")
        for url in cat_dataset[:-1]:
            download_and_extract_archive(url, download_root=str(path / "CAT_DATASET"), remove_finished=True)
        # fix
    if not (path / "CAT_DATASET" / "CAT_00" / "00000003_015.jpg.cat").exists():
        download_url(cat_dataset[-1], str(path / "CAT_DATASET" / "CAT_00"), "00000003_015.jpg.cat", None)


def download_data_25(path):
    if (path / "data_25").exists():
        print("Skipping data_25")
    else:
        print("Downloading data_25")
        for url in data_25:
            download_and_extract_archive(url, download_root=str(path), remove_finished=True)


def download_kashtanka_test(path):
    if (path / "_blip_split_v3_public").exists():
        print("Skipping _blip_split_v3_public")
    else:
        print("Downloading _blip_split_v3_public")
        for url in kashtanka_test:
            download_and_extract_archive(url, download_root=str(path), remove_finished=True)


def download_data_25_labeled(path):
    if (path / "data_25_labeled").exists():
        print("Skipping data_25 _labelled")
    else:
        print("Downloading data_25 _labelled")
        for url in data_25_labeled:
            download_and_extract_archive(url, download_root=str(path), remove_finished=True,
                                         filename="data_25_labeled.zip")


def download_pet_finder_cats(path):
    if (path / "petfinder_extra_cats").exists():
        print("Skipping Petfinder cats")
    else:
        print("Downloading Petfinder cats")
        for i, url in enumerate(pet_finder_cats):
            download_and_extract_archive(url, download_root=str(path / "petfinder_extra_cats"), remove_finished=True,
                                         filename=f'petfinder_extra_cats{i + 1}.zip')


def download_pet_finder_dogs(path):
    if (path / "petfinder_extra_dogs").exists():
        print("Skipping Petfinder dogs")
    else:
        print("Downloading Petfinder dogs")
        for i, url in enumerate(pet_finder_dogs):
            download_and_extract_archive(url, download_root=str(path / "petfinder_extra_dogs"), remove_finished=True,
                                         filename=f'petfinder_extra_dogs{i + 1}.zip')


def download_all(path: Path):
    download_oxford(path)
    download_cat_dataset(path)
    download_data_25(path)
    download_data_25_labeled(path)
    download_kashtanka_test(path)
    download_pet_finder_cats(path)
    download_pet_finder_dogs(path)


download_options = {
    'oxford': download_oxford,
    'cat_dataset': download_cat_dataset,
    'data_25': download_data_25,
    'data_25_labeled': download_data_25_labeled,
    'kashtanka_test': download_kashtanka_test,
    'petfinder_dogs': download_pet_finder_dogs,
    'petfinder_cats': download_pet_finder_cats,
    'all': download_all
}


def main():
    p = Path("../pets_datasets")
    p.mkdir(parents=True, exist_ok=True)
    args = sys.argv
    c = 0
    for i in args:
        if i in download_options:
            download_options[i](p)
            c += 1
    if c == 0:
        download_all(p)


if __name__ == "__main__":
    main()
