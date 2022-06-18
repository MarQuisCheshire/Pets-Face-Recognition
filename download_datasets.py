import os
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
    "https://transfer.sh/get/a0C1H3/data_25.zip"
)

pet_finder = (
)


def download(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)

    if (path / "oxford-iiit-pet").exists():
        print("Skipping Oxford IIIT Pet")
    else:
        print("Downloading Oxford IIIT Pet")
        for url, md5 in oxford_ds:
            download_and_extract_archive(url, download_root=str(path / "oxford-iiit-pet"), md5=md5, remove_finished=True)

    if (path / "CAT_DATASET").exists():
        print("Skipping Cat Dataset with landmarks")
    else:
        print("Downloading Cat Dataset with landmarks")
        for url in cat_dataset[:-1]:
            download_and_extract_archive(url, download_root=str(path / "CAT_DATASET"), remove_finished=True)
        # fix
        os.remove(str(path / "CAT_DATASET" / "CAT_00" / "00000003_015.jpg.cat"))
        download_url(cat_dataset[-1], str(path / "CAT_DATASET" / "CAT_00"), "00000003_015.jpg.cat", None)

    if (path / "data_25").exists():
        print("Skipping data_25")
    else:
        print("Downloading data_25")
        for url in data_25:
            download_and_extract_archive(url, download_root=str(path), remove_finished=True)

    if (path / "petfinder_extra_cats").exists() and (path / "petfinder_extra_dogs").exists():
        print("Skipping Petfinder")
    else:
        print("Downloading Petfinder")
        for url in pet_finder:
            download_and_extract_archive(url, download_root=str(path), remove_finished=True)


if __name__ == "__main__":
    download(Path("../pets_datasets"))
