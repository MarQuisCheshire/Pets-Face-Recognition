from pathlib import Path

from torchvision.datasets.utils import download_and_extract_archive

urls = [
    "https://zenodo.org/record/6665690/files/to_reproduce.zip?download=1"
]


def download(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

    for url in urls:
        download_and_extract_archive(url, download_root=str(path), remove_finished=True, filename="to_reproduce.zip")


if __name__ == "__main__":
    download(Path("configs"))
