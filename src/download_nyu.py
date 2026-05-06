import sys
from pathlib import Path

_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import io
import os
import tarfile

import datasets
import h5py
import numpy as np
import requests
from tqdm import tqdm

_BASE_URL = "https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main"
_VAL_TARS = ["data/val-000000.tar", "data/val-000001.tar"]
_CACHE_DIR = Path("data/nyu_v2/.cache")
_OUT_DIR = Path("data/nyu_v2/validation")


def download_file(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"  Already cached: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def iter_h5_from_tar(tar_path: Path):
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".h5"):
                raw = tf.extractfile(member).read()
                f = io.BytesIO(raw)
                with h5py.File(f, "r") as h5f:
                    rgb = np.transpose(np.array(h5f["rgb"]), (1, 2, 0))
                    depth = np.array(h5f["depth"])
                yield rgb, depth


def main():
    os.makedirs(_CACHE_DIR, exist_ok=True)

    print("Downloading validation TAR files...")
    tar_paths = []
    for rel in _VAL_TARS:
        url = f"{_BASE_URL}/{rel}"
        dest = _CACHE_DIR / Path(rel).name
        tar_paths.append(download_file(url, dest))

    print("\nBuilding dataset from H5 files...")
    images, depths = [], []
    for tar_path in tar_paths:
        print(f"  Processing {tar_path.name}...")
        for rgb, depth in iter_h5_from_tar(tar_path):
            images.append(rgb)
            depths.append(depth)

    print(f"\nConverting {len(images)} samples to HuggingFace Dataset...")
    ds = datasets.Dataset.from_dict({
        "image": [img for img in images],
        "depth_map": [d for d in depths],
    })

    print(f"Saving to {_OUT_DIR}...")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(_OUT_DIR))
    print(f"NYU V2 validation: {len(ds)} samples saved to {_OUT_DIR}")


if __name__ == "__main__":
    main()
