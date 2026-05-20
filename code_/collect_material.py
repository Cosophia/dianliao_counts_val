import shutil
import sys
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from operator import attrgetter
from random import randint, randrange

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

# sys.path.insert(0, r"D:\\")
from utils import MaterialSize
from dataset import CountingDataset


class ImageSizeContainer:
    IMG_SIZE_PATH = Path(r"D:\tan\20251205-val\img_size.pkl")

    def __enter__(self):
        if self.IMG_SIZE_PATH.is_file():
            with open(self.IMG_SIZE_PATH, "rb") as f:
                d = pickle.load(f)
        else:
            d = {}

        self.d = d
        return d

    def __exit__(self, exc_type, exc_val, exc_tb):
        with open(self.IMG_SIZE_PATH, "wb") as f:
            pickle.dump(self.d, f)


def img_size_distribution(cls_type: MaterialSize | None):
    dataset = CountingDataset(folder=SRC_FOLDER, date="250101")
    if cls_type is not None:
        dataset.iter_type = cls_type

    wh = []
    ratio = []

    with ImageSizeContainer() as d:
        for file, *_ in tqdm(dataset):
            if (pair := d.get(file.name)) is None:
                img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                h, w = img.shape
                d[file.name] = w, h
            else:
                w, h = pair

            wh.append(np.array([w, h]))
            ratio.append(w / h)

    hw = np.stack(wh, axis=0)
    ratio = np.array(ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    _, _, _, im = ax1.hist2d(hw[:, 0], hw[:, 1], bins=30, cmap='viridis')
    ax1.set_xlabel('Width')
    ax1.set_ylabel('Height')
    fig.colorbar(im, ax=ax1, label='Count')

    ax2.hist(ratio)
    ax2.set_xlabel('Ratio')
    ax2.set_ylabel('Frequency')

    desc = "all classes" if cls_type is None else f"class {cls_type}"
    plt.suptitle(f"Distribution of height and weight on {desc}")

    plt.tight_layout()
    plt.show()


def collect_images(cls_type: MaterialSize | None, num: int, dst_folder: Path | None = None) -> None:
    dataset = CountingDataset(folder=SRC_FOLDER, date="250101")
    dataset.iter_type = cls_type

    cnt = 0
    cnter = Counter()
    is_dry_run = dst_folder is None

    target = []

    center = 864
    lo, hi = int(center * 0.75), int(center * 1.33)
    r0, r1 = 0.75, 1.33

    tmp = str(cls_type) if cls_type is not None else 'none'

    with tqdm(dataset, desc=tmp) as pbar, ImageSizeContainer() as d:
        for i, (file, *_) in enumerate(pbar, 1):
            if (pair := d.get(file.name)) is None:
                img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
                h, w = img.shape
                d[file.name] = w, h
            else:
                w, h = pair

            r = w / h

            if not lo <= w <= hi or not lo <= h <= hi or not r0 <= r <= r1:
                continue

            cnt += 1
            pbar.set_postfix(image_matched=cnt)

            cls = file.parent.name.split('_')[0]
            cnter[cls] += 1

            if cnt <= num:
                target.append(file)
                continue
            if (t := randrange(cnt)) < num:
                target[t] = file

    cnter2 = Counter()
    for file in target:
        cnter2[file.parent.name.split('_')[0]] += 1

    if is_dry_run:
        print(cnter, cnt)
        print(cnter2, len(target))
        # import numpy as np
        # print(np.fromiter(cnter.values(), dtype=np.int32) / cnt)
        # print(np.fromiter(cnter2.values(), dtype=np.int32) / len(target))
        return

    dst_folder.mkdir(parents=True, exist_ok=True)
    for cls in cnter:
        (dst_folder / cls).mkdir(parents=True, exist_ok=True)

    for file in target:
        folder = dst_folder / file.parent.name.split('_')[0]
        if not (folder / file.name).is_file():
            shutil.copy(file, folder)

    print(cnter, cnt)
    print(cnter2, len(target))


def stripper():  # remove the legacy folder naming with 5-digit-id prefix
    for x in SRC_FOLDER.iterdir():
        for y in x.iterdir():
            if y.name.count('_') == 1:
                continue

            v = y.name[y.name.find('_') + 1:]
            z = y.parent / v

            if z.is_dir():
                for file in y.iterdir():
                    shutil.move(file, z)
                y.rmdir()
            else:
                y.rename(z)


def _fetch_all_images():
    for x in SRC_FOLDER.iterdir():
        if not x.is_dir():
            continue
        for y in x.iterdir():
            for z in y.iterdir():
                yield z


def find_duplicate():
    d = defaultdict(list)
    it = _fetch_all_images()

    for z in it:
        d[z.stem].append(str(z.parents[0].relative_to(SRC_FOLDER)))

    to_del = [k for k, v in d.items() if len(v) == 1]
    for ele in to_del:
        d.pop(ele)

    print(sum(map(len, d.values())) - len(d), end='\n\n')
    print(tabulate(d.items(), headers=["image", "folder"]))


def _is_correct_naming(p: Path):
    def is_hex_prefix(s):
        try:
            int(s, base=16)
        except ValueError:
            return False
        return True

    t = p.stem
    return t.find('_')== 4 and is_hex_prefix(t[:4])


def find_error_naming():
    error_naming = {
        x.stem: x.parents[0].relative_to(SRC_FOLDER)
        for x in _fetch_all_images()
        if not _is_correct_naming(x)
    }

    print(len(error_naming), end='\n\n')
    print(tabulate(error_naming.items(), headers=["image", "folder"]))


def find_missing_id(r: int):
    s = {
        int(x.stem[:4], base=16)
        for x in _fetch_all_images()
        if _is_correct_naming(x)
    }

    missing = [x for x in range(r) if x not in s]
    print(len(missing), end='\n\n')

    for ele in missing:
        print(f"{ele:04X}")


if __name__ == '__main__':

    SRC_FOLDER = Path(r"D:\tan\20251205-val")

    # img_size_distribution(None)

    # p = Path(r"E:\\collect_material")
    # collect_images(None, 1200)

    # print(sum(1 for _ in _fetch_all_images()))
    # find_duplicate()
    # find_error_naming()
    find_missing_id(0x0D21)
