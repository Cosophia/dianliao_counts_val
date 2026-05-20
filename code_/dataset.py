import math
import random
from bisect import bisect_left, bisect_right
from collections import defaultdict
from functools import singledispatchmethod
from itertools import chain, pairwise
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tabulate import tabulate

from utils import MaterialSize, is_date


class CountingDataset:
    def __init__(self, folder: Path, date: int | str | tuple[str, str], *, constraint: float = 0.00):
        # 图片扩展名
        self.img_ext = 'png', 'bmp', 'jpg', 'jpeg'
        self.ruler = 0, 10, 100, 1000, 10000, float("inf")

        # 将传递过来的文件夹按日期排序
        folders = sorted(filter(lambda x: is_date(x.name), folder.iterdir()))
        # 获取子文件夹
        subfolders = self.get_subfolders(date, folders)

        assert len(subfolders) > 0, (f"Empty counting dataset, please make sure data is contained in "
                                     f"{folder} and check the param 'date'({date})")

        self._iter_type = None
        self.imgs: dict[Path, tuple[MaterialSize, int]] = {}
        # 获取图片信息
        for img in self.get_images(constraint, subfolders):
            order,category, cnt = img.name.split('_')
            cnts = cnt.split('.')[0]
            self.imgs[img] = MaterialSize(category.upper()), int(cnts)


        self.data: dict[MaterialSize, list[tuple[Path, int]]] = defaultdict(list)
        for path, category, cnt in self:
            if category not in MaterialSize:
                raise ValueError(f"Only categories {tuple(MaterialSize.values())} are allowed, got '{category}'.")
            self.data[category].append((path, cnt))

        self.folders = None
        print(self)

    # 泛型函数: 函数参数为泛型
    @singledispatchmethod
    def get_subfolders(self, date: Any, folders: list[Path]) -> list[Path]:
        raise TypeError(f"Unsupported type {type(date)} of date.")

    # 靠函数参数注解自动识别类型register
    # date : int | str | tuple[str, str]
    @get_subfolders.register
    def _(self, date: int, folders: list[Path]) -> list[Path]:
        assert date == -1 or 0 < date <= len(folders), \
            f"The number of folders should range from (0, {len(folders)}] or equal to -1, got {date}."
        return folders if date == -1 else folders[-date:]

    @get_subfolders.register
    def _(self, date: str, folders: list[Path]) -> list[Path]:
        assert is_date(date), f"The format of date should be 'yymmdd', got {date}."

        l = bisect_left(folders, date, key=attrgetter("name"))
        return folders[l:]

    @get_subfolders.register
    def _(self, date: tuple, folders: list[Path]) -> list[Path]:
        # ("XXX",) # ("XXX","XXX"),# ("XXX",)
        # assert len(date) == 2, f"When param date is tuple, it should be [start_date, end_date]."
        assert date[0] != "", f"When param date is tuple, it should be [start_date, end_date],got {date}."
        assert date[1] != "", f"When param date is tuple, it should be [start_date, end_date],got {date}."
        for x in date:
            assert is_date(x), f"The format of date should be 'yymmdd', got {x}."
        # 二分法：找到第一个大于等于date[0]的索引 和 最后一个小于等于date[1]的索引
        l = bisect_left(folders, date[0], key=attrgetter("name"))
        r = bisect_right(folders, date[1], key=attrgetter("name"))
        return folders[l: r]

    def get_images(self, constraint: float, subfolders: list[Path]) -> list[Path]:
        t = lambda f: chain.from_iterable(f.rglob(f"*.{x}") for x in self.img_ext)
        imgs = list(chain.from_iterable(map(t, subfolders)))
        cnt = prv_cnt = sum(1 for _ in imgs)

        print("\nConstructing dataset: ", end='')

        if constraint < 0:
            raise ValueError(f"Constraint should be >= 0, got {constraint}.")
        elif 0 < constraint < 1:
            cnt = math.ceil(prv_cnt * constraint)
            print(f"{cnt} images were sampled from {prv_cnt} with proportion {constraint:.2f} for analysing...")
        else:
            constraint = int(constraint)
            if constraint == 0 or prv_cnt <= constraint:
                print(f"Got {cnt} images for analysing...")
            else:
                cnt = constraint
                print(f"{cnt} images were sampled from {prv_cnt} for analysing...")

        return random.sample(imgs, cnt) if cnt < prv_cnt else list(imgs)

    def _overview(self):
        cnt_arr = list()
        folder2cnt = defaultdict(int)

        for path, _, cnt in self:
            cnt_arr.append(cnt)
            folder2cnt[path.parents[1].name] += 1

        time_info = sorted(folder2cnt.items())
        size_info = list(len(self.data[x]) for x in MaterialSize.values())

        indices = np.digitize(cnt_arr, self.ruler)
        range_info = np.array([np.sum(indices == i) for i in range(1, len(self.ruler))]).tolist()

        self.folders = sorted(folder2cnt.keys())
        return time_info, size_info, range_info

    @property
    def iter_type(self):
        return self._iter_type

    @iter_type.setter
    def iter_type(self, v: MaterialSize):
        if v is not None and v not in MaterialSize:
            raise ValueError(f"Unsupported iter type {v}, expected {tuple(MaterialSize.values())} or None.")
        self._iter_type = v

    def get_labels(self, v: MaterialSize):
        return np.fromiter(map(itemgetter(1), self.data[v]), dtype=np.int32)

    def __len__(self):
        match self._iter_type:
            case None:
                return len(self.imgs)
            case x if x in MaterialSize:
                return len(self.data[x])
            case _:
                raise ValueError(f"Unsupported iter type {type(self._iter_type)}.")

    def __iter__(self) -> Iterator[tuple[Path, MaterialSize, int]] | Iterator[tuple[Path, int]]:
        if self._iter_type in MaterialSize:
            yield from self.data[self._iter_type]
        elif self._iter_type is None:
            for path, (category, cnt) in self.imgs.items():
                yield path, category, cnt
        else:
            raise ValueError(f"Unsupported iter type {self._iter_type}.")

    def __repr__(self) -> str:
        total = len(self)
        time_info, category_info, range_info = self._overview()
        percentage = lambda x: f"{x / total:.2%}"

        total_number = f"Total samples used for counting analysis: {total}"

        dot = "Distribution over time\n" + tabulate(
            time_info,
            headers=("folder", "img cnt")
        )

        row1 = ["freq"] + category_info
        row2 = ["rel freq"] + list(map(percentage, category_info))
        doc = "Distribution over categories\n" + tabulate(
            (row1, row2),
            headers=chain(("",), MaterialSize.values()),
        )

        row1 = ["freq"] + range_info
        row2 = ["rel freq"] + list(map(percentage, range_info))
        dor = "Distribution over ranges of quantities\n" + tabulate(
            (row1, row2),
            headers=chain(("",), map(lambda p: f"[{p[0]}, {p[1]})", pairwise(self.ruler))),
        )

        return '\n\n'.join((
            "\n--- Info of counting dataset ---",
            total_number, dot, doc, dor, ''
        ))
