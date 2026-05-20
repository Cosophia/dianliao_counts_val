from datetime import datetime
from enum import StrEnum
from operator import attrgetter
from typing import Iterator, NamedTuple

FOLDER = r"D:\AI训练平台\点料项目\dianliao_count_val"
MODEL_FOLDER = r"D:\AI训练平台\点料项目\dianliao_count_val"
MODEL_NAMES = {
    's_model': 'dianliao-S.onnx',
    'l_model': 'dianliao-B.onnx',
}

class MaterialSize(StrEnum):
    S = 'S'
    M = 'M'
    L = 'L'
    E = 'XL'

    @classmethod
    def values(cls) -> Iterator[str]:
        return map(attrgetter("value"), cls)


class SizeConfiguration(NamedTuple):
    scale: float
    nms_threshold: float
    conf: float


Size2Config = {
    MaterialSize.S: SizeConfiguration(1.80, 0.60, 0.60),
    MaterialSize.M: SizeConfiguration(1.30, 0.60, 0.60),
    MaterialSize.L: SizeConfiguration(1.00, 0.35, 0.45),
    MaterialSize.E: SizeConfiguration(0.45, 0.35, 0.45),
}


def is_date(x: str) -> bool:
    try:
        # 检查x输入的格式，正确应为如260101
        datetime.strptime(x, "%y%m%d")
    except ValueError:
        return False
    else:
        return True


