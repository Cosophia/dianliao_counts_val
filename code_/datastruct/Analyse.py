import pickle
import shutil
from collections import Counter
from itertools import pairwise, chain, repeat
from operator import itemgetter
from pathlib import Path
from typing import NamedTuple
import cv2
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from code_.dataset import CountingDataset
from code_.obb_rectified import ONNXInfer
from code_.utils import MaterialSize, Size2Config, is_date, MODEL_NAMES


class TypeInfo(NamedTuple):
    type: tuple[MaterialSize, ...]
    models: list[Path]
class Analyse:
    def __init__(self, folder: Path,model_folder: Path, date: int | str | tuple[str, str] , latest_model: int, *,
                 constraint: float = 0.00):

        s_model_name: str = MODEL_NAMES['s_model']
        l_model_name: str = MODEL_NAMES['l_model']

        folder = Path(folder) # 将windows路径类型转化为字符串
        # model_folder = Path(model_folder)
        self.dst_folder = folder / "result"

        self.latest_model = latest_model
        # self.model_folder = folder / "model"
        self.model_folder = model_folder/"model"
        self.model_folders = sorted(
            filter(lambda x: is_date(x.name.split('_', 1)[0]), self.model_folder.iterdir()),
            reverse=True
        )

        self.ruler = 0, 1, 3, 5, 10, float("inf")
        self.metrics = ["MAE", "RMSE", "rel_MAE", "error_cnt"]

        self.dataset = CountingDataset(folder / "data", date, constraint=constraint)
        #self.dataset = CountingDataset(folder , date, constraint=constraint)
        self.labels = {x: self.dataset.get_labels(x) for x in MaterialSize}

        self.type_info = {
            's': TypeInfo((MaterialSize.S, MaterialSize.M), self.fetch_model(s_model_name)),
            'l': TypeInfo((MaterialSize.L, MaterialSize.E), self.fetch_model(l_model_name)),
        }

        print("--- Info of counting models ---\n")
        print(tabulate(
            chain.from_iterable(map(
                lambda p: zip(map(lambda x: x.parents[0].name, p[1].models), repeat(p[0])),
                self.type_info.items()
            )),
            ["Folder", "Type"]
        ), end='\n\n')

        tmp = input("Please confirm the dataset and models (y/n): ")
        if tmp not in 'yY':
            exit(1)

        print("\n\n")
        self.run()

    def fetch_model(self, name: str) -> list[Path]:
        lst = []
        prv = 0.0

        for folder in self.model_folders:
            model = folder / name
            time = model.stat().st_mtime

            if time == prv:
                lst[-1] = model
            elif len(lst) == self.latest_model:
                break
            else:
                lst.append(model)
                prv = time

        # if len(lst) < self.latest_model:
        #     raise ValueError(f"The actual number of models ({len(lst)}) in {self.model_folder} "
        #                      f"is less than the given model cnt ({self.latest_model}).")

        lst.reverse()
        return lst

    def get_statistic(self, pd: np.ndarray, gt: np.ndarray) -> dict | None:
        if pd.size == 0:
            return None

        bias = pd - gt
        absolute_error = np.abs(bias)

        mae = absolute_error.mean()
        rel_mae = np.mean(absolute_error / gt) * 10000
        rmse = np.sqrt(np.square(bias).mean())

        indices = np.digitize(absolute_error, self.ruler, right=False)
        ae_freq = np.array([np.sum(indices == i) for i in range(1, len(self.ruler))])
        ae_cum_freq = np.cumsum(ae_freq)

        return {
            "MAE": mae,
            "rel_MAE": rel_mae,
            "RMSE": rmse,
            "ae_freq": ae_freq,
            "ae_cum_freq": ae_cum_freq,
        }

    def show_statistic(self, stats: list[dict], errors: list[dict], model_alias: tuple[str, ...],
                       samples: int, verbose: bool = False):
        if samples == 0:
            print("NO DATA\n")
            return

        n = len(model_alias)

        for i in range(n):
            stats[i]["error_cnt"] = sum(errors[i].values())

        print(tabulate(
            map(lambda i: [model_alias[i]] + list(stats[i][x] for x in self.metrics), range(n)),
            self.metrics,
            floatfmt='.3f'
        ), end='\n\n')

        print(tabulate(
            chain(
                (chain(("Total",), map(itemgetter("error_cnt"), stats)),),
                map(
                    lambda f: chain((f,), map(itemgetter(f), errors)),
                    self.dataset.folders
                ),
            ),
            model_alias
        ), end='\n\n')

        if not verbose:
            return

        ruler_header = [f"[{a}, {b})" for a, b in pairwise(self.ruler)]

        rows = map(lambda i: [model_alias[i]] + stats[i]["ae_freq"].tolist(), range(n))
        print(tabulate(rows, ["AE_freq"] + ruler_header), end='\n\n')
        rows = map(lambda i: [model_alias[i]] + stats[i]["ae_cum_freq"].tolist(), range(n))
        print(tabulate(rows, ["AE_cum_freq"] + ruler_header), end='\n\n')

        rows = map(lambda i: [model_alias[i]] + (stats[i]["ae_freq"] * 100 / samples).tolist(), range(n))
        print(tabulate(rows, ["AE_rel_freq %"] + ruler_header, floatfmt='.3f'), end='\n\n')
        rows = map(lambda i: [model_alias[i]] + (stats[i]["ae_cum_freq"] * 100 / samples).tolist(), range(n))
        print(tabulate(rows, ["AE_cum_rel_freq %"] + ruler_header, floatfmt='.3f'), end='\n\n')

    def analyse_single_model(self, model_path: Path, tpe: str):
        name = f"{model_path.parent.name}_{tpe}"
        dst_folder = self.dst_folder / name

        cache_path = dst_folder / "record.pkl"
        cache = pickle.load(open(cache_path, "rb")) if cache_path.is_file() else {}

        model = ONNXInfer(onnx_model=model_path)
        is_empty_dir = lambda t: t.is_dir() and next(t.iterdir()) is None

        def analyse_single_type(size: MaterialSize):
            local_dst_folder = dst_folder / size
            pred_folder   = local_dst_folder / "pred"
            scaled_folder = local_dst_folder / "scaled"
            origin_folder = local_dst_folder / "origin"

            pred_folder  .mkdir(parents=True, exist_ok=True)
            scaled_folder.mkdir(parents=True, exist_ok=True)
            origin_folder.mkdir(parents=True, exist_ok=True)

            self.dataset.iter_type = size

            scale, nms_threshold, conf = Size2Config[size]

            model.conf_thres = conf
            model.nms_thres = nms_threshold

            needs_write = False
            arr: list[int] = []
            folder2error = Counter()

            for path, gt in tqdm(self.dataset, desc=f"Analysing model {name} on material {size}"):

                if (pd := cache.get(path.stem)) is None:
                    scaled_img = cv2.resize(cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR), None, fx=scale, fy=scale)


                    needs_write = True
                    predictions = model.predict(scaled_img)
                    cache[path.stem] = pd = len(predictions[0])

                    img_name = f"{path.stem}.png"
                    # if pd != gt and not (scaled_folder / img_name).exists():
                    #     cv2.imwrite(str(scaled_folder / img_name), scaled_img)
                    #     cv2.imwrite(str(pred_folder / img_name), model.drawshow(scaled_img, predictions))
                    #     shutil.copy(path, origin_folder / path.name)
                    encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
                    if pd != gt and not (scaled_folder / img_name).exists():

                        cv2.imencode('.png', scaled_img,encode_params)[1].tofile(str(scaled_folder / img_name))
                        pred_img = model.drawshow(scaled_img, predictions)
                        cv2.imencode('.png', pred_img,encode_params)[1].tofile(str(pred_folder / img_name))
                        shutil.copy(path, origin_folder / path.name)

                arr.append(pd)
                folder2error[path.parents[1].name] += pd != gt

            if all(map(is_empty_dir, local_dst_folder.iterdir())):
                shutil.rmtree(local_dst_folder)

            return np.array(arr, dtype=np.int32), needs_write, folder2error

        results = {}
        needs_write = False

        for subtype in self.type_info[tpe].type:
            pd, flag, f2e = analyse_single_type(subtype)
            needs_write |= flag
            results[subtype] = pd, f2e

        if needs_write:
            with open(cache_path, "wb") as f:
                pickle.dump(cache, f)

        return results  # dict[MaterialSize, [np.ndarray, dict[str, int]]]

    def run(self):
        for tpe, v in self.type_info.items():
            print(f"\n\n----- Analyzing {tpe} model -----")

            n = len(v.models)
            model_alias = tuple(model.parent.name for model in v.models)
            results = tuple(self.analyse_single_model(model, tpe) for model in v.models)

            entire_gt = np.array([])
            entire_pd = [np.array([]) for _ in range(n)]
            entire_errors = [Counter() for _ in range(n)]

            for subtype in v.type:
                gt = self.labels[subtype]
                entire_gt = np.r_[entire_gt, gt]

                stats  = [None] * n
                errors = [None] * n

                for i, v in enumerate(results):
                    pd, error = v[subtype]
                    stats[i] = self.get_statistic(pd, gt)
                    errors[i] = error

                    entire_pd[i] = np.r_[entire_pd[i], pd]
                    entire_errors[i] += error

                print(f"\n-- Analyzing material size {subtype} --")
                self.show_statistic(stats, errors, model_alias, gt.size)

            print(f"\n-- Overview of {tpe} model --")
            entire_stats = [self.get_statistic(pd, entire_gt) for pd in entire_pd]

            self.show_statistic(entire_stats, entire_errors, model_alias, entire_gt.size, True)