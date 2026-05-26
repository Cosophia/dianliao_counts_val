import json
from pathlib import Path
from itertools import chain

def remove_chinese_char(folder: Path):

    for img in chain(folder.rglob(f"*png"), folder.rglob(f"*jpg")):
        name = img.name[5:]
        parent = img.parent

        img.rename(parent / name)

        origin_lbl = img.with_suffix(".json")
        lbl = (parent / name).with_suffix('.json')
        origin_lbl.rename(lbl)

        with open(lbl, "r", encoding='utf-8') as f:
            data = json.load(f)
        data["imagePath"] = name
        with open(lbl, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == '__main__':
    remove_chinese_char(Path(r"C:\Users\Administrator\Desktop\sanxing"))
