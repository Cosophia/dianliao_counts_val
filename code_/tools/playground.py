import shutil
from pathlib import Path

if __name__ == '__main__':
    p = Path(r"D:\tan\20251205-val\MaterialCount_Val_DataSets_251029_251205")
    sep = merge = 0

    for folder in p.iterdir():
        if folder.name.count('_') != 2:
            continue

        _, x = folder.name.split('_', 1)
        new_folder = p / x

        if new_folder.is_dir():
            for file in folder.iterdir():
                shutil.move(file, new_folder / file.name)
            folder.rmdir()
            merge += 1
        else:
            folder.rename(new_folder)
            sep += 1

    print(merge, sep, merge + sep)
