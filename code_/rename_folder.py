import os
import shutil
from os.path import isdir

root = "D:\stablity_counts_analysis\data"

for root, dirs, files in os.walk(root):
    if files:
        for i in os.listdir(root):
            file = os.path.join(os.path.join(root, i))
            if (isdir(file)):
                for pic in os.listdir(file):
                    src = os.path.join(root, i, pic)
                    dst = os.path.join(root, pic)
                    #print(src, dst)
                    shutil.copy(src, dst)

                shutil.rmtree(file)



