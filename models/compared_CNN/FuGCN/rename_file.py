import os
import shutil
import glob


if __name__ == '__main__':
    path = "results/Rain200H"
    file_lists = glob.glob(path + "/*")
    print(file_lists)

    for file in file_lists:
        shutil.move(file, file.split('.')[0][:-2] + ".png")