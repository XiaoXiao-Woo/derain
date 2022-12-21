import os
import cv2
from multiprocessing import Pool


file_list = os.listdir('./data/RainTrainL/train/rain')
print(file_list[0])
def replace_name(file):
    img = cv2.imread(os.path.join('./data/RainTrainL/train/rain', file))
    new_file = ''.join([file.split('.')[0][2:], '.png'])
    print(file, new_file)
    cv2.imwrite(os.path.join('./data/RainTrainL/train/tmp', new_file), img)
if __name__ == "__main__":
    pool = Pool(4)
    pool.map(replace_name, file_list)
    pool.close()
    pool.join()

