import imageio
import os
import shutil
import glob


data_path = './resultsL_new'
img_f_list = glob.glob(data_path+'/*.png')
# img_f_list = os.listdir(data_path)
print(img_f_list)
for file in img_f_list:
    # file = file.split('.')[0][:-2]
    shutil.move(file, file.replace('x2.png', '.png'))