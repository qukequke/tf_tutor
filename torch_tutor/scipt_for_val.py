import os
import shutil
import numpy as np


os.chdir('../cat_vs_dog_datasets')
# print(os.listdir('./train'))
file_names = os.listdir('./train')
if not os.path.exists('val'):
    os.mkdir('./val')

while len(os.listdir('./val')) < 5000:
    file_name = np.random.choice(os.listdir('./train'))
    shutil.move('./train/' + file_name, './val/' + file_name)



# shutil.move()
