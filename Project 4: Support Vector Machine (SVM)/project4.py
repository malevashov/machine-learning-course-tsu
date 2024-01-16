from PIL import Image
import os
import numpy as np

path_to_files = "/my/path/"
array_of_images = []

for _, file in enumerate(os.listdir(path_to_files)):
    if "direction.jpg" in file: # to check if file has a certain name
        single_im = Image.open(file)
        single_array = np.array(im)
        array_of_images.append(single_array)
np.savez("all_images.npz",array_of_images) # save all in one file