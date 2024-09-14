"""
Before running this code, you need to extract the OpenSRH dataset into the following structure,
and run it twice to generate both the 'train' and 'val' folders:

srh_split/
├── train
    ├── hgg
    │   ├── NIO_001-xxx.tif
    │   ├── NIO_001-xxx.tif
    │   └── ...
    └── lgg
        ├── NIO_053-2-xxx.tif
        ├── NIO_053-2-xxx.tif
        └── ...
    └── ...
├── val
    ├── hgg
    │   ├── NIO_004-xxx.tif
    │   ├── NIO_004-xxx.tif
    │   └── ...
    └── lgg
        ├── NIO_069-xxx.tif
        ├── NIO_069-xxx.tif
        └── ...
    └── ...
"""

import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Set the input and output folder paths
input_folder = "/srh_split/train"
output_folder = "/srh_rgb_train"  # Update to the new output folder path

def min_max_rescaling(array, percentile_clip=3):
    p_low, p_high = np.percentile(array, (percentile_clip, 100 - percentile_clip))
    array = array.clip(min=p_low, max=p_high)
    img = (array - p_low) / (p_high - p_low)
    return img

def process_folder(folder_path):
    num = 0
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".tif"):
                file_list.append((root, filename))
    
    for root, filename in tqdm(file_list, desc="Processing images", unit="image"):
        file_path = os.path.join(root, filename)
        img = Image.open(file_path)
        img.seek(0)
        img_arr1 = np.array(img)

        CH2 = min_max_rescaling(img_arr1)
        img.seek(1)
        img_arr2 = np.array(img)
        CH3 = min_max_rescaling(img_arr2)

        subtracted_array = np.subtract(CH3, CH2)
        subtracted_array[subtracted_array < 0] = 0.0
        stack = np.zeros((300, 300, 3), dtype=np.float)
        stack[:, :, 0] = subtracted_array
        stack[:, :, 1] = CH2
        stack[:, :, 2] = CH3
        stack = stack * 255

        result_image = Image.fromarray(stack.astype(np.uint8))
        output_subfolder = os.path.join(output_folder, os.path.relpath(root, input_folder))
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        output_path = os.path.join(output_subfolder, os.path.splitext(filename)[0] + ".png")
        result_image.save(output_path)
        num += 1
        
    print(f'Total images processed: {num}')

# Start processing the input folder and its subfolders
process_folder(input_folder)
print('Finished saving all images!')