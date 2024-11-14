import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

def stitch_images(folder_path, save_dir, canvas_size=5400, crop_size=300, step=100):
    x = 0
    y = 0

    counter = 0

    # Initialize a blank canvas to stitch images onto
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    files = sorted(os.listdir(folder_path))

    # Process each image in the folder
    for filename in tqdm(files, desc="Processing images", ncols=100):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            # Extract coordinates from the filename and calculate position on canvas
            file_name = os.path.splitext(filename)[0]
            coords = file_name.split("-")[2].split("_")
            y1, x1, y2, x2 = map(int, coords)

            y_start = y1 + y2
            x_start = x1 + x2

            # Place the image onto the canvas at the calculated position
            canvas[y_start + y:y_start + y + crop_size, x_start + x:x_start + x + crop_size] = img

            counter += 1

            if counter % 9 == 0:
                x -= 100

            if counter % 54 == 0:
                y -= 100
                x = 0

    cv2.imwrite(f"{save_dir}/SRS.png", canvas)

    return canvas

def split_and_save_image(image, save_dir, crop_size=(300, 300), step=100):
    image = Image.open(f'{save_dir}/SRS.png')

    image_width, image_height = image.size
    crop_width, crop_height = crop_size
    crops = []

    # Split the image into smaller crops
    for y in tqdm(range(0, image_height - crop_height + 1, step), desc="Vertical splits", ncols=100):
        for x in range(0, image_width - crop_width + 1, step):
            box = (x, y, x + crop_width, y + crop_height)
            crop = image.crop(box)
            crops.append((x, y, crop))

    for i, (x, y, crop) in enumerate(tqdm(crops, desc="Saving crops", ncols=100)):
        crop.save(f"{save_dir}/crop_{x}_{y}.png")
    

folder_path = "data/002-3-rgb" # All rgb images of a slide
save_dir = './t' # Temporarily stored folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

canvas = stitch_images(folder_path, save_dir)
crops = split_and_save_image(canvas,save_dir)
