import os
import tifffile as tiff
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

def split_image(image_path, output_directory):
    # Open the image file
    tiff_image = tiff.imread(image_path).astype('uint8')    

    # Split the image into four quadrants
    width, height = tiff_image.shape[1], tiff_image.shape[0]
    half_width = width // 2
    half_height = height // 2
    
    filename = Path(image_path).stem
    
    top_left = tiff_image[0:half_height, 0:half_width]
    top_right = tiff_image[0:half_height, half_width:width]
    bottom_left = tiff_image[half_height:height, 0:half_width]
    bottom_right = tiff_image[half_height:height, half_width:width]

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Save the quadrants as PNG files with appended tile numbers
    plt.imsave(os.path.join(output_directory, filename + "_patch_00.png"), top_left)
    plt.imsave(os.path.join(output_directory, filename + "_patch_01.png"), top_right)
    plt.imsave(os.path.join(output_directory, filename + "_patch_10.png"), bottom_left)
    plt.imsave(os.path.join(output_directory, filename + "_patch_11.png"), bottom_right)



os.makedirs('xBD_512', exist_ok=True)
os.makedirs('xBD_512/post_images', exist_ok=True)
os.makedirs('xBD_512/pre_images', exist_ok=True)
os.makedirs('xBD_512/masks', exist_ok=True)


for split in tqdm(['tier1', 'hold', 'test']):
    files = [f for f in os.listdir(f'geotiffs/{split}/images') if 'post' in f]
    for f in files:
        split_image(f'geotiffs/{split}/images/{f}', f'xBD_512/post_images/')


for split in tqdm(['tier1', 'hold', 'test']):
    files = [f for f in os.listdir(f'geotiffs/{split}/images') if 'pre' in f]
    for f in files:
        split_image(f'geotiffs/{split}/images/{f}', f'xBD_512/pre_images/')

for split in tqdm(['tier1', 'hold', 'test']):
    files = [f for f in os.listdir(f'{split}/masks/tifs/')]
    for f in files:
        split_image(f'{split}/masks/tifs/{f}', f'xFBD_512/masks/')


# xFBD

# pre_masks = [f for f in os.listdir('tier1/masks') if 'pre'in f]
# pre_images = [f for f in os.listdir('tier1/images/') if 'pre' in f]
# post_images = [f for f in os.listdir('tier1/images/') if 'post' in f]

# os.makedirs('xFBD_512/masks', exist_ok=True)
# os.makedirs('xFBD_512/post_images', exist_ok=True)
# os.makedirs('xFBD_512/pre_images', exist_ok=True)

# for split in tqdm(['tier1', 'hold', 'test']):
#     files = [f for f in os.listdir(f'{split}/masks/tifs/')]
#     for f in files:
#         split_image(f'{split}/masks/tifs/{f}', f'xFBD_512/masks/')
        
# for split in tqdm(['tier1', 'hold', 'test']):
#     files = [f for f in os.listdir(f'{split}/images/') if 'post' in f]
#     for f in files:
#         split_image(f'{split}/images/{f}', f'xFBD_512/post_images/')

# for split in tqdm(['tier1', 'hold', 'test']):
#     files = [f for f in os.listdir(f'{split}/images/') if 'pre' in f]
#     for f in files:
#         split_image(f'{split}/images/{f}', f'xFBD_512/pre_images/')