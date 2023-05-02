import random
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
import os
import cv2 

def display_segmentation_mask(mask):
    # Define the color codes for each class
    color_codes = {
        0: [0, 0, 0],     # background (black)
        1: [0, 255, 0],   # no visible damage (green)
        2: [255, 255, 0], # possibly damaged (yellow)
        3: [255, 165, 0], # damaged (orange)
        4: [255, 0, 0]    # destroyed (red)
    }
    
    # Create an empty array for the colored mask
    height, width = mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Loop over each pixel in the mask and set the corresponding color
    for i in range(height):
        for j in range(width):
            pixel_class = mask[i, j]
            colored_mask[i, j] = color_codes[pixel_class]
    
    # Display the colored mask using OpenCV
    return colored_mask


direc = '/home/julian/Documents/MS4D-Net-Building-Damage-Assessment/'
diir = 'save_d100/'
os.chdir(direc + diir)
os.makedirs('prediction_examples_vgg19/', exist_ok=True)

diir_ext = '/media/julian/Extreme Pro/out_images/'
data_set = '10300100E291D100'

dmg_msks = os.listdir('predictions_vgg_19_stack/')

gt_dir = diir_ext + data_set + f'/masks/'

dmg_msk = random.choice(dmg_msks)


# dmg_msk = f'post_{random_set}_2_14.png'
gt_msk = dmg_msk.replace('post', 'mask')

pred = cv2.imread('predictions_vgg_19_stack/' + dmg_msk, 0)
gt = cv2.imread(gt_dir + f'{gt_msk}', 0)

pre_im_dir = diir_ext + data_set + '/pre_images/'
post_im_dir = diir_ext + data_set + '/post_images/'


pre_i = dmg_msk.replace('post', 'pre')
pre_im = cv2.imread(pre_im_dir + f'{pre_i}')[:,:,::-1]
post_im = cv2.imread(post_im_dir + f'{dmg_msk}')[:,:,::-1]

gt_im = display_segmentation_mask(pred)
pred_im = display_segmentation_mask(gt)


mpl.rcParams.update(mpl.rcParamsDefault)
f, axarr = plt.subplots(2, 2)

axarr[0,0].imshow(pred_im)
axarr[0,1].imshow(gt_im)
axarr[1,0].imshow(pre_im)
axarr[1,1].imshow(post_im)

axarr[0,0].title.set_text('Ground Truth')
axarr[0,1].title.set_text('MS4D-Net Prediction')
axarr[1,0].title.set_text('Pre-event Image')
axarr[1,1].title.set_text('Post-event Image')


plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
print(f'Prediction for dataset: {data_set} Split: test, tile: {dmg_msk.replace(f"post_{data_set}_", "").replace(".png", "")}')
# plt.savefig('prediction_examples/' + f'{split}_{dmg_msk.replace(f"post_{random_set}_", "changeos_prediction_")}')
plt.show()


cv2.imread('/media/julian/Extreme Pro/out_images/10300100E291D100/pre_images/pre_10300100E291D100_7_8.png')