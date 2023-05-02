import os
import shutil


def write_file_list(file_list, output_file):
    with open(output_file, "w") as f:
        for file_name in file_list:
            f.write(file_name + "\n")


ms4dnet_path = '/home/julian/Documents/MS4D-Net-Building-Damage-Assessment/'
ext_folder = '/media/julian/Extreme Pro/out_images/'

datasets = [f for f in os.listdir(ms4dnet_path + 'txt_files/') if f != 'experiments']


for folder in datasets:
    txt_fs = [f for f in os.listdir(ms4dnet_path + 'txt_files/' + folder) if f.endswith('.txt')]
    for txt_f in txt_fs:
        x = ''
        if 'masks' in txt_f:
            fold2 = 'masks'
            subset = txt_f.replace(f'{fold2}_', '').replace('.txt', '') 
            if subset == 'train': x = '_unsup'
            ms4d_name = subset + x + '_label.txt'
        elif 'pre_images' in txt_f:
            fold2 = 'pre_images'
            subset = txt_f.replace(f'{fold2}_', '').replace('.txt', '')
            if subset == 'train': x = '_unsup'
            ms4d_name = subset + x + '_image_pre.txt'
        elif 'post_images' in txt_f:
            fold2 = 'post_images'
            subset = txt_f.replace(f'{fold2}_', '').replace('.txt', '')
            if subset == 'train': x = '_unsup'
            ms4d_name = subset + x + '_image_post.txt'
        if ms4d_name == 'train_unsup_label.txt': continue
        with open(ms4dnet_path + 'txt_files/' + folder + '/' + txt_f) as fp:
            lines = fp.read().splitlines()
        os.makedirs(ms4dnet_path + 'txt_files/' + folder + '/' + 'old_prep_files', exist_ok=True)
        shutil.move(ms4dnet_path + 'txt_files/' + folder + '/' + txt_f, ms4dnet_path + 'txt_files/' + folder + '/' + 'old_prep_files/' + txt_f)
        t = [ext_folder + folder + '/' + fold2 + '/' + l for l in lines]
        print(ms4dnet_path + 'txt_files/' + folder + '/' + ms4d_name)
        write_file_list(t, ms4dnet_path + 'txt_files/' + folder + '/' + ms4d_name)
    shutil.move(ms4dnet_path + 'txt_files/' + folder + '/' + 'masks_train.txt', ms4dnet_path + 'txt_files/' + folder + '/' + 'old_prep_files/' + 'masks_train.txt')

