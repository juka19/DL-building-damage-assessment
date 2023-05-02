def split_file(input):
    """
    Split the lines of a txt file into 4 parts and write them to new files.
    The split percentages should be given by the user.
    
    :param input_file: the path to the input txt file
    :param output_files: a list of 4 output file paths
    :param split_percentages: a list of 3 percentages that add up to 100
    """
    input_file, output_files, split_percentages = input
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # calculate the number of lines for each output file
    total_lines = len(lines)
    num_lines = []
    for percentage in split_percentages:
        num_lines.append(int(total_lines * percentage // 100))
    num_lines.append(total_lines - sum(num_lines))
    
    # write the lines to the output files
    start_idx = 0
    for i, num in enumerate(num_lines):
        end_idx = start_idx + num
        with open(output_files[i], 'w') as f:
            f.writelines(lines[start_idx:end_idx])
        start_idx = end_idx


output_files_post = ['MS4DNet_files/train_image_post.txt', 'MS4DNet_files/train_unsup_image_post.txt', 'MS4DNet_files/val_image_post.txt', 'MS4DNet_files/test_image_post.txt']
output_files_pre = ['MS4DNet_files/train_image_pre.txt', 'MS4DNet_files/train_unsup_image_pre.txt', 'MS4DNet_files/val_image_pre.txt', 'MS4DNet_files/test_image_pre.txt']
output_files_labels = ['MS4DNet_files/train_label.txt', 'MS4DNet_files/train_unsup_label.txt', 'MS4DNet_files/val_label.txt', 'MS4DNet_files/test_label.txt']

input_files = ['post_images.txt', 'pre_images.txt', 'labels.txt']
output_files = [output_files_post, output_files_pre, output_files_labels]
split_percentages = [[60, 20, 20], [60, 20, 20], [60, 20, 20]]

inputs = [(i, o, s) for i, o, s in zip(input_files, output_files, split_percentages)]

list(map(split_file, inputs))