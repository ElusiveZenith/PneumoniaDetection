'''
Creates image.txt file containing all images.
'''

import os

cwd = os.getcwd()
image_dir = input("Image directory: ")
rel_path = os.path.relpath(image_dir, cwd)

filenames = os.listdir(image_dir)
new_file = "images.txt"
with open(new_file, 'w') as file:
    for filename in filenames:
        _, ext = os.path.splitext(filename)
        if ext != ".jpg":
            continue
        file.write(os.path.join(rel_path, filename) + "\n")