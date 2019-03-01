'''
Creates image.txt file from images listed in validation file.
'''

import os

init_paths = []
init_file = input("Path to file with names of images to run: ")
with open(init_file, 'r') as file:
    init_paths = file.read().rsplit("\n")

dir_rel = "..\\PneimoniaData"
new_file = "images.txt"
with open(new_file, 'w') as file:
    for init_path in init_paths:
        base = os.path.basename(init_path)
        new_path = os.path.join(dir_rel, base)
        file.write(new_path + "\n")