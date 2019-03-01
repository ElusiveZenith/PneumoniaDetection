'''
Creates train and validation files of images that have detection boxes.
'''

import os
from random import shuffle

val_size = 2000

def main():
    darknet_dir = input("Path to directory of darknet.exe: ")
    image_dir = input("Path to directory of images: ")
    if not os.path.isdir(darknet_dir):
        print(darknet_dir, "is not a directory.")
    if not os.path.isdir(image_dir):
        print(image_dir, "is not a directory.")
    if os.path.islink(image_dir):
        image_dir = os.readlink(image_dir)

    image_files = []
    rel_path = os.path.relpath(image_dir, darknet_dir)
    file_paths = os.listdir(image_dir)
    for file_path in file_paths:
        name, ext = os.path.splitext(file_path)
        if ext != ".jpg":
            continue
        with open(os.path.join(image_dir, name + ".txt")) as file:
            if len([x for x in file.read().rsplit("\n") if x != '']) > 0:
                image_files.append(os.path.join(rel_path, file_path))


    shuffle(image_files)

    train_file = os.path.join(darknet_dir, "data/train.txt")
    with open(train_file, 'w') as file:
        file.write('\n'.join(image_files[val_size:]))
    
    val_file = os.path.join(darknet_dir, "data/validate.txt")
    with open(val_file, 'w') as file:
        file.write('\n'.join(image_files[:val_size]))

    print("Train", len(image_files[val_size:]))
    print("Val", len(image_files[:val_size]))


if __name__ == "__main__":
    main()
    input("Press enter to close.")
