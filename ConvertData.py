'''
Convers dicom images and execl labels to jpg and txt labels. Also converts the name to a numaric name.
'''

import os
import cv2
import pandas as pd
import pydicom as dicom


def yieldFilename(directory_path):
    '''Yeilds the next unused filename in the directory.
    Args:
        directory_path: Path to the directory where the file will be saved.
    Yeilds:
        filename: Next unused filename in the directory.
    '''
    try:
        os.makedirs(directory_path)
    except OSError:
        pass

    i = 0
    while True:
        filename = os.path.join(directory_path, "%06.d.jpg"%i)
        if not os.path.exists(filename):
            yield filename
        i += 1


def writeLableFile(entries, save_path, img_shape=(1024,1024)):
    '''Writes entry from excel to a txt.
    Args:
        entries: list of boxes to save to a txt file. [[class, x, y, width, height]] x of the left, and y of the top.
        save_path: The path to save the txt file to.
        image_shape: shape of the image.
    Returns:
        None
    '''
    with open(save_path, 'w') as file:
        for entry in entries:
            if entry[-1] != 0:
                width = entry[3]
                height = entry[4]
                x_center = entry[1] + width / 2
                y_center = entry[2] + height / 2

                x_center = x_center / img_shape[0]
                y_center = y_center / img_shape[1]
                width = width / img_shape[0]
                height = height / img_shape[1]
                file.write(' '.join(str(x) for x in [0, x_center, y_center, width, height]) + "\n")


def saveToJPG(dcm_path, jpg_path):
    '''Reads a dcm file and writes it to a jpg file.
    Args:
        dcm_path: Path to the dcm file.
        jpg_path: Path to where the jpg will be saved.
    Returns:
        None
    '''
    dcm = dicom.dcmread(dcm_path)
    img = dcm.pixel_array
    cv2.imwrite(jpg_path, img)


def main():
    #Gets paths to directories and files.
    images_path = input("Path to directory of images: ")
    labels_path = input("Path to Excel file of labels: ")
    new_path = input("Path of directory to save images and label files too: ")

    #Varifys that the paths provided are valid.
    if not os.path.exists(images_path):
        print("Invalid image folder.")
        return
    if not os.path.isdir(images_path):
        print(images_path, "is not a directory.")
        return
    if not os.path.exists(labels_path):
        print("Invalid label file or path.")
        return
    if not os.path.isfile(labels_path):
        print(labels_path, "is not a file.")
        return
    if os.path.splitext(os.path.basename(labels_path))[1] != ".csv":
        print(type(os.path.splitext(os.path.basename(labels_path))[1]))
        print(os.path.splitext(os.path.basename(labels_path))[1])
        print(os.path.basename(labels_path), "is not a csv file.")
        return

    running_list = []
    last = None
    fileGen = yieldFilename(new_path)
    entry_list = pd.read_csv(labels_path).values.tolist()
    
    #Adding none list so that it runs one more time to save last item.
    #Loops through the entries in the excel file and combines the ones that go to the same image.
    for entry in entry_list + [[None]]:
        #If the entry is not to the same image as the last, save the image and label files for the last image.       
        if entry[0] is not last:
            if len(running_list) > 0:
                dicom_path = os.path.join(images_path, (running_list[0][0] + ".dcm"))
                if os.path.exists(dicom_path):
                    jpg_path = fileGen.__next__()
                    saveToJPG(dicom_path, jpg_path)
                    txt_path = jpg_path.replace(".jpg", ".txt")
                    writeLableFile(running_list, txt_path)

            running_list = [entry]
            last = entry[0]
        #If the entry is to the same image as the last, combines with the last image.
        else:
            running_list.append(entry)


if __name__ == "__main__":
    main()
    input("Press enter to close.")