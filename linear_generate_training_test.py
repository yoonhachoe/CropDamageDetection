import csv
import numpy as np
from PIL import Image

def train():
    # generate training csv
    f1 = open('Partner_TUM_Dataset 1_w_test.csv', 'r', encoding='utf-8')
    f2 = open('Green_training.csv', 'w', encoding='utf-8', newline='')
    r = csv.reader(f1)
    w = csv.writer(f2)
    data_path_class_list = sorted(line for line in r if line[8] == "False")
    for i in range(8850):
        img = Image.open(data_path_class_list[i][0]).convert('HSV')

        # Extract Hue channel and make Numpy array for fast processing
        Hue = np.array(img.getchannel('H'))

        # Make mask of zeroes in which we will set greens to 1
        mask = np.zeros_like(Hue, dtype=np.uint8)

        # Set all green pixels to 1
        mask[(Hue > int(60 * 0.708333333)) & (Hue < int(180 * 0.708333333))] = 1

        # Now print percentage of green pixels
        green = (mask.mean() * 100)
        w.writerow([data_path_class_list[i][0], data_path_class_list[i][4], data_path_class_list[i][7], green])

    f1.close()
    f2.close()

    # generate test csv
    f1 = open('Partner_TUM_Dataset 1_w_test.csv', 'r', encoding='utf-8')
    f2 = open('Green_test.csv', 'w', encoding='utf-8', newline='')
    r = csv.reader(f1)
    w = csv.writer(f2)
    data_path_class_list = sorted(line for line in r if line[8] == "True")
    for i in range(935):
        img = Image.open(data_path_class_list[i][0]).convert('HSV')

        # Extract Hue channel and make Numpy array for fast processing
        Hue = np.array(img.getchannel('H'))

        # Make mask of zeroes in which we will set greens to 1
        mask = np.zeros_like(Hue, dtype=np.uint8)

        # Set all green pixels to 1
        mask[(Hue > int(60 * 0.708333333)) & (Hue < int(180 * 0.708333333))] = 1

        # Now print percentage of green pixels
        green = (mask.mean() * 100)
        w.writerow([data_path_class_list[i][0], data_path_class_list[i][4], data_path_class_list[i][7], green])

    f1.close()
    f2.close()

if __name__ == "__main__":
    train()