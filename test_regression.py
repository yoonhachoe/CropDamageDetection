import os
import csv
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import cv2
import torch.nn as nn
import train_regression_model_working
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def test():
    device = torch.device('cuda')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    transform = A.Compose([A.Resize(224, 224), A.Normalize(mean=(0.2611, 0.4760, 0.3845), std=(0.2331, 0.2544, 0.2608)), ToTensorV2()])

    ### test for one image from mobile ###
    image_path = "/mnt/Plant_Images/Partner/9291000000087.jpg" # change the image path
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)["image"]
    image = image.unsqueeze(0)
    image = image.to(device)  # torch.Size([1, 3, 224, 224])

    model = train_regression_model_working.ResNet()
    model.load_state_dict(torch.load("./regression_Plant subset_Dataset 2_ResNet50_300epochs.pth")['model_state_dict']) # regression best performing model
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred = model.forward(image)  # torch.Size([1])

    print(pred.item())



    ### test for csv file ###
    # #f1 = open('Partner_TUM_Dataset 1_w_test.csv', 'r', encoding='utf-8') # Dataset 1
    # f1 = open('Partner_TUM_Drone_Dataset 2_w_test.csv', 'r', encoding='utf-8') # Dataset 2
    # #f2 = open('test_ImageNet_Dataset 1_ResNet34_total.csv', 'w', encoding='utf-8') # Dataset 1
    # f2 = open('test_Plant subset_Dataset 2_ResNet50_total.csv', 'w', encoding='utf-8') # Dataset 2
    # r = csv.reader(f1)
    # w = csv.writer(f2)
    # data_list = sorted(line for line in r if line[8] == "True")
    #
    # w.writerow(['gt', 'pred', 'quality'])
    # MSE_loss = nn.MSELoss(reduction="mean")
    # MAE_loss = nn.L1Loss()
    # sum_test_loss = 0.0
    # sum_test_mae = 0.0
    #
    # for i in range(1105): # Dataset1 - TUM:907, Partner:28, total: 935 # Dataset2 - TUM:1105, Partner:32, Drone:59, total:1196
    #     image = cv2.imread(data_list[i][0])
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image = transform(image=image)["image"]
    #     image = image.unsqueeze(0)
    #     image = image.to(device) # torch.Size([1, 3, 224, 224])
    #     leaf_damage = torch.tensor(float(data_list[i][4]), dtype=torch.float)
    #     leaf_damage = leaf_damage.to(device)
    #     leaf_damage = leaf_damage.unsqueeze(0) # torch.Size([1])
    #
    #     model = train_regression_model_working.ResNet() # change resnet model properly
    #     # Choose one of the below models
    #     #model.load_state_dict(torch.load("./regression_ImageNet_Dataset 1_ResNet34_200epochs.pth")) # ImageNet, Dataset 1, ResNet-34
    #     #model.load_state_dict(torch.load("./regression_Plant subset_Dataset 1_ResNet34_300epochs.pth")) # Plant subset, Dataset 1, ResNet-34
    #     #model.load_state_dict(torch.load("./regression_Plant subset_Dataset 2_ResNet34_200epochs.pth")['model_state_dict']) # Plant subset, Dataset 2, ResNet-34
    #     model.load_state_dict(torch.load("./regression_Plant subset_Dataset 2_ResNet50_300epochs.pth")['model_state_dict'])  # Plabt_subset, Dataset 2, ResNet-50
    #
    #     model.to(device)
    #     model.eval()
    #
    #     with torch.no_grad():
    #         pred = model.forward(image) # torch.Size([1])
    #
    #     print(data_list[i][4], pred.item())
    #     w.writerow([data_list[i][4], pred.item(), data_list[i][7]])
    #
    #     test_mse = MSE_loss(pred, leaf_damage)
    #     test_mae = MAE_loss(pred, leaf_damage)
    #     print(test_mae)
    #     sum_test_loss += test_mse.item()
    #     sum_test_mae += test_mae.item()
    #
    # f1.close()
    # f2.close()
    # print('test loss(RMSE)', np.sqrt(sum_test_loss / 1105)) # change the number of test images properly
    # print('test loss(MAE)', sum_test_mae / 1105) # change the number of test images properly

if __name__ == "__main__":
    test()
