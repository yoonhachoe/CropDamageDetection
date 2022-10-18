import click
import os
import datetime
import csv
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.nn as nn
from dataset_regression import DatasetReg
import train_regression_model_working

@click.command()
@click.option('--epochs', help='Number of epochs to train', type=int, required=True)
@click.option('--lr', help='Learning rate', type=float, default=1e-4)
@click.option('--betas', help='Momentum parameters for Adam', type=tuple, default=(0.9, 0.999))
@click.option('--batch_size', help='Batch size for train- and valloader', type=int, default=64)
@click.option('--reg', help='L2 Regularization strength as Adam weight decay', type=float, default=1e-5)

def train(
        epochs: int,
        lr: float,
        betas: tuple,
        batch_size: int,
        reg: float
):
    device = torch.device('cuda')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.mkdir('./model/' + time)
    writer = SummaryWriter(log_dir="tensorboard_logs/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    transform = A.Compose([A.Resize(224, 224), A.Normalize(mean=(0.2611, 0.4760, 0.3845), std=(0.2331, 0.2544, 0.2608)), A.Rotate(limit=90, p=0.5), A.Flip(p=0.5), ToTensorV2()])
    f = open('Partner_TUM_Drone_Dataset 2_w_test.csv', 'r', encoding='utf-8')
    r = csv.reader(f)
    data_path_class_list = sorted(line for line in r if line[8] == "False")
    f.close()
    dataset = DatasetReg(data_path_class_list, transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # define model
    model = train_regression_model_working.ResNet()
    model.to(device)
    summary(model, (3, 224, 224))

    # define loss
    MSE_loss = nn.MSELoss(reduction="mean")
    MAE_loss = nn.L1Loss()

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr, betas, weight_decay=reg)

    for e in range(0, epochs):
        sum_train_loss = 0.0
        sum_val_loss = 0.0
        sum_train_mae = 0.0
        sum_val_mae = 0.0
        print("Epoch:", e)
        for i, item in enumerate(trainloader):
            model.train()
            image, leaf_damage = item
            image = image.to(device) # torch.Size([B, 3, 224, 224])
            leaf_damage = leaf_damage.to(device) # torch.Size([B])
            optimizer.zero_grad()
            pred = model.forward(image) # torch.Size([B])
            train_criterion = MSE_loss(pred, leaf_damage)
            train_loss = torch.sqrt(train_criterion)
            train_mae = MAE_loss(pred, leaf_damage)
            train_loss.backward()
            optimizer.step()
            sum_train_loss += train_loss.item()
            sum_train_mae += train_mae.item()

        writer.add_scalar('train loss(RMSE)', sum_train_loss / len(trainloader), e)
        print('train loss(RMSE)', sum_train_loss / len(trainloader))
        writer.add_scalar('train loss(MAE)', sum_train_mae / len(trainloader), e)
        print('train loss(MAE)', sum_train_mae / len(trainloader))

        for i, item in enumerate(testloader):
            model.eval()
            image, leaf_damage = item
            image = image.to(device)
            leaf_damage = leaf_damage.to(device)
            with torch.no_grad():
                pred = model.forward(image)
            val_criterion = MSE_loss(pred, leaf_damage)
            val_loss = torch.sqrt(val_criterion)
            val_mae = MAE_loss(pred, leaf_damage)
            sum_val_loss += val_loss.item()
            sum_val_mae += val_mae.item()

        writer.add_scalar('val loss(RMSE)', sum_val_loss / len(testloader), e)
        print('val loss(RMSE)', sum_val_loss / len(testloader))
        writer.add_scalar('val loss(MAE)', sum_val_mae / len(testloader), e)
        print('val loss(MAE)', sum_val_mae / len(testloader))

        torch.save({'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, './model/' + time + '/model_{epoch:03d}epochs.pth'.format(epoch=e+1))

if __name__ == "__main__":
    train()
