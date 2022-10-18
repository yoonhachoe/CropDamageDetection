import click
import os
import datetime
import csv
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.nn as nn
from dataset_plant import DatasetPlant
import pretrain_plant_model

@click.command()
@click.option('--epochs', help='Number of epochs to train', type=int, required=True)
@click.option('--lr', help='Learning rate', type=float, default=3e-4)
@click.option('--betas', help='Momentum parameters for Adam', type=tuple, default=(0.9, 0.999))
@click.option('--batch_size', help='Batch size for train- and valloader', type=int, default=256)
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
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    f = open('plant_dataset_500classes.csv', 'r', encoding='utf-8')
    r = csv.reader(f)
    data_path_class_list = sorted(line for line in r)
    f.close()
    dataset = DatasetPlant(data_path_class_list, transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # define model
    model = pretrain_plant_model.ResNet()
    model.to(device)
    summary(model, (3, 224, 224))

    # define loss
    loss = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr, betas, weight_decay=reg)

    for e in range(epochs):
        sum_train_loss = 0.0
        sum_train_err = 0.0
        train_top1 = 0.0
        train_top5 = 0.0
        sum_val_loss = 0.0
        sum_val_err = 0.0
        val_top1 = 0.0
        val_top5 = 0.0
        print("Epoch:", e)
        for i, item in enumerate(trainloader):
            model.train()
            image, plant_class = item
            image = image.to(device) # torch.Size([B, 3, 224, 224])
            plant_class = plant_class.to(device) # torch.Size([B])
            optimizer.zero_grad()
            pred = model.forward(image) # torch.Size([B, C])
            _, predicted_top1 = torch.max(pred.data, 1) # predicted class, torch.Size([B])
            _, predicted_top5 = torch.topk(pred.data, 5) # predicted class, torch.Size([B, 5])
            sum_train_err += plant_class.size(0) # B
            train_top1 += (predicted_top1 == plant_class).sum().item()
            for i in range(plant_class.size(dim=0)):
                train_top5 += torch.isin(plant_class[i], predicted_top5[i]).sum().item()
            train_loss = loss(pred, plant_class)
            train_loss.backward()
            optimizer.step()
            sum_train_loss += train_loss.item()

        writer.add_scalar('train loss', sum_train_loss / len(trainloader), e)
        print('train loss', sum_train_loss / len(trainloader))
        train_top1 = 100 * train_top1 / sum_train_err
        writer.add_scalar('train top1 accuracy', train_top1, e)
        print('train top1 accuracy', train_top1)
        train_top5 = 100 * train_top5 / sum_train_err
        writer.add_scalar('train top5 accuracy', train_top5, e)
        print('train top5 accuracy', train_top5)

        for i, item in enumerate(testloader):
            model.eval()
            image, plant_class = item
            image = image.to(device)
            plant_class = plant_class.to(device)
            with torch.no_grad():
                pred = model.forward(image)
                _, predicted_top1 = torch.max(pred.data, 1)
                _, predicted_top5 = torch.topk(pred.data, 5)
                sum_val_err += plant_class.size(0)
                val_top1 += (predicted_top1 == plant_class).sum().item()
                for i in range(plant_class.size(dim=0)):
                    val_top5 += torch.isin(plant_class[i], predicted_top5[i]).sum().item()
            val_loss = loss(pred, plant_class)
            sum_val_loss += val_loss.item()

        writer.add_scalar('val loss', sum_val_loss / len(testloader), e)
        print('val loss', sum_val_loss / len(testloader))
        val_top1 = 100 * val_top1 / sum_val_err
        writer.add_scalar('val top1 accuracy', val_top1, e)
        print('val top1 accuracy', val_top1)
        val_top5 = 100 * val_top5 / sum_val_err
        writer.add_scalar('val top5 accuracy', val_top5, e)
        print('val top5 accuracy', val_top5)

        torch.save({'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, './model/' + time + '/model_{epoch:03d}epochs.pth'.format(epoch=e + 1))

if __name__ == "__main__":
    train()
