from torch.utils.data import Dataset
import torch
import cv2

class DatasetReg(Dataset):
    def __init__(self, data_path_class_list, transform):
        self.list = data_path_class_list
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = cv2.imread(self.list[idx][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        leaf_damage = torch.tensor(float(self.list[idx][4]), dtype=torch.float)

        return image, leaf_damage

