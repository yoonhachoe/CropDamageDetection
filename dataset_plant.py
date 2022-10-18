from torch.utils.data import Dataset
from PIL import Image
import torch
import warnings

class DatasetPlant(Dataset):
    def __init__(self, data_path_class_list, transform):
        self.list = data_path_class_list
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
        image = Image.open(self.list[idx][0])
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        plant_class = torch.tensor(float(self.list[idx][1]), dtype=torch.long)

        return image, plant_class

