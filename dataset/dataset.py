from torch.utils.data import Dataset
import albumentations as A

class SeDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        if self.transform:
            A.Compose([])
