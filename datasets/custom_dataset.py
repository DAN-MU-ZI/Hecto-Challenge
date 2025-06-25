import cv2
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False, classes=None, class_to_idx=None):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.samples = []
        self.classes = classes
        self.class_to_idx = class_to_idx
        if self.is_test:
            self.samples = [(img_path,) for img_path in self.df['img_path'].tolist()]
        else:
            self.samples = [(row['img_path'], row['label']) for _, row in self.df.iterrows()]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        if self.is_test:
            img_path = self.samples[idx][0]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image
        else:
            img_path, label = self.samples[idx]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, label
