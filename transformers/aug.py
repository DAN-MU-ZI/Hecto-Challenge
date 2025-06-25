import torch
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(img_size, train=True):
    if train: return get_train_transform(img_size)
    else: return get_val_transform(img_size)

def get_train_transform(img_size):
    return A.Compose([
        # 기본 증강
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
        # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    
        # 정규화
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
def get_val_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_tta_transforms():
    def make_tta(extra=None):
        augmentations = []
        if extra is not None:
            if isinstance(extra, (list, tuple)):
                augmentations.extend(extra)
            else:
                augmentations.append(extra)
        augmentations += [
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        return A.Compose(augmentations)
    return [
        make_tta(),                                       # 원본
        make_tta(A.HorizontalFlip(p=1)),                  # 좌우반전
        make_tta(A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=1.0))
    ]


class TTAWrapper:
    def __init__(self, model, tta_transforms, device):
        self.model = model
        self.tta_transforms = tta_transforms
        self.device = device
        
    def predict(self, x_np):  # x_np: numpy array (batch of images)
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for transform in self.tta_transforms:
                batch = []
                for img in x_np:
                    augmented = transform(image=img)['image']
                    batch.append(augmented)
                batch_tensor = torch.stack(batch).to(self.device)
                logits = self.model(batch_tensor)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        # 평균 (TTA개수, batch, class) -> (batch, class)
        return torch.stack(predictions).mean(dim=0)
