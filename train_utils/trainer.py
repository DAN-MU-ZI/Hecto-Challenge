import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import log_loss
import random


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_mixup=True, use_cutmix=True):
    model.train()
    
    probs_list, labels_list = [], []
    for i, (images, labels) in enumerate(tqdm(loader, desc="Train")):
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        
        apply_mixup = use_mixup and random.random() < 0.5
        apply_cutmix = use_cutmix and not apply_mixup and random.random() < 0.5
        
        # if apply_mixup:
        #     images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=1.0)
        # elif apply_cutmix:
        #     images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)

        apply_cutmix = True
        images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
        with autocast(device_type="cuda"):
            if apply_mixup or apply_cutmix:
                logits = model(images)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                logits = model(images, labels)
                loss = criterion(logits, labels)
                
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs_list.append(probs)
        labels_list.append(labels.detach().cpu().numpy())
        
    all_probs = np.concatenate(probs_list)
    all_labels = np.concatenate(labels_list)
    logloss = log_loss(all_labels, all_probs, labels=list(range(logits.shape[1])))
    preds = np.argmax(all_probs, axis=1)
    acc = (preds == all_labels).mean()
    return logloss, acc, all_probs, all_labels

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Valid"):
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            probs_list.append(probs)
            labels_list.append(labels.detach().cpu().numpy())

    all_probs = np.concatenate(probs_list)
    all_labels = np.concatenate(labels_list)

    logloss = log_loss(all_labels, all_probs, labels=list(range(logits.shape[1])))
    preds = np.argmax(all_probs, axis=1)
    acc = (preds == all_labels).mean()
    return logloss, acc, all_probs, all_labels


import torch
import numpy as np

# MixUp 구현
def mixup_data(x, y, alpha=1.0):
    """MixUp 데이터 증강 적용"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp 손실 계산"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# CutMix 구현
def rand_bbox(size, lam):
    """CutMix를 위한 경계 상자 생성"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 중앙 좌표 무작위 선택
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 경계 상자 좌표 계산
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    """CutMix 데이터 증강 적용"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # 혼합 비율 조정
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[2] * x.size()[3]))

    return x, y_a, y_b, lam