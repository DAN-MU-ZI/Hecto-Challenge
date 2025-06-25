import torch
import torch.nn as nn

def get_optimizer(model, cfg):
    optimizer_name = cfg['optimizer']  # 예: 'Adam'
    optimizer_cls = getattr(torch.optim, optimizer_name)
    
    return optimizer_cls(model.parameters(), **cfg['optimizer_params'])

def get_scheduler(optimizer, cfg):
    scheduler_name = cfg.get('scheduler', 'CosineAnnealingLR')  # 기본값 제공
    scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
    scheduler_params = cfg.get('scheduler_params', {'T_max': cfg['epochs']})
    return scheduler_cls(optimizer, **scheduler_params)

def get_criterion(cfg):
    criterion_name = cfg.get('criterion', 'CrossEntropyLoss')
    criterion_cls = getattr(nn, criterion_name)
    criterion_params = cfg.get('criterion_params', {})
    return criterion_cls(**criterion_params)
