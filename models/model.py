import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from torch.nn.parameter import Parameter

# GeM Pooling 정의
class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(1./self.p).squeeze(-1).squeeze(-1)

class AdaFaceDecoder(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.4, h=0.333, eps=1e-6, t_alpha=0.01):
        super(AdaFaceDecoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m  # base margin
        self.h = h  # adaptive margin scaling
        self.eps = eps
        self.t_alpha = t_alpha

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Batch-wise moving average for feature norm statistics
        self.register_buffer('batch_mean', torch.tensor(20.0))
        self.register_buffer('batch_std', torch.tensor(100.0))

    def forward(self, input, label=None):
        norm = input.norm(dim=1, keepdim=True).clamp(min=self.eps)  # (B, 1)
        input_normalized = input / norm
        weight_normalized = F.normalize(self.weight)

        cosine = F.linear(input_normalized, weight_normalized).clamp(-1 + self.eps, 1 - self.eps)

        # adaptive margin (논문: (norm - batch_mean) / (batch_std + eps))
        with torch.no_grad():
            batch_mean = norm.mean()
            batch_std = norm.std()
            self.batch_mean = (1 - self.t_alpha) * self.batch_mean + self.t_alpha * batch_mean
            self.batch_std = (1 - self.t_alpha) * self.batch_std + self.t_alpha * batch_std

        margin_scaler = (norm - self.batch_mean) / (self.batch_std + self.eps)
        margin = self.m + self.h * margin_scaler

        if label is None:
            return cosine * self.s

        # ArcFace 방식: cos(θ + m) = cosθ * cosm - sinθ * sinm
        theta = torch.acos(cosine)
        # 각 샘플별 마진 적용 (정답 레이블 위치만 margin, 나머진 0)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        margin_per_sample = (margin * one_hot).sum(dim=1, keepdim=True)
        theta_m = theta + margin_per_sample
        target_logits = torch.cos(theta_m)

        output = cosine * (1 - one_hot) + target_logits * one_hot
        output *= self.s

        return output

class Encoder(nn.Module):
    def __init__(self, model_name, embedding, pretrained=True, freeze_until=None):
        super().__init__()
        self.backbone = create_model(model_name, pretrained=pretrained, features_only=True)
        feat_dim = self.backbone.feature_info.channels()[-1]

        self.pool = GeMPooling()
        self.neck = nn.Sequential(
            nn.Linear(feat_dim, embedding),
            nn.BatchNorm1d(embedding),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
      x = self.backbone(x)[-1]
      x = self.pool(x)
      x = self.neck(x)
      return x

class BaseModel(nn.Module):
    def __init__(self, model_name, num_classes, embedding, freeze_until=None):
        super().__init__()
        self.encoder = Encoder(model_name, embedding, freeze_until=freeze_until)
        self.decoder = AdaFaceDecoder(embedding, num_classes, s=30.0, m=0.7, h=0.5)

    def forward(self, x, labels=None, **kwargs):
        x = self.encoder(x)
        return self.decoder(x, labels)