from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np

class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(input_size=2048, hidden_size=128, num_layers=2, 
                             batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(128 * 2, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        batch_size, channels, height, width = x.shape
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch_size, width, 2048)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


def load_dialect_classifiers(model_paths: Dict[int, str], device: torch.device):
    classifiers = {}
    for dialect, path in model_paths.items():
        model = CNN_BiLSTM().to(device)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        classifiers[dialect] = model
    return classifiers

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np

class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(input_size=2048, hidden_size=128, num_layers=2, 
                             batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(128 * 2, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        batch_size, channels, height, width = x.shape
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch_size, width, 2048)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


def load_dialect_classifiers(model_paths: Dict[int, str], device: torch.device):
    classifiers = {}
    for dialect, path in model_paths.items():
        model = CNN_BiLSTM().to(device)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        classifiers[dialect] = model
    return classifiers

def compute_dialect_adv_loss(
    mfcc_batch: torch.Tensor,
    dialect_classifiers: Dict[int, nn.Module],
    dialect_ids: torch.Tensor
):
    """
    mfcc_batch: (B, 1, 128, 128)
    dialect_classifiers: {dialect_id: model (with sigmoid)}
    dialect_ids: (B,) LongTensor
    """
    device = mfcc_batch.device
    probs_list = []
    with torch.no_grad():
        for m in dialect_classifiers.values():
            m.to(device)
            m.eval()
            prob = m(mfcc_batch).squeeze(1)  # (B,)
            probs_list.append(prob)

    probs = torch.stack(probs_list, dim=1)  # (B, num_dialects)
    target = F.one_hot(dialect_ids, num_classes=probs.size(1)).float().to(device)
    return F.binary_cross_entropy(probs, target)


def compute_contrastive_loss(
    embeds: torch.Tensor,    # (B, D)
    ids:    torch.LongTensor,# (B,)
    margin: float = 1.0,
) -> torch.Tensor:
    B = embeds.size(0)
    if B < 2:
        return torch.tensor(0.0, device=embeds.device)

    # 1) 임베딩 정규화
    embeds = F.normalize(embeds, p=2, dim=1)   # (B, D)

    # 2) pairwise Euclidean distance
    dists = torch.cdist(embeds, embeds, p=2)   # (B, B)

    # 3) same / diff mask
    same = ids.unsqueeze(1) == ids.unsqueeze(0)   # (B, B)
    diff = ~same

    # 4) 자기 자신 제외
    eye = torch.eye(B, dtype=torch.bool, device=embeds.device)
    same &= ~eye
    diff &= ~eye

    # 5) pull loss (same-dialect)
    pos_loss = dists[same].pow(2).mean() if same.any() else torch.tensor(0.0, device=embeds.device)

    # 6) push loss (different-dialect)
    neg_term = F.relu(margin - dists[diff])
    neg_loss = neg_term.pow(2).mean() if diff.any() else torch.tensor(0.0, device=embeds.device)

    return pos_loss + neg_loss
