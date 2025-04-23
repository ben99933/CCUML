import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, from_logits=False, softmax=True, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.from_logits = from_logits
        self.softmax = softmax

    def forward(self, inputs, targets):
        if self.softmax:
            inputs = F.softmax(inputs, dim=1)[:, 1]
        elif self.from_logits:
            inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice_score