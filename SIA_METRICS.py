import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score



class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        dice_per_image = []
        for input_, target in zip(inputs, targets):
            intersection = (input_ * target).sum()
            dice = (2.*intersection + smooth)/(input_.sum() + target.sum() + smooth)
            dice_per_image.append(1 - dice)
        return torch.mean(torch.stack(dice_per_image))
    
    
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        iou_per_image = []
        for input_, target in zip(inputs, targets):
            intersection = (input_ * target).sum()
            total = (input_ + target).sum()
            union = total - intersection
            iou = (intersection + smooth) / (union + smooth)
            iou_per_image.append(1 - iou)
        
        return torch.mean(torch.stack(iou_per_image))

class FocalLoss(nn.Module):
    def __init__(self, init_alpha=0.25, gamma=2, max_alpha=1.0):
        super(FocalLoss, self).__init__()
        self.alpha_logit = nn.Parameter(torch.log(torch.tensor(init_alpha / (1 - init_alpha))))
        self.gamma = gamma
        self.max_alpha = max_alpha

    def forward(self, inputs, targets):
        alpha = torch.sigmoid(self.alpha_logit) * self.max_alpha
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)

        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        return torch.mean(F_loss)

class AutoDFLoss(nn.Module):
    def __init__(self, init_alpha=0.25, gamma=2, max_alpha=1.0, dice_weight=1.0, focal_weight=1.0):
        super(AutoDFLoss, self).__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(init_alpha=init_alpha, gamma=gamma, max_alpha=max_alpha)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets, smooth=1e-6):
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        dice_focal_per_image = []
        for input_, target in zip(inputs, targets):
            intersection = (input_ * target).sum()
            dice_loss = 1 - (2. * intersection + smooth) / (input_.sum() + target.sum() + smooth)
            focal_loss = self.focal(input_.unsqueeze(0), target.unsqueeze(0)).squeeze()
            dice_focal = (self.dice_weight * dice_loss) / (self.dice_weight + self.focal_weight) + \
                         (self.focal_weight * focal_loss) / (self.dice_weight + self.focal_weight)
            dice_focal_per_image.append(dice_focal)
        return torch.mean(torch.stack(dice_focal_per_image))
    
def Threshold_DiceLoss(inputs, targets, thresh=0.5, smooth=1e-8):
    inputs = 1*(inputs >= thresh)
    inputs = inputs.view(inputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    dice_per_image = []
    for input_, target in zip(inputs, targets):
        intersection = (input_ * target).sum()
        dice = (2.*intersection + smooth)/(input_.sum() + target.sum() + smooth)
        dice_per_image.append(1 - dice)
    return torch.mean(torch.stack(dice_per_image))


def Threshold_IoULoss(inputs, targets, thresh=0.5, smooth=1e-6):
    inputs = 1*(inputs >= thresh)
    inputs = inputs.view(inputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    iou_per_image = []
    for input_, target in zip(inputs, targets):
        intersection = (input_ * target).sum()
        total = (input_ + target).sum()
        union = total - intersection
        iou = (intersection + smooth)/(union + smooth)
        iou_per_image.append(1 - iou)
    return torch.mean(torch.stack(iou_per_image))


def custom_precision_score(inputs, targets, thresh=0.5, smooth=1e-6):
    inputs = 1*(inputs >= thresh)
    inputs = inputs.view(inputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    precision_per_image = []
    for input_, target in zip(inputs, targets):
        TP = ((input_ == 1) & (target == 1))
        FP = ((input_ == 1) & (target == 0))
        precision = torch.sum(TP.float())/(torch.sum(TP.float())+torch.sum(FP.float())+smooth)
        precision_per_image.append(precision)
    return torch.mean(torch.stack(precision_per_image))


def custom_recall_score(inputs, targets, thresh=0.5, smooth=1e-6):
    inputs = 1*(inputs >= thresh)
    inputs = inputs.view(inputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    recall_per_image = []
    for input_, target in zip(inputs, targets):
        TP = ((input_ == 1) & (target == 1))
        FN = ((input_ == 0) & (target == 1))
        recall = torch.sum(TP.float())/(torch.sum(TP.float())+torch.sum(FN.float()) + smooth)
        recall_per_image.append(recall)
    return torch.mean(torch.stack(recall_per_image))
