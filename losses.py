import torch
import torch.nn as nn

# loss functions
class DiceLoss(nn.Module):
    def forward(self, input, target):
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        
        return 1 - ((2. * intersection + smooth) /
                  (iflat.sum() + tflat.sum() + smooth))

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, input, target):
        return self.alpha * self.dice_loss(input, target) + (1 - self.alpha) * self.bce_loss(input, target)


# If softmax as the final layer

def Dice(output, target, eps=1e-5):
        target = target.float()
        num = 2 * (output * target).sum()
        den = output.sum() + target.sum() + eps
        return 1.0 - num/den

class softmax_dice(nn.Module):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    def __init__(self):
        super(softmax_dice, self).__init__()
        
    def forward(self, output, target):
        target[target == 4] = 3 
        output = output.cuda()
        target = target.cuda()
        loss0 = Dice(output[:, 0, ...], (target == 0).float())
        loss1 = Dice(output[:, 1, ...], (target == 1).float())
        loss2 = Dice(output[:, 2, ...], (target == 2).float())
        loss3 = Dice(output[:, 3, ...], (target == 3).float())

        return loss1 + loss2 + loss3 + loss0 #, 1-loss1.data, 1-loss2.data, 1-loss3.data