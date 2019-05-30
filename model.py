import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

SAVE_EVERY = 10

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

# inception net v3
def inceptionnet(num_classes):
    model = models.inception_v3(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Sequential(nn.Linear(num_features, 1024),
                                        nn.Linear(1024, num_classes))
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 1024),
                            nn.Linear(1024, num_classes))
    return model

# inception net v1 - what James used, PyTorch implementation in progress
def googlenet(num_classes):
    model = models.googlenet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Sequential(nn.Linear(num_features, 1024),
                                        nn.Linear(1024, num_classes))
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 1024),
                            nn.Linear(1024, num_classes))
    return model