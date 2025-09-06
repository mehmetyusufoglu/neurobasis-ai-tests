from torchvision import models
from torch import nn
from typing import Literal

def get_resnet(name: Literal['resnet18','resnet34','resnet50']='resnet18', pretrained: bool=False, num_classes: int=10) -> nn.Module:
    """Return a torchvision ResNet with adjustable num_classes."""
    fn = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50
    }[name]
    model = fn(weights='DEFAULT' if pretrained else None)
    # Replace classifier
    if hasattr(model, 'fc'):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    return model
