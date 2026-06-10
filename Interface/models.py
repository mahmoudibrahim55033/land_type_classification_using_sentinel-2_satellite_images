import torchvision.models as models
import torch.nn as nn
from torchvision.models.googlenet import BasicConv2d

def create_model(num_classes=10, in_channels=8, pretrained=True, model_name='resnet50'):
    weights = 'IMAGENET1K_V1' if pretrained else None

    if model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'alexnet':
        model = models.alexnet(weights=weights)
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == 'googlenet':
        model = models.googlenet(weights=weights)
        model.transform_input = False
        model.conv1 = BasicConv2d(in_channels,64,kernel_size=7,stride=2,padding=3)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'efficientnet':
        model = models.efficientnet_b2(weights=weights)
        model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Model '{model_name}' is not supported. "
                        "Choose from: ['resnet50', 'alexnet', 'googlenet', 'efficientnet']")

    return model