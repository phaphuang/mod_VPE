import torch.nn as nn
from torchvision.models.resnet import Bottleneck, ResNet

def modify_resnet_model(model, *, cifar_stem=True, v1=True):
    assert isinstance(model, ResNet)
    if cifar_stem:
        conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")
        model.conv1 = conv1
        model.maxpool = nn.Identity()
    if v1:
        for l in range(2, 5):
            layer = getattr(model, "layer{}".format(l))
            block = list(layer.children())[0]
            if isinstance(block, Bottleneck):
                assert block.conv1.kernel_size == (1, 1) and block.conv1.stride == (1, 1)
                assert block.conv2.kernel_size == (3, 3) and block.conv2.stride == (2, 2)
                assert block.conv2.dilation == (1, 1), "Currently, only models with dilation=1 are supported"
                block.conv1.stride = (2, 2)
                block.conv2.stride = (1, 1)
    return model