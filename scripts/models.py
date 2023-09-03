"""Implement image classifier models for use in RSNA project."""
import torch
import torch.nn as nn


class CustomClassifier(nn.Module):
    """Implement simple extension model with a mobilenetv2 backbone."""

    def __init__(self, kernel_d=8, neck_size=32, img_size=512, train_backbone=False):
        super().__init__()
        mobilenet = torch.hub.load('pytorch/vision', 'mobilenet_v2',
                                   weights='MobileNet_V2_Weights.DEFAULT')
        self.backbone = mobilenet.features
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.gap = nn.AvgPool2d(kernel_size=(kernel_d, kernel_d))

        # Using dictionaries to store layers for elegance
        organs = ['bowel', 'extra', 'liver', 'kidney', 'spleen']
        self.necks = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        # Figure out what size the first linear web should be
        with torch.no_grad():
            _x = torch.randn(1, 3, img_size, img_size)
            _x = self.gap(self.backbone(_x))
            linear_start_dim = _x.view(_x.size(0), -1).shape[1]

        # Define neck/head structure
        for organ in organs:
            self.necks[organ] = nn.Sequential(nn.Linear(linear_start_dim, neck_size), nn.SiLU())
            if organ in ['bowel', 'extra']:
                self.heads[organ] = nn.Sequential(nn.Linear(neck_size, 1), nn.Sigmoid())
            else:
                self.heads[organ] = nn.Sequential(nn.Linear(neck_size, 3), nn.Softmax(dim=1))

    def forward(self, x):
        """Run the Network."""
        x = self.backbone(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten

        outputs = {}
        for organ, neck in self.necks.items():
            outputs[organ] = self.heads[organ](neck(x))
        return outputs


if __name__ == '__main__':
    net = CustomClassifier()
    data = torch.randn(1, 3, 512, 512)
    print(net(data))
    for _net in net(data).values():
        print(_net)
        print(_net.shape)
