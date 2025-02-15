from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # Squeeze 구현
        self.pool = nn.AdaptiveAvgPool2d(1) # Tensorflow의 Global AveragePooling으로 활용하는 법
        # Excitation part 구현
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b,c)
        y = self.fc(y).view(b, c, 1, 1) # torch.view()는 contigious해야 동작, tensor 공간 유지하면서 shape만 다르게.
        return x * y.expand_as(x) # Expand y tensor to the same size as x(

