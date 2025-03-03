from torch import nn
from SELayer import SELayer
from torchvision.models import Inception3

class SEInception(nn.Module):
    def __init__(self, num_classes, aux_logits=True, transform_input=False):
        super().__init__()
        model = Inception3(num_classes=num_classes, aux_logits=aux_logits, transform_input=transform_input)
        model.Mixed_5b.add_module("SELayer", SELayer(192))
        model.Mixed_5c.add_module("SELayer", SELayer(256))
        model.MIxed_5d.add_module("SELayer", SELayer(288))
        model.Mixed_6a.add_module("SELayer", SELayer(288))
        model.Mixed_6b.add_module("SELayer", SELayer(768))
        model.Mixed_6c.add_module("SELayer", SELayer(768))
        model.Mixed_6d.add_module("SELayer", SELayer(768))
        model.Mixed_6e.add_module("SELayer", SELayer(768))
        if aux_logits:
            model.AuxLogits.add_module("SELayer", SELayer(768))
        model.Mixed_7a.add_module("SELayer", SELayer(768))
        model.Mixed_7b.add_module("SELayer", SELayer(1280))
        model.Mixed_7c.add_module("SELayer", SELayer(2048))
        self.model = model

    def forward(self, x):
        _, _, h, w = x.size()
        if (h, w) != (299, 299):
            raise ValueError('입력 크기는 299x299를 유지하시시오!')
        return self.model(x)