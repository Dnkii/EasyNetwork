import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url




__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),#1*64*55*55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),#1*64*27*27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),#1*192*27*27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),#1*192*13*13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),#1*384*13*13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),#1*256*13*13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),#1*256*13*13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),#1*256*6*6
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))#1*256*6*6
        self.classifier = nn.Sequential(
            nn.Dropout(),#1*9216
            nn.Linear(256 * 6 * 6, 4096),#1*4096
            nn.ReLU(inplace=True),
            nn.Dropout(),#1*4096
            nn.Linear(4096, 4096),#1*4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),#1*1000
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)#1*256*6*6
        x = self.avgpool(x)#1*256*6*6
        x = torch.flatten(x, 1)#1*9216
        x = self.classifier(x)#1*1000
        return x


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn(1, 3, 244, 244).to(device) # 这里的对应前面fforward的输入是32
    net = alexnet().to(device)
    #Generate network structure figure
    # from tensorboardX import SummaryWriter
    # with SummaryWriter(comment='AlexNet') as w:
    #     w.add_graph(net, inputs)
    out = net(inputs)
    netsize=count_param(net)
    print(out.size(),"params:%0.3fM"%(netsize/1000000),"(%s)"%netsize)
    input("按任意键结束")