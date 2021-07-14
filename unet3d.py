from torch import nn
import torch

class pub(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(pub, self).__init__()
        inter_channels = in_channels//2 if in_channels > out_channels else out_channels//2
        layers = [
                    nn.Conv3d(in_channels, inter_channels, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv3d(inter_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True)
                 ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm3d(inter_channels))
            layers.insert(len(layers)-1, nn.BatchNorm3d(out_channels))
        self.pub = nn.Sequential(*layers)

    def forward(self, x):
        return self.pub(x)


class unet3dDown(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(unet3dDown, self).__init__()
        self.pub = pub(in_channels, out_channels, batch_norm)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        x = self.pub(x)
        return x


class unet3dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet3dUp, self).__init__()
        self.pub = pub(in_channels//2+in_channels, out_channels, batch_norm)
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        x = self.sample(x)
        # c1 = (x1.size(2) - x.size(2)) // 2
        # c2 = (x1.size(3) - x.size(3)) // 2
        # x1 = x1[:, :, c1:-c1, c2:-c2, c2:-c2]
        x = torch.cat((x, x1), dim=1)
        x = self.pub(x)
        return x


class unet3d(nn.Module):
    def __init__(self, init_channels=1, class_nums=1, batch_norm=True, sample=True):
        super(unet3d, self).__init__()
        self.down1 = pub(init_channels, 64, batch_norm)
        self.down2 = unet3dDown(64, 128, batch_norm)
        self.down3 = unet3dDown(128, 256, batch_norm)
        self.down4 = unet3dDown(256, 512, batch_norm)
        self.up3 = unet3dUp(512, 256, batch_norm, sample)
        self.up2 = unet3dUp(256, 128, batch_norm, sample)
        self.up1 = unet3dUp(128, 64, batch_norm, sample)
        self.con_last = nn.Conv3d(64, class_nums, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.down1(x)#1*64*80*80*80
        x2 = self.down2(x1)#1*128*40*40*40
        x3 = self.down3(x2)#1*256*20*20*20
        x4 = self.down4(x3)#1*512*10*10*10
        x = self.up3(x4, x3)#1*256*20*20*20
        x = self.up2(x, x2)#1*128*40*40*40
        x = self.up1(x, x1)#1*64*80*80*80
        x = self.con_last(x)#1*3*80*80*80
        return self.softmax(x)#1*3*80*80*80

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn(1, 1, 80, 80, 80).to(device) # 这里的对应前面fforward的输入是32
    net = unet3d(1,3,batch_norm=True, sample=False).to(device)
    #Generate network structure figure
    from tensorboardX import SummaryWriter
    with SummaryWriter(comment='3D U-Net') as w:
        w.add_graph(net, inputs)
    with torch.no_grad():
        out = net(inputs)
    netsize=count_param(net)
    print(out.size(),"params:%0.3fM"%(netsize/1000000),"(%s)"%netsize)
    input("按任意键结束")