import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)#1*64*512*512
        self.pool1 = nn.MaxPool2d(2)#1*64*256*256
        self.conv2 = DoubleConv(64, 128)#1*128*256*256
        self.pool2 = nn.MaxPool2d(2)#1*128*128*128
        self.conv3 = DoubleConv(128, 256)#1*256*128*128
        self.pool3 = nn.MaxPool2d(2)#1*256*64*64
        self.conv4 = DoubleConv(256, 512)#1*512*64*64
        self.pool4 = nn.MaxPool2d(2)#1*512*32*32
        self.conv5 = DoubleConv(512, 1024)#1*1024*32*32
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)#1*512*64*64
        self.conv6 = DoubleConv(1024, 512)#1*512*64*64
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)#1*256*128*128
        self.conv7 = DoubleConv(512, 256)#1*256*128*128
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)#1*128*256*256
        self.conv8 = DoubleConv(256, 128)#1*128*256*256
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)#1*64*512*512
        self.conv9 = DoubleConv(128, 64)#1*64*512*512
        self.conv10 = nn.Conv2d(64,out_ch, 1)#1*1*512*512

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)#1*512*64*64
        merge6 = torch.cat([up_6, c4], dim=1)#1*1024*64*64
        c6=self.conv6(merge6)#1*512*64*64
        up_7=self.up7(c6)#1*256*128*128
        merge7 = torch.cat([up_7, c3], dim=1)#1*512*128*128
        c7=self.conv7(merge7)#1*256*128*128
        up_8=self.up8(c7)#1*128*256*256
        merge8 = torch.cat([up_8, c2], dim=1)#1*256*256*256
        c8=self.conv8(merge8)#1*128*256*256
        up_9=self.up9(c8)#1*64*512*512
        merge9=torch.cat([up_9,c1],dim=1)#1*128*512*512
        c9=self.conv9(merge9)#1*64*512*512
        c10=self.conv10(c9)#1*1*512*512
        out = nn.Sigmoid()(c10)#1*1*512*512
        return out

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn(1, 3, 512, 512).to(device) # 这里的对应前面fforward的输入是32
    net = Unet(3,1).to(device)
    #Generate network structure figure
    # from tensorboardX import SummaryWriter
    # with SummaryWriter(comment='U-Net') as w:
    #     w.add_graph(net, inputs)
    out = net(inputs)
    netsize=count_param(net)
    print(out.size(),"params:%0.3fM"%(netsize/1000000),"(%s)"%netsize)
    input("按任意键结束")








