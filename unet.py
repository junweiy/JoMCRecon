import torch
import torch.fft
from signal_utils import torch_fft, torch_ifft

class ConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel,batch_normalization=True):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel,output_channel,3,padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()
        self.batch_normalization = batch_normalization

    def forward(self,x):
        x = self.conv1(x)
        if self.batch_normalization:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.batch_normalization:
            x = self.bn2(x)

        x=self.relu(x)

        return x

class DownSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(DownSample, self).__init__()
        self.down_sample = torch.nn.MaxPool2d(factor, factor)

    def forward(self,x):
        return self.down_sample(x)


class UpSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(UpSample, self).__init__()
        self.up_sample = torch.nn.Upsample(scale_factor=factor, mode='bilinear')
    def forward(self,x):
        return self.up_sample(x)


class CropConcat(torch.nn.Module):
    def __init__(self,crop = True):
        super(CropConcat, self).__init__()
        self.crop = crop

    def do_crop(self,x, tw, th):
        b,c,w, h = x.size()
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return x[:,:,x1:x1 + tw, y1:y1 + th]

    def forward(self,x,y):
        b, c, h, w = y.size()
        if self.crop:
            x = self.do_crop(x,h,w)
        return torch.cat((x,y),dim=1)

class ResUNet(torch.nn.Module):
    def __init__(self,input_channel,output_channel):
        super(ResUNet, self).__init__()
        self.input_channel = input_channel

        #Down Blocks
        self.conv_block1 = ConvBlock(input_channel,64)
        self.conv_block2 = ConvBlock(64,128)
        self.conv_block3 = ConvBlock(128,256)
        self.conv_block4 = ConvBlock(256,512)
        self.conv_block5 = ConvBlock(512,1024)

        #Up Blocks
        self.conv_block6 = ConvBlock(1024+512, 512)
        self.conv_block7 = ConvBlock(512+256, 256)
        self.conv_block8 = ConvBlock(256+128, 128)
        self.conv_block9 = ConvBlock(128+64, 64)

        #Last convolution
        self.last_conv = torch.nn.Conv2d(64, output_channel,1)

        self.crop = CropConcat()

        self.downsample = DownSample()
        self.upsample =   UpSample()

    def forward(self, x, mask):
        if self.input_channel == 2:
            t2u_input = x[:,1].view([x.shape[0], 1, x.shape[2], x.shape[3]]).clone()
        elif self.input_channel == 1:
            t2u_input = x.clone()
        elif self.input_channel == 6:
            t2u_input = x[:,3:].clone()
        if torch.is_complex(x):
            x = abs(x).type(torch.float)
        x1 = self.conv_block1(x) #  64      192 192

        x = self.downsample(x1)  #  64      96  96
        x2 = self.conv_block2(x) # 128      96  96

        x= self.downsample(x2)   # 128      48  48
        x3 = self.conv_block3(x) # 256      48  48

        x= self.downsample(x3)   # 256      24  24
        x4 = self.conv_block4(x) # 512      24  24

        x = self.downsample(x4)  # 512      12  12
        x5 = self.conv_block5(x) #1024      12  12

        x = self.upsample(x5)    #1024      24  24
        x6 = self.crop(x4, x)     #1024+512  24  24
        x6 = self.conv_block6(x6)  # 512      24  24

        x = self.upsample(x6)     # 512      48  48
        x7 = self.crop(x3,x)      # 512+256  48  48
        x7 = self.conv_block7(x7)  # 256      48  48

        x = self.upsample(x7)     # 256      96  96
        x8 = self.crop(x2,x)       # 256+128  96  96
        x8 = self.conv_block8(x8)   # 128      96  96

        x = self.upsample(x8)      # 128     192 192
        x = self.crop(x1,x)       # 128+64  192 192
        x = self.conv_block9(x)   #  64     192 192

        x = self.last_conv(x) + t2u_input    #   1     192 192

        if mask is not None:
            DCed = self.DC(x, t2u_input, mask)
            return DCed
        else:
            return x


    def DC(self, pred, x_input, mask):
        ffted_pred = torch_fft(pred, (-2, -1))
        ffted_input = torch_fft(x_input, (-2, -1))
        combined = ffted_pred * (1 - mask) + ffted_input * mask
        combined = torch_ifft(combined, (-2, -1))

        return abs(combined).type(torch.float)


