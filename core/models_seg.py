import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
from .model_utils import load_model, save_model


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out = 0, mode = None):
        super(BasicBlock, self).__init__()
        self.mode = mode
        self.drop = nn.Dropout3d(drop_out)
        self.conv = nn.Conv3d(in_channels, out_channels,3, stride=1,padding = 1, bias = False)
        self.bn = nn.BatchNorm3d(out_channels, affine = False, track_running_stats = False)
    def forward(self, x):
        out = self.conv(x)
        out = self.drop(out)
        out = F.relu(self.bn(out))
        if self.mode =="res":
            return out + x
        else:
            return out

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out = 0, mode = None):
        super(DenseBlock, self).__init__()
        self.growrate = out_channels//4
        self.conv1 = BasicBlock(in_channels,                self.growrate,drop_out,mode = mode)
        self.conv2 = BasicBlock(in_channels+self.growrate  ,self.growrate,drop_out,mode = mode)
        self.conv3 = BasicBlock(in_channels+self.growrate*2,self.growrate,drop_out,mode = mode)
        self.conv4 = BasicBlock(in_channels+self.growrate*3,self.growrate,drop_out,mode = mode)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.cat((x,x1),1)
        x3 = self.conv2(x2)
        x4 = torch.cat((x2,x3),1)
        x5 = self.conv3(x4)
        x6 = torch.cat((x4,x5),1)
        x7 = self.conv4(x6)
        return torch.cat((x1,x3,x5,x7),1)

class Dense_Unet_3L(nn.Module):
    def __init__(self,features,drop_out = 0.10,in_channel = 4, classes = 4, mode = None):
        super(Dense_Unet_3L, self).__init__()
        self.max_pool = nn.MaxPool3d((1,2,2),(1,2,2))
        self.conv1 = ConvBlock3D(in_channel,features  ,drop_out,mode = mode)
        self.dense1 = DenseBlock(features  ,features  ,drop_out,mode = mode)
        self.dense2 = DenseBlock(features*2,features  ,drop_out,mode = mode)
        self.dense3 = DenseBlock(features*3,features  ,drop_out,mode = mode)

        self.up_pool_1 = nn.ConvTranspose3d(features  ,features,(1,2,2),stride = (1,2,2))
        self.dense4 = DenseBlock(features*4,features  ,drop_out,mode = mode)
        self.up_pool_2 = nn.ConvTranspose3d(features  ,features,(1,2,2),stride = (1,2,2))
        self.conv2 = ConvBlock3D(features*2,features*2,drop_out,mode = mode)
        self.dense5 = DenseBlock(features*3,features  ,drop_out,mode = mode)
        self.conv_final = nn.Conv3d(features,classes,3,padding = 1, stride=1, bias = False)

    def load(self, path, filename, mode = "single", device = None):
        load_model(self, path = path, model_name = filename, mode = mode, device = device)

    def save(self, path, filename, optimizer = None):
        save_model(self, optimizer, path, filename)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.dense1(x1)
        x3 = torch.cat((x1,x2),1)
        x4 = self.max_pool(x3)
        x5 = self.dense2(x4)
        x6 = torch.cat((x4,x5),1)
        x7 = self.max_pool(x6)
        x8 = self.dense3(x7)

        x9 = self.up_pool_1(x8)
        x10= torch.cat((x6,x9),1)
        x11= self.dense4(x10)
        x12= self.up_pool_2(x11)
        x13= torch.cat((x3,x12),1)
        x14= self.dense5(x13)
        out = self.conv_final(x14)
        return out

class Dense_Unet_4L(nn.Module):
    def __init__(self,features,drop_out = 0.10,in_channel = 4, classes = 4, mode = None):
        super(Dense_Unet_4L, self).__init__()
        self.max_pool = nn.MaxPool3d((1,2,2),(1,2,2))
        self.conv1 = ConvBlock3D(in_channel,features,drop_out,mode = mode)
        self.dense1 = DenseBlock(features  ,features,drop_out,mode = mode)
        self.dense2 = DenseBlock(features*2,features,drop_out,mode = mode)
        self.dense3 = DenseBlock(features*3,features,drop_out,mode = mode)
        self.dense4 = DenseBlock(features*4,features,drop_out,mode = mode)

        self.up_pool_1 = nn.ConvTranspose3d(features,features,(1,2,2),stride = (1,2,2))
        self.dense5 = DenseBlock(features*5,features,drop_out,mode = mode)
        self.up_pool_2 = nn.ConvTranspose3d(features,features,(1,2,2),stride = (1,2,2))
        self.dense6 = DenseBlock(features*4,features,drop_out,mode = mode)
        self.up_pool_3 = nn.ConvTranspose3d(features,features,(1,2,2),stride = (1,2,2))
        self.dense7 = DenseBlock(features*3,features,drop_out,mode = mode)
        self.conv_final = nn.Conv3d(features,classes,3,padding = 1, stride=1, bias = False)

    def load(self, path, filename, mode = "single", device = None):
        load_model(self, path = path, model_name = filename, mode = mode, device = device)

    def save(self, path, filename, optimizer = None):
        save_model(self, optimizer, path, filename)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.dense1(x1)
        x3 = torch.cat((x1,x2),1)
        x4 = self.max_pool(x3)
        x5 = self.dense2(x4)
        x6 = torch.cat((x4,x5),1)
        x7 = self.max_pool(x6)
        x8 = self.dense3(x7)
        x9 = torch.cat((x7,x8),1)
        x10= self.max_pool(x9)
        x11= self.dense4(x10)

        x12= self.up_pool_1(x11)
        x13= torch.cat((x9,x12),1)
        x14= self.dense5(x13)
        x15= self.up_pool_2(x14)
        x16= torch.cat((x6,x15),1)
        x17= self.dense6(x16)
        x18= self.up_pool_3(x17)
        x19= torch.cat((x3,x18),1)
        x20= self.dense7(x19)
        out = self.conv_final(x20)
        return out

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, drop_out = 0, mode = None):
        super(ConvBlock3D, self).__init__()
        self.mode = mode
        self.drop = nn.Dropout3d(drop_out)
        self.conv = nn.Conv3d(in_channels, out_channels,3, stride=1,padding = 1, bias = False)
        self.bn = nn.BatchNorm3d(out_channels, affine = False, track_running_stats = False)
    def forward(self, x):
        out = self.conv(x)
        out = self.drop(out)
        out = F.relu(self.bn(out))
        if self.mode =="res":
            return out + x
        else:
            return out

class Unet_seg3D(nn.Module):
    def __init__(self,features,drop_out = 0.10,in_channel = 1, classes = 4, mode = None):
        super(Unet_seg3D, self).__init__()
        self.max_pool = nn.MaxPool3d(2,2)
        self.conv1 = ConvBlock3D(features,features,drop_out)
        self.conv2 = ConvBlock3D(features*2,features*2,drop_out,mode = mode)
        self.conv3 = ConvBlock3D(features*4,features*4,drop_out,mode = mode)
        self.conv4 = ConvBlock3D(features*8,features*8,drop_out,mode = mode)
        self.conv5 = ConvBlock3D(features*4,features*4,drop_out,mode = mode)
        self.conv6 = ConvBlock3D(features*2,features*2,drop_out,mode = mode)
        self.conv7 = ConvBlock3D(features  ,features  ,drop_out,mode = mode)
        self.down1 = ConvBlock3D(in_channel,features)
        self.down2 = ConvBlock3D(features  ,features*2,drop_out)
        self.down3 = ConvBlock3D(features*2,features*4,drop_out)
        self.down4 = ConvBlock3D(features*4,features*8,drop_out)
        self.up_pool_1 = nn.ConvTranspose3d(features*8,features*4,2,2)
        self.up_pool_2 = nn.ConvTranspose3d(features*4,features*2,2,2)
        self.up_pool_3 = nn.ConvTranspose3d(features*2,features  ,2,2)
        self.up1 = ConvBlock3D(features*8,features*4,drop_out)
        self.up2 = ConvBlock3D(features*4,features*2,drop_out)
        self.up3 = ConvBlock3D(features*2,features  ,drop_out)
        self.conv_final = nn.Conv3d(features,classes,3,stride = 1,padding = 1)

    def load(self, path, filename, mode = "single", device = None):
        load_model(self, path = path, model_name = filename, mode = mode, device = device)

    def save(self, path, filename, optimizer = None):
        save_model(self, optimizer, path, filename)

    def forward(self, x):
        x1 = self.down1(x)
        x1 = self.conv1(x1)
        x2 = self.max_pool(x1)
        x2 = self.down2(x2)
        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)
        x3 = self.down3(x3)
        x3 = self.conv3(x3)
        x4 = self.max_pool(x3)
        x4 = self.down4(x4)
        x4 = self.conv4(x4)

        x5 = self.up_pool_1(x4)
        x6 = torch.cat((x3,x5),1)
        x6 = self.up1(x6)
        x6 = self.conv5(x6)
        x7 = self.up_pool_2(x6)
        x8 = torch.cat((x2,x7),1)
        x8 = self.up2(x8)
        x8 = self.conv6(x8)
        x9 = self.up_pool_3(x8)
        x10 = torch.cat((x1,x9),1)
        x10 = self.up3(x10)
        x10 = self.conv7(x10)
        out = self.conv_final(x10)
        return out

class Unet_seg3D_2Dpool(nn.Module):
    def __init__(self,features,drop_out = 0.10,in_channel = 1, classes = 4, mode = None):
        super(Unet_seg3D_2Dpool, self).__init__()
        self.max_pool = nn.MaxPool3d((1,2,2),(1,2,2))
        self.conv1 = ConvBlock3D(features,features,drop_out)
        self.conv2 = ConvBlock3D(features*2,features*2,drop_out,mode = mode)
        self.conv3 = ConvBlock3D(features*4,features*4,drop_out,mode = mode)
        self.conv4 = ConvBlock3D(features*8,features*8,drop_out,mode = mode)
        self.conv5 = ConvBlock3D(features*4,features*4,drop_out,mode = mode)
        self.conv6 = ConvBlock3D(features*2,features*2,drop_out,mode = mode)
        self.conv7 = ConvBlock3D(features  ,features  ,drop_out,mode = mode)
        self.down1 = ConvBlock3D(in_channel,features)
        self.down2 = ConvBlock3D(features,features*2,drop_out)
        self.down3 = ConvBlock3D(features*2,features*4,drop_out)
        self.down4 = ConvBlock3D(features*4,features*8,drop_out)
        self.up_pool_1 = nn.ConvTranspose3d(features*8,features*4,(1,2,2),stride = (1,2,2))
        self.up_pool_2 = nn.ConvTranspose3d(features*4,features*2,(1,2,2),stride = (1,2,2))
        self.up_pool_3 = nn.ConvTranspose3d(features*2,features  ,(1,2,2),stride = (1,2,2))
        self.up1 = ConvBlock3D(features*8,features*4,drop_out)
        self.up2 = ConvBlock3D(features*4,features*2,drop_out)
        self.up3 = ConvBlock3D(features*2,features  ,drop_out)
        self.conv_final = nn.Conv3d(features,classes,3,padding = 1, stride=1, bias = False)

    def load(self, path, filename, mode = "single", device = None):
        load_model(self, path = path, model_name = filename, mode = mode, device = device)

    def save(self, path, filename, optimizer = None):
        save_model(self, optimizer, path, filename)

    def forward(self, x):
        x1 = self.down1(x)
        x1 = self.conv1(x1)
        x2 = self.max_pool(x1)
        x2 = self.down2(x2)
        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)
        x3 = self.down3(x3)
        x3 = self.conv3(x3)
        x4 = self.max_pool(x3)
        x4 = self.down4(x4)
        x4 = self.conv4(x4)

        x5 = self.up_pool_1(x4)
        x6 = torch.cat((x3,x5),1)
        x6 = self.up1(x6)
        x6 = self.conv5(x6)
        x7 = self.up_pool_2(x6)
        x8 = torch.cat((x2,x7),1)
        x8 = self.up2(x8)
        x8 = self.conv6(x8)
        x9 = self.up_pool_3(x8)
        x10 = torch.cat((x1,x9),1)
        x10 = self.up3(x10)
        x10 = self.conv7(x10)
        out = self.conv_final(x10)
        return out