import cv2, math
import torch
import torch.nn as nn

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class DilatedConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, padding, kernel_size):
        super(DilatedConvNet, self).__init__()
        self.dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):

        x = self.dilated_conv(x)
        x = self.relu(x)

        return x

class LAM(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.eca = eca_block(ch)
        self.conv1 = nn.Conv2d(6, 3, 3, padding=1)

    def forward(self, x):
        x = self.eca(x)
        x = self.conv1(x)
        return x

class RFEM(nn.Module):
    def __init__(
            self,
            ch_blocks=64,
            ch_mask=16,
    ):
        super().__init__()

        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(16, ch_blocks, 3, padding=1),
                                     nn.LeakyReLU(True))

        self.dconv1 = DilatedConvNet(ch_blocks,
                                  ch_blocks // 4,
                                  kernel_size=3,
                                  padding=1, dilation=1)
        self.dconv2 = DilatedConvNet(ch_blocks,
                                  ch_blocks // 4,
                                  kernel_size=3,
                                  padding=2, dilation=2)
        self.dconv3 = DilatedConvNet(ch_blocks,
                                  ch_blocks // 4,
                                  kernel_size=3,
                                  padding=3, dilation=3)
        self.dconv4 = nn.Conv2d(ch_blocks,
                                  ch_blocks // 4,
                                  kernel_size=7,
                                  padding=3)

        self.decoder = nn.Sequential(nn.Conv2d(ch_blocks, 16, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(16, 3, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     )

        self.lam = LAM(ch_mask)

    def forward(self, x):
        x1 = self.encoder(x)
        x1_1 = self.dconv1(x1)
        x1_2 = self.dconv2(x1)
        x1_3 = self.dconv3(x1)
        x1_4 = self.dconv4(x1)
        x1 = torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1)
        x1 = self.decoder(x1)
        out = x + x1
        out = torch.relu(out)
        mask = self.lam(torch.cat([x, out], dim=1))
        return out, mask

class ATEM(nn.Module):
    def __init__(self, in_ch=3, inter_ch=32, out_ch=3, kernel_size=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(True),
        )
        self.shift_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2))
        self.scale_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2))

        self.decoder = nn.Sequential(
            nn.Conv2d(inter_ch, out_ch, kernel_size, padding=kernel_size // 2))

    def forward(self, x, tag):
        x = self.encoder(x)
        scale = self.scale_conv(tag)
        shift = self.shift_conv(tag)
        x = x +(x * scale + shift)
        x = self.decoder(x)
        return x

class Trans_high(nn.Module):
    def __init__(self, in_ch=3, inter_ch=16, out_ch=3, kernel_size=3):
        super().__init__()
        self.atem = ATEM(in_ch, inter_ch, out_ch, kernel_size)
    def forward(self, x, tag):
        x = x + self.atem(x, tag)
        return x

class Up_tag(nn.Module):
    def __init__(self, kernel_size=1, ch=3):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ch,
                      ch,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels)

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                mode='reflect') 
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x
    def downsample(self, x):
        return x[:, :, ::2, ::2]
    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))
    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            image = up + level
        return image

class FAENet(nn.Module):
    def __init__(self,
                 num_high=1,
                 ch_blocks=32,
                 up_ksize=1,
                 high_ch=32,
                 high_ksize=3,
                 ch_mask=32,
                 gauss_kernel=7):
        super().__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)
        self.rfem = RFEM(ch_blocks, ch_mask)

        for i in range(0, self.num_high):
            self.__setattr__('up_tag_layer_{}'.format(i),
                             Up_tag(up_ksize, ch=3))
            self.__setattr__('trans_high_layer_{}'.format(i),
                             Trans_high(3, high_ch, 3, high_ksize))

    def forward(self, x):
        pyrs = self.lap_pyramid.pyramid_decom(img=x)

        trans_pyrs = []
        trans_pyr, tag = self.rfem(pyrs[-1])
        trans_pyrs.append(trans_pyr)

        commom_tag = []
        for i in range(self.num_high):
            tag = self.__getattr__('up_tag_layer_{}'.format(i))(tag)
            commom_tag.append(tag)

        for i in range(self.num_high):
            trans_pyr = self.__getattr__('trans_high_layer_{}'.format(i))(
                pyrs[-2 - i], commom_tag[i])
            trans_pyrs.append(trans_pyr)

        out = self.lap_pyramid.pyramid_recons(trans_pyrs)

        return out