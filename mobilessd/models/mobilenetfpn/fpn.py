import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2FPN(nn.Module):
    def __init__(self, n_class=1000, input_size=512, width_mult=1.):
        super(MobileNetV2FPN, self).__init__()

        # building first layer
        assert input_size % 64 == 0

        self.width_mult = width_mult
        self._input_channel = int(32 * width_mult)

        self.c1 = conv_bn(3, self._input_channel, 2)

        # building inverted residual blocks
        # input [32 x 256 x 256] input size assumed to be 512
        self.r1 = self._make_residual_layer(1, 16, 1, 1)  # output [16 x 256 x 112]
        self.r2 = self._make_residual_layer(6, 24, 2, 2)  # output [24 x 128 x 128]
        self.r3 = self._make_residual_layer(6, 32, 3, 1)  # output [32 x 128 x 128]

        r4_output_channel = 64
        self.r4 = self._make_residual_layer(6, r4_output_channel, 4, 2)  # output [64 x 64 x 64]

        r5_output_channel = 96
        self.r5 = self._make_residual_layer(6, r5_output_channel, 3, 2)  # output [96 x 32 x 32]

        r6_output_channel = 160
        self.r6 = self._make_residual_layer(6, r6_output_channel, 3, 2)  # output [160 x 16 x 16]

        r7_output_channel = 320
        self.r7 = self._make_residual_layer(6, r7_output_channel, 1, 2)  # output [320 x 8 x 8]

        r8_output_channel = 640
        self.r8 = self._make_residual_layer(6, r8_output_channel, 1, 2)  # output [640 x 4 x 4]

        self.c9 = nn.Conv2d(r8_output_channel, 256, kernel_size=3, stride=2, padding=1)  # output [256 x 2 x 2]
        self.c10 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)  # output [256 x 1 x 1]

        # Top-down layers
        self.toplayer = nn.Conv2d(r8_output_channel, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(r7_output_channel, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(r6_output_channel, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(r5_output_channel, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(r4_output_channel, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self._initialize_weights()

    def _make_residual_layer(self, expand_ratio, channels, n, stride):
        layers = []
        output_channel = int(channels * self.width_mult)
        for i in range(n):
            if i == 0:
                # inp, oup, stride, expand_ratio
                layers.append(InvertedResidual(self._input_channel, output_channel, stride, expand_ratio))
            else:
                layers.append(InvertedResidual(self._input_channel, output_channel, 1, expand_ratio))
            self._input_channel = output_channel
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        c1 = self.c1(x)

        r1 = self.r1(c1)
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        r4 = self.r4(r3)
        r5 = self.r5(r4)
        r6 = self.r6(r5)
        r7 = self.r7(r6)
        r8 = self.r8(r7)

        p9 = self.c9(r8)
        p10 = self.c10(p9)

        p8 = self.toplayer(r8)  # out [256 x 16 x 16]
        p7 = self._upsample_add(p8, self.latlayer1(r7))
        p6 = self._upsample_add(p7, self.latlayer2(r6))
        p5 = self._upsample_add(p6, self.latlayer3(r5))
        p4 = self._upsample_add(p5, self.latlayer4(r4))

        p7 = self.smooth1(p7)
        p6 = self.smooth2(p6)
        p5 = self.smooth3(p5)
        p4 = self.smooth4(p4)

        # x = x.view(-1, self.last_channel)
        # x = self.classifier(x)

        return p4, p5, p6, p7, p8, p9, p10

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def test():
    net = MobileNetV2FPN()
    fms = net(torch.randn(1, 3, 512, 512))
    for fm in fms:
        print(fm.size())


#test()