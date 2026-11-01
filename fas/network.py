"""
MiniFASNet from Minivision's Silent-Face-Anti-Spoofing (Apache-2.0), rewritten compactly.
Layer names match the original exactly, so the released weights load unchanged.
Only the two variants with released weights are kept: MiniFASNetV2 and MiniFASNetV1SE.
"""
from torch import nn

KEEP_V1 = [32, 32, 103, 103, 64, 13, 13, 64, 26, 26, 64, 13, 13, 64, 52, 52, 64, 231, 231, 128,
           154, 154, 128, 52, 52, 128, 26, 26, 128, 52, 52, 128, 26, 26, 128, 26, 26, 128, 308, 308,
           128, 26, 26, 128, 26, 26, 128, 512, 512]
KEEP_V2 = [32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 231, 231, 128,
           231, 231, 128, 52, 52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26, 128, 308, 308,
           128, 26, 26, 128, 26, 26, 128, 512, 512]
ARCHS = {"MiniFASNetV2": (KEEP_V2, False, 0.2), "MiniFASNetV1SE": (KEEP_V1, True, 0.75)}  # keep, SE, dropout


class ConvBlock(nn.Module):
    """conv -> batch norm (-> PReLU)."""

    def __init__(self, cin, cout, k=1, s=1, p=0, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.prelu = nn.PReLU(cout) if act else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.prelu(x) if self.prelu is not None else x


class SE(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, ch, r):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1, self.bn1, self.relu = nn.Conv2d(ch, ch // r, 1, bias=False), nn.BatchNorm2d(ch // r), nn.ReLU(True)
        self.fc2, self.bn2, self.sigmoid = nn.Conv2d(ch // r, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.Sigmoid()

    def forward(self, x):
        w = self.relu(self.bn1(self.fc1(self.avg_pool(x))))
        return x * self.sigmoid(self.bn2(self.fc2(w)))


class DepthWise(nn.Module):
    """1x1 expand -> 3x3 depthwise -> 1x1 project, optional residual + SE."""

    def __init__(self, c1, c2, c3, residual=False, stride=2, se=0):
        super().__init__()
        self.conv = ConvBlock(*c1)
        self.conv_dw = ConvBlock(*c2, k=3, s=stride, p=1, groups=c2[0])
        self.project = ConvBlock(*c3, act=False)
        self.residual = residual
        self.se_module = SE(c3[1], se) if se else None

    def forward(self, x):
        y = self.project(self.conv_dw(self.conv(x)))
        if not self.residual:
            return y
        return x + (self.se_module(y) if self.se_module is not None else y)


def _triple(k, b):
    return (k[b], k[b + 1]), (k[b + 1], k[b + 2]), (k[b + 2], k[b + 3])


class Residual(nn.Module):
    """n residual DepthWise blocks; with SE, only the last block gets it (as in the original)."""

    def __init__(self, k, start, n, se=0):
        super().__init__()
        self.model = nn.Sequential(*[DepthWise(*_triple(k, start + 3 * i), residual=True, stride=1,
                                               se=se if i == n - 1 else 0) for i in range(n)])

    def forward(self, x):
        return self.model(x)


class MiniFASNet(nn.Module):
    def __init__(self, keep, conv6_kernel=(5, 5), drop_p=0.2, se=False, num_classes=3, embedding=128):
        super().__init__()
        k, r = keep, (4 if se else 0)
        self.conv1 = ConvBlock(3, k[0], k=3, s=2, p=1)
        self.conv2_dw = ConvBlock(k[0], k[1], k=3, s=1, p=1, groups=k[1])
        self.conv_23 = DepthWise(*_triple(k, 1))
        self.conv_3 = Residual(k, 4, 4, r)
        self.conv_34 = DepthWise(*_triple(k, 16))
        self.conv_4 = Residual(k, 19, 6, r)
        self.conv_45 = DepthWise(*_triple(k, 37))
        self.conv_5 = Residual(k, 40, 2, r)
        self.conv_6_sep = ConvBlock(k[46], k[47])
        self.conv_6_dw = ConvBlock(k[47], k[48], k=conv6_kernel, groups=k[48], act=False)
        self.linear = nn.Linear(512, embedding, bias=False)
        self.bn = nn.BatchNorm1d(embedding)
        self.drop = nn.Dropout(drop_p)
        self.prob = nn.Linear(embedding, num_classes, bias=False)

    def forward(self, x, return_feat=False):
        x = self.conv_23(self.conv2_dw(self.conv1(x)))
        feat = self.conv_4(self.conv_34(self.conv_3(x)))  # 128 x 10 x 10, used by the FFT head
        x = self.conv_6_dw(self.conv_6_sep(self.conv_5(self.conv_45(feat)))).flatten(1)
        logits = self.prob(self.drop(self.bn(self.linear(x))))
        return (logits, feat) if return_feat else logits


def build(arch, h=80, w=80):
    keep, se, drop = ARCHS[arch]
    return MiniFASNet(keep, conv6_kernel=((h + 15) // 16, (w + 15) // 16), drop_p=drop, se=se)
