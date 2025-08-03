import torch
from torchvision import models as resnet_model
import torch.nn as nn

from model.BSBlock import BasicStage

from model.transformer import TransformerModel
from thop import profile, clever_format


class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(in_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)

        return self.sigmoid(x)


class HTCNet(nn.Module):
    def __init__(self, n_channels=3, num_classes=9, heads=8, dim=128, depth=(3, 3, 3), patch_size=2):
        super(HTCNet, self).__init__()
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.heads = heads
        self.depth = depth
        self.dim = dim
        mlp_dim = [2 * dim, 4 * dim, 8 * dim, 16 * dim]
        embed_dim = [dim, 2 * dim, 4 * dim, 8 * dim]
        resnet = resnet_model.resnet34(weights=resnet_model.ResNet34_Weights.DEFAULT)  # pretrained = True
        self.vit_1 = TransformerModel(dim=embed_dim[0], mlp_dim=mlp_dim[0], depth=depth[0], heads=heads)# dim=128, mlp_dim=256, depth=3, heads=8
        self.vit_2 = TransformerModel(dim=embed_dim[1], mlp_dim=mlp_dim[1], depth=depth[1], heads=heads)
        self.vit_3 = TransformerModel(dim=embed_dim[2], mlp_dim=mlp_dim[2], depth=depth[2], heads=heads)
        self.patch_embed_1 = nn.Conv2d(n_channels, embed_dim[0], kernel_size=2 * patch_size, stride=2 * patch_size)
        self.patch_embed_2 = nn.Conv2d(embed_dim[0], embed_dim[1], kernel_size=patch_size, stride=patch_size)
        self.patch_embed_3 = nn.Conv2d(embed_dim[1], embed_dim[2], kernel_size=patch_size, stride=patch_size)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        depths = (1, 2, 4)
        dpr = [x.item()
               for x in torch.linspace(0, 0.1, sum(depths))]
        self.BS_1 = BasicStage(
            dim=int(8 * dim),
            stage=0,
            depth=1,
            att_kernel=11,
            mlp_ratio=2.,
            drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU,
        )
        self.BS_2 = BasicStage(
            dim=int(4 * dim),
            stage=1,
            depth=2,
            att_kernel=11,
            mlp_ratio=2.,
            drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU,
        )
        self.BS_3 = BasicStage(
            dim=int(2 * dim),
            stage=2,
            depth=4,
            att_kernel=11,
            mlp_ratio=2.,
            drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU,
        )

        self.CA3 = CAB(8*dim)
        self.CA2 = CAB(8*dim)
        self.CA1 = CAB(4*dim)

        self.SA = SAB()


        self.decoder1 = DecoderBottleneckLayer(8 * dim)
        self.decoder2 = DecoderBottleneckLayer(8 * dim)
        self.decoder3 = DecoderBottleneckLayer(4 * dim)
        self.up3_1 = nn.ConvTranspose2d(8 * dim, 4 * dim, kernel_size=2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(8 * dim, 2 * dim, kernel_size=2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(4 * dim, dim, kernel_size=4, stride=4)
        self.out = nn.Conv2d(dim, num_classes, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        patch_size = self.patch_size
        dim = self.dim
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        print("X:", x.shape)

        v1 = self.patch_embed_1(x)
        v1 = v1.permute(0, 2, 3, 1).contiguous()
        v1 = v1.view(b, -1, dim)
        v1 = self.vit_1(v1)
        print("v1:", v1.shape)
        v1_cnn = v1.view(b, int(h / (2 * patch_size)), int(w / (2 * patch_size)), dim)
        print("v1_cnn.shape", v1_cnn.shape)
        v1_cnn = v1_cnn.permute(0, 3, 1, 2).contiguous()
        print("v1_cnn.shape", v1_cnn.shape)

        v2 = self.patch_embed_2(v1_cnn)
        v2 = v2.permute(0, 2, 3, 1).contiguous()
        v2 = v2.view(b, -1, 2 * dim)
        v2 = self.vit_2(v2)
        v2_cnn = v2.view(b, int(h / (patch_size * 2 * 2)), int(w / (2 * 2 * patch_size)), dim * 2)
        v2_cnn = v2_cnn.permute(0, 3, 1, 2).contiguous()
        print(v2_cnn.shape)

        v3 = self.patch_embed_3(v2_cnn)
        v3 = v3.permute(0, 2, 3, 1).contiguous()
        v3 = v3.view(b, -1, 4 * dim)
        v3 = self.vit_3(v3)
        v3_cnn = v3.view(b, int(h / (patch_size * 2 * 2 * 2)), int(w / (2 * 2 * 2 * patch_size)), dim * 2 * 2)
        v3_cnn = v3_cnn.permute(0, 3, 1, 2).contiguous()
        print(v3_cnn.shape)

        cat_1 = torch.cat([v3_cnn, e4], dim=1)
        cat_1 = self.BS_1(cat_1)
        cat_1 = self.CA3(cat_1) * cat_1
        cat_1 = self.SA(cat_1) * cat_1
        cat_1 = self.decoder1(cat_1)
        cat_1 = self.up3_1(cat_1)

        cat_2 = torch.cat([v2_cnn, e3], dim=1)
        cat_2 = self.BS_2(cat_2)
        cat_2 = torch.cat([cat_2, cat_1], dim=1)
        cat_2 = self.CA2(cat_2) * cat_2
        cat_2 = self.SA(cat_2) * cat_2
        cat_2 = self.decoder2(cat_2)
        cat_2 = self.up2_1(cat_2)
        print("cat_2.shape", cat_2.shape)

        cat_3 = torch.cat([v1_cnn, e2], dim=1)
        cat_3 = self.BS_3(cat_3)
        print("cat_3.shape", cat_3.shape)
        cat_3 = torch.cat([cat_3, cat_2], dim=1)
        cat_3 = self.CA1(cat_3) * cat_3
        cat_3 = self.SA(cat_3) * cat_3
        cat_3 = self.decoder3(cat_3)
        cat_3 = self.up1_1(cat_3)
        out = self.out(cat_3)

        return out


if __name__ == '__main__':
    data = torch.randn(1, 3, 224, 224)
    model = HTCNet(num_classes=9)
    print(model(data).shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    flops, params = profile(model, inputs=(data,))
    flops, params = clever_format([flops, params], '%.3f')
    print(f"Params：{params}, FLOPs：{flops}")

