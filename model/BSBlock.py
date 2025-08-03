from typing import List

import antialiased_cnns
import torch
from timm.layers import DropPath
from torch import nn, Tensor
import torch.nn.functional as F


class DRFD(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = act_layer()
        self.norm_c = norm_layer(dim * 2)
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = norm_layer(dim * 2)
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)

    def forward(self, x):  # x = [B, C, H, W]

        x = self.conv(x)  # x = [B, 2C, H, W]
        max = self.norm_m(self.max_m(x))  # m = [B, 2C, H/2, W/2]
        conv = self.norm_c(self.act_c(self.conv_c(x)))  # c = [B, 2C, H/2, W/2]
        x = torch.cat([conv, max], dim=1)  # x = [B, 2C+2C, H/2, W/2]  -->  [B, 4C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 4C, H/2, W/2]     -->  [B, 2C, H/2, W/2]

        return x


# 不删
class PA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            norm_layer(dim * 4),
            act_layer(),
            nn.Conv2d(dim * 4, dim, 1, bias=False)
        )
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        att = self.p_conv(x)
        x = x * self.gate_fn(att)

        return x


# 不删
class LA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            norm_layer(dim),
            act_layer()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# 不删
class MRA(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = antialiased_cnns.BlurPool(channel, stride=3)
        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.norm = norm_layer(channel)

    def forward(self, x):
        x_tem = self.max_m1(x)
        x_tem = self.max_m2(x_tem)
        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)
        x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
        x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))

        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)

        out = x[:, :self.channel, :, :] * F.interpolate(self.gate_fn(att),
                                                        size=(x.shape[-2], x.shape[-1]),
                                                        mode='nearest')
        return out

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


# 不删
class GA12(nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = act_layer()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x_ = self.proj_1(x_)
        x_ = self.activation(x_)
        attn1 = self.conv0(x_)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        x_ = x_ * attn
        x_ = self.proj_2(x_)
        x = self.uppool(x_, indices=idx)
        return x


# 不删
class D_GA(nn.Module):

    def     __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = GA(dim)
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x = self.norm(self.attn(x_))
        x = self.uppool(x, indices=idx)

        return x


# 不删
class GA(nn.Module):
    def __init__(self, dim, head_dim=4, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)
        return x


class LWGA_Block(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 att_kernel,
                 mlp_ratio,
                 drop_path,
                 act_layer,
                 norm_layer,
                 ):
        super().__init__()
        self.stage = stage
        self.dim_split = dim // 4
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.PA = PA(self.dim_split, norm_layer, act_layer)  # PA is point attention
        self.LA = LA(self.dim_split, norm_layer, act_layer)  # LA is local attention
        self.MRA = MRA(self.dim_split, att_kernel, norm_layer)  # MRA is medium-range attention
        if stage == 2:
            self.GA3 = D_GA(self.dim_split, norm_layer)  # GA3 is global attention (stage of 3)
        elif stage == 3:
            self.GA4 = GA(self.dim_split)  # GA4 is global attention (stage of 4)
            self.norm = norm_layer(self.dim_split)
        else:
            self.GA12 = GA12(self.dim_split, act_layer)  # GA12 is global attention (stages of 1 and 2)
            self.norm = norm_layer(self.dim_split)
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        shortcut = x.clone()
        x1, x2, x3, x4 = torch.split(x, [self.dim_split, self.dim_split, self.dim_split, self.dim_split], dim=1)
        x1 = x1 + self.PA(x1)
        x2 = self.LA(x2)
        x3 = self.MRA(x3)
        if self.stage == 2:
            x4 = x4 + self.GA3(x4)
        elif self.stage == 3:
            x4 = self.norm(x4 + self.GA4(x4))
        else:
            x4 = self.norm(x4 + self.GA12(x4))
        x_att = torch.cat((x1, x2, x3, x4), 1)

        x = shortcut + self.norm1(self.drop_path(self.mlp(x_att)))
        return x


class BasicStage(nn.Module):
    def __init__(self,
                 dim,
                 stage,
                 depth,
                 att_kernel,
                 mlp_ratio,
                 drop_path,
                 norm_layer,
                 act_layer
                 ):
        super().__init__()

        blocks_list = [
            LWGA_Block(
                dim=dim,
                stage=stage,
                att_kernel=att_kernel,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


if __name__ == '__main__':
    depths = (2, 4, 2)
    dpr = [x.item()
           for x in torch.linspace(0, 0.1, sum(depths))]

    depths = (2, 4, 2)
    dpr = [x.item()
           for x in torch.linspace(0, 0.1, sum(depths))]

    m1 = BasicStage(
        dim=int(1024),
        stage=0,
        depth=2,
        att_kernel=11,
        mlp_ratio=2.,
        drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.GELU,
    )
    m2 = BasicStage(
        dim=int(512),
        stage=1,
        depth=4,
        att_kernel=11,
        mlp_ratio=2.,
        drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.GELU,
    )
    m3 = BasicStage(
        dim=int(256),
        stage=2,
        depth=2,
        att_kernel=11,
        mlp_ratio=2.,
        drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
        norm_layer=nn.BatchNorm2d,
        act_layer=nn.GELU,
    )
    d1 = torch.randn(1, 1024, 14, 14)
    # d2 = torch.randn(1, 512, 28, 28)
    # d3 = torch.randn(1, 256, 56, 56)
    print(m1(d1).shape)
    # print(m2(d2).shape)
    # print(m3(d3).shape)
