import torch
import torch.nn as nn
from segmentation_models_pytorch.base import modules as md
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from numpy.fft import fft2
import torch.fft
import einops
from einops import rearrange

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.025):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, 
                 attn_drop=0., proj_drop=0., pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

       
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :,
                                  0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :,
                                  1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1) 
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :] 
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1) 
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(
                self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (F.normalize(q, dim=-1) @
                F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(
            torch.tensor(1. / 0.01)).to(self.logit_scale.get_device())).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(
            self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, 
                 shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

           
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 
                                                           float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x


        x_windows = window_partition(shifted_x, self.window_size)
        
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

      
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
     
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, "input feature has wrong size"

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  
        x1 = x[:, 1::2, 0::2, :]  
        x2 = x[:, 0::2, 1::2, :] 
        x3 = x[:, 1::2, 1::2, :]  
        x = torch.cat([x0, x1, x2, x3], -1)  
        x = x.view(B, -1, 4 * C)  

        x = self.reduction(x)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0.025, attn_drop=0.025,
                 drop_path=0.1, norm_layer=nn.LayerNorm, downsample=None, 
                 use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (
                                     i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, 
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        #print("PARTITION:", x.size())
        if self.norm is not None:
            x = self.norm(x)
        return x


class SwinTransformerV2(nn.Module):

    def __init__(self, img_size=384, patch_size=4, in_chans=3, 
                 num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., 
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, 
                 patch_norm=True, use_checkpoint=False, 
                 pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm 
                                      else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(
                                   depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (
                                   i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(
            self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def extra_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        feature = []

        for layer in self.layers:
            x = layer(x)
            bs, n, f = x.shape
            h = int(n**0.5)

            feature.append(
                x.view(-1, h, h, f).permute(0, 3, 1, 2).contiguous())
        return feature

    def get_unet_feature(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        bs, n, f = x.shape
        h = int(n**0.5)
        feature = [x.view(-1, h, h, f).permute(0, 3, 1, 2).contiguous()]

        for layer in self.layers:
            x = layer(x)
            bs, n, f = x.shape
            h = int(n**0.5)

            feature.append(
                x.view(-1, h, h, f).permute(0, 3, 1, 2).contiguous())
        return feature

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


model_urls = {"swinv2_base_window12to24_192to384_22kft1k":
              "swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth"}


def swin_v2(size, checkpoint_path, img_size=384, in_22k=False, **kwargs):
    if size == "swinv2_base_window12to24_192to384_22kft1k":
        model = SwinTransformerV2(img_size=img_size, 
                                  window_size=24, 
                                  embed_dim=128, 
                                  depths=[2, 2, 18, 2], 
                                  num_heads=[4, 8, 16, 32], 
                                  drop_rate=0.025, 
                                  attn_drop_rate=0.025, 
                                  drop_path_rate=0.1, **kwargs).cuda()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f"Loaded model from {checkpoint_path}")

    return model


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type="scse",
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.attention2 = md.Attention(
            attention_type, in_channels=out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels

    def forward(self, x, skip=None):
        if skip is None:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

            x = self.attention1(x)

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.attention2(x)

        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()

        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        kwargs = dict(use_batchnorm=use_batchnorm,
                      attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, 
                                              skip_channels, 
                                              out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):


        features = features[::-1]
        head = features[0]
        skips = features[1:]

        x = nn.Identity()(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x

class BilinearLearnedUpsampling(nn.Module):
    def __init__(self, scale_factor, in_channels):
        super(BilinearLearnedUpsampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = nn.functional.upsample_bilinear(x, scale_factor=self.scale_factor)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels,
                           kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling_layer = BilinearLearnedUpsampling(
            scale_factor=upsampling, in_channels=out_channels) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer)


class unet_swin(nn.Module):
    def __init__(
        self, size="base", img_size=384, checkpoint_path=None
    ):
        super().__init__()

        self.encoder = swin_v2(size=size, img_size=img_size, checkpoint_path=checkpoint_path)
        if size.split("_")[1] in ["small", "tiny"]:
            feature_channels = (3, 192, 384, 768, 768)
        elif size.split("_")[1] in ["base"]:
            feature_channels = (128, 256, 512, 1024, 1024)
        self.decoder = UnetDecoder(encoder_channels=feature_channels, 
                                   n_blocks=4, 
                                   decoder_channels=(512, 256, 128, 64), 
                                   attention_type=None)

        self.segmentation_head = SegmentationHead(in_channels=64, 
                                                  out_channels=64, 
                                                  kernel_size=3, 
                                                  upsampling=4
                                                  )

    def forward(self, input):
        encoder_featrue = self.encoder.get_unet_feature(input)
        decoder_output = self.decoder(*encoder_featrue)
        masks = self.segmentation_head(decoder_output)

        return masks
    
class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Mish(), 
            nn.GroupNorm(32, out_channels)
        )

        self.out_layers = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Mish(),
            nn.GroupNorm(32, out_channels)
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class SpatialUNet(nn.Module):

    def __init__(
            self,
            in_channels=3,
            min_level_channels=64,
            min_channel_mults=[1, 1, 2, 2, 4, 4],
            n_levels_down=6,
            n_levels_up=6,
            n_RBs=2,
    ):
        super().__init__()

        self.enc_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1)]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(
                        RB(ch, min_channel_mult * min_level_channels))
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2))
                )
                enc_block_chans.append(ch)

        
        self.middle_block = nn.Sequential(RB(ch, ch), RB(ch, ch))

        
        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                    )
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),  
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))

        self.output_channels = ch

    def forward(self, x):
        hs = []
        h = x
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)

        h = self.middle_block(h)

        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)

        return h


class RFFTBlock(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels

        
        self.compress = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.Mish(),
            nn.GroupNorm(16, channels // reduction)
        )

       
        self.spectral_process = nn.Sequential(
            nn.Conv2d(channels // reduction * 2, channels // reduction * 2, 1),
            nn.Mish(),
            nn.GroupNorm(16, channels // reduction * 2)
        )

        
        self.expand = nn.Sequential(
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Mish(),
            nn.GroupNorm(16, channels)
        )

        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        identity = x

       
        compressed = self.compress(x)

        
        fft_feat = torch.fft.rfft2(compressed, norm='ortho')
        fft_combined = torch.cat([fft_feat.real, fft_feat.imag], dim=1)

        
        processed = self.spectral_process(fft_combined)

        
        real, imag = torch.chunk(processed, 2, dim=1)
        restored = torch.fft.irfft2(torch.complex(real, imag),
                                    s=compressed.shape[-2:], norm='ortho')

        
        expanded = self.expand(restored)

        
        return expanded + self.gamma * identity


class HybridConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.Mish(),
            nn.GroupNorm(16, out_channels)
        )

        
        self.fft_block = RFFTBlock(out_channels) if out_channels >= 128 else None

    def forward(self, x):
        x = self.spatial_conv(x)
        if self.fft_block is not None:
            x = self.fft_block(x)
        return x


class DownBlock(nn.Module):
    

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = HybridConvBlock(in_channels, out_channels)
        self.down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
            nn.Mish(),
            nn.GroupNorm(16, out_channels)
        )

    def forward(self, x):
        skip = self.conv(x)
        down = self.down(skip)
        return down, skip


class UpBlock(nn.Module):
    

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.Mish(),
            nn.GroupNorm(16, in_channels)
        )
        self.conv = HybridConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FrequencyUNet(nn.Module):

    def __init__(self, in_channels=3, base_channels=64, depth=4):
        super().__init__()

       
        self.output_channels = base_channels

        
        self.init_conv = HybridConvBlock(in_channels, base_channels)

        
        self.down_path = nn.ModuleList()
        self.up_path = nn.ModuleList()

        channels = base_channels
        skip_channels = []

        
        for i in range(depth):
            self.down_path.append(DownBlock(channels, channels * 2))
            skip_channels.append(channels * 2)
            channels *= 2

        
        self.bottleneck = nn.Sequential(
            HybridConvBlock(channels, channels),
            HybridConvBlock(channels, channels)
        )

        
        for i in range(depth):
            self.up_path.append(UpBlock(channels, skip_channels.pop(), channels // 2))
            channels //= 2

       
        self.final_conv = nn.Conv2d(channels, base_channels, 1)

    def forward(self, x):
        skips = []

        
        x = self.init_conv(x)
        for down in self.down_path:
            x, skip = down(x)
            skips.append(skip)

       
        x = self.bottleneck(x)

        
        for up in self.up_path:
            skip = skips.pop()
            x = up(x, skip)

        
        return self.final_conv(x)


class SFAA(nn.Module):
    
    def __init__(self, channel, reduction=16, use_bn=True, use_gn=True):
        super(SFAA, self).__init__()
        self.use_bn = use_bn
        self.use_gn = use_gn
        
       
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        
        input_channels = channel * 2
        
      
        layers = [
            nn.Conv2d(input_channels, input_channels // reduction, 1, bias=False),
        ]
        
        
        if use_bn:
            layers.append(nn.BatchNorm2d(input_channels // reduction))
            
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(input_channels // reduction, input_channels, 1, bias=False))
        
        
        if use_gn:
            layers.append(nn.GroupNorm(1, input_channels)) 
            
        self.mlp = nn.Sequential(*layers)
        
        
        self.spatial_weight = nn.Parameter(torch.ones(1))
        self.freq_weight = nn.Parameter(torch.ones(1))
        
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, spatial_feat, freq_feat):
       
        if spatial_feat.size() != freq_feat.size():
            freq_feat = nn.functional.interpolate(
                freq_feat, size=spatial_feat.size()[2:], 
                mode='bilinear', align_corners=True
            )
            
        
        combined_feat = torch.cat([spatial_feat, freq_feat], dim=1)
        
       
        avg_out = self.mlp(self.avg_pool(combined_feat))
        max_out = self.mlp(self.max_pool(combined_feat))
        attn = avg_out + max_out
        attn = nn.functional.sigmoid(attn)
        
       
        b, c, h, w = combined_feat.size()
        spatial_attn, freq_attn = torch.split(attn, c//2, dim=1)
        
        
        spatial_out = spatial_feat * spatial_attn
        freq_out = freq_feat * freq_attn
        
       
        fused_feat = (self.spatial_weight * spatial_out + 
                      self.freq_weight * freq_out) / (self.spatial_weight + self.freq_weight)
        
        return fused_feat


class SFUNet(nn.Module):

    def __init__(
            self,
            in_channels=3,
            min_level_channels=64,
            min_channel_mults=[1, 1, 2, 2, 4, 4],
            n_levels_down=6,
            n_levels_up=6,
            n_blocks=2,
    ):
        super().__init__()

       
        self.spatial_branch = SpatialUNet(
            in_channels=in_channels,
            min_level_channels=min_level_channels,
            min_channel_mults=min_channel_mults,
            n_levels_down=n_levels_down,
            n_levels_up=n_levels_up,
            n_RBs=n_blocks
        )

       
        self.frequency_branch = FrequencyUNet(
            in_channels=in_channels,
            base_channels=min_level_channels 
        )

        
        self.fusion = SFAA(min_level_channels)

        self.output_channels = min_level_channels

    def forward(self, x):
     
        spatial_feat = self.spatial_branch(x)
        freq_feat = self.frequency_branch(x)

        
        fused_feat = self.fusion(spatial_feat, freq_feat)
        return fused_feat


class SwinV2_SF(nn.Module):
    def __init__(self, size=384, checkpoint_path=None):
        super().__init__()

        self.TB = unet_swin(
            img_size=384, size="swinv2_base_window12to24_192to384_22kft1k", checkpoint_path=checkpoint_path)
        self.SFUNet = SFUNet()
        self.PH = nn.Sequential(
            RB(128, 64), RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )
        self.up_tosize = nn.Upsample(size=size)

    def forward(self, x):
        x1 = self.TB(x)
        x2 = self.SFUNet(x)
        x = torch.cat((x1, x2), dim=1)
        out = self.PH(x)

        return out


if __name__ == "__main__":
    image = torch.rand((1, 3, 384, 384)).cuda()
    model = SwinV2_SF().cuda()
    print(model(image).size())
    del model
    del image
