import torch
import torch.nn as nn


class ConvBNGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class MultiKernelInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 4

        self.branch1 = ConvBNGELU(in_channels, branch_channels, 1)
        self.branch3 = nn.Sequential(
            ConvBNGELU(in_channels, branch_channels, 1),
            ConvBNGELU(branch_channels, branch_channels, 3),
        )
        self.branch5 = nn.Sequential(
            ConvBNGELU(in_channels, branch_channels, 1),
            ConvBNGELU(branch_channels, branch_channels, 5),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNGELU(in_channels, branch_channels, 1),
        )
        self.fusion = ConvBNGELU(out_channels, out_channels, 1)
        self.use_residual = in_channels == out_channels

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch3(x)
        b3 = self.branch5(x)
        b4 = self.branch_pool(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fusion(out)
        if self.use_residual:
            out = out + x
        return out


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        drop_path=0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        shortcut = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=False,
        )
        x = shortcut + self.drop_path1(attn_out)

        shortcut = x
        x = shortcut + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class IEViT(nn.Module):
    def __init__(
        self,
        num_classes,
        img_size=224,
        embed_dim=384,
        depth=8,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNGELU(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(
            MultiKernelInceptionBlock(64, 128),
            MultiKernelInceptionBlock(128, 128),
        )
        self.down1 = ConvBNGELU(128, 128, kernel_size=3, stride=2)
        self.stage2 = nn.Sequential(
            MultiKernelInceptionBlock(128, 256),
            MultiKernelInceptionBlock(256, 256),
        )
        self.down2 = ConvBNGELU(256, 256, kernel_size=3, stride=2)
        self.stage3 = nn.Sequential(
            MultiKernelInceptionBlock(256, embed_dim),
            MultiKernelInceptionBlock(embed_dim, embed_dim),
        )

        # The current checkpoint architecture assumes a 224x224 input,
        # yielding a 14x14 token map before the transformer.
        self.num_patches = 14 * 14
        self.patch_embed = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        batch_size = x.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)
