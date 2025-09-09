import torch.nn as nn
import torch
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, residual: bool = False):
        super().__init__()
        assert out_channels is not None, "DoubleConv requires out_channels"
        self.residual = residual
        self.mid_channels = mid_channels if mid_channels is not None else out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, self.mid_channels),
            nn.GELU(),
            nn.Conv2d(self.mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16, out_channels),
        )

    def forward(self, x):
        if self.residual:
            # residual add only valid if shapes match
            return F.gelu(self.double_conv(x) + x)
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, embed_dim=256):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            # in_channels -> out_channels (no residual, shapes differ)
            DoubleConv(in_channels, out_channels, out_channels, residual=False),
            # out_channels -> out_channels (residual ok)
            DoubleConv(out_channels, out_channels, out_channels, residual=True),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels)
        )

    def forward(self, x, emb):
        x = self.down(x)
        emb = self.emb_layer(emb)
        x = x + emb[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        # in_channels here is the concatenated channels (skip + upsampled)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            # keep channels then residual (in -> in)
            DoubleConv(in_channels, in_channels, in_channels, residual=True),
            # reduce to out_channels
            DoubleConv(in_channels, in_channels // 2, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=256, image_size=128,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        # Encoder
        self.inc = DoubleConv(in_channels, 64, 64)
        self.down1 = Down(64, 128)
           # 128x128 -> 64x64 after down1
        self.down2 = Down(128, 256)
          # 64x64 -> 32x32
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 16)   # 32x32 -> 16x16

        # Bottleneck
        self.bot1 = DoubleConv(256, 512, 512)
        self.bot2 = DoubleConv(512, 512, 512)
        self.bot3 = DoubleConv(512, 256, 256)

        # Decoder
        self.up1 = Up(512, 128)             # cat(256, 256) = 512 -> 128
         # 16x16 up to 32x32
        self.up2 = Up(256, 64)              # cat(128, 128) = 256 -> 64
           # 32x32 up to 64x64
        self.up3 = Up(128, 64)              # cat(64, 64) = 128 -> 64
           # 64x64 up to 128x128

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)            # 64, 128x128
        x2 = self.down1(x1, t)      # 128, 64x64
        
        x3 = self.down2(x2, t)      # 256, 32x32
        x4 = self.down3(x3, t)      # 256, 16x16
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)          # 256, 16x16

        x = self.up1(x4, x3, t)     # 128, 32x32
        x = self.up2(x, x2, t)      # 64, 64x64
        x = self.up3(x, x1, t)      # 64, 128x128
        output = self.outc(x)
        return output


"""

Input: x → (B, 3, 128, 128)
inc (DoubleConv 3→64): (B, 64, 128, 128) = x1
down1:
MaxPool2d(2): (B, 64, 64, 64)
DoubleConv 64→128: (B, 128, 64, 64)
DoubleConv 128→128: (B, 128, 64, 64) = x2
sa1 (SelfAttention 128 ch, size=64): (B, 128, 64, 64)
down2:
MaxPool2d(2): (B, 128, 32, 32)
DoubleConv 128→256: (B, 256, 32, 32)
DoubleConv 256→256: (B, 256, 32, 32) = x3
sa2 (SelfAttention 256 ch, size=32): (B, 256, 32, 32)
down3:
MaxPool2d(2): (B, 256, 16, 16)
DoubleConv 256→256: (B, 256, 16, 16)
DoubleConv 256→256: (B, 256, 16, 16) = x4
sa3 (SelfAttention 256 ch, size=16): (B, 256, 16, 16)
Bottleneck

bot1 (DoubleConv 256→512): (B, 512, 16, 16)
bot2 (DoubleConv 512→512): (B, 512, 16, 16)
bot3 (DoubleConv 512→256): (B, 256, 16, 16)
Decoder

up1:
Upsample x4: (B, 256, 32, 32)
Concatenate with skip x3: cat([256, 256], dim=1) → (B, 512, 32, 32)
DoubleConv 512→512 (residual), then 512→128: (B, 128, 32, 32)
sa4 (SelfAttention 128 ch, size=32): (B, 128, 32, 32)
up2:
Upsample: (B, 128, 64, 64)
Concatenate with skip x2: cat([128, 128]) → (B, 256, 64, 64)
DoubleConv 256→256 (residual), then 256→64: (B, 64, 64, 64)
sa5 (SelfAttention 64 ch, size=64): (B, 64, 64, 64)
up3:
Upsample: (B, 64, 128, 128)
Concatenate with skip x1: cat([64, 64]) → (B, 128, 128, 128)
DoubleConv 128→128 (residual), then 128→64: (B, 64, 128, 128)
sa6 (SelfAttention 64 ch, size=128): (B, 64, 128, 128)
Head

outc (Conv2d 64→out_channels): (B, out_channels, 128, 128)
Notes

Time embedding t is shaped to (B, time_dim) and is added at each Down/Up stage as a bias per-channel broadcast; it doesn’t change the spatial sizes.
The SelfAttention modules assume square feature maps with the provided size; for 128×128 input the sizes above are correct.
If your input size differs from 128×128, the spatial sizes scale accordingly: each Down halves H,W; each Up doubles H,W. Make sure the SelfAttention size arguments match those feature map sizes at each level.

"""        