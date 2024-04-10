import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import torchvision.models as models
import clip

# pytorch pretrained vgg
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        #pretrained vgg19
        vgg19 = models.vgg19(weights='DEFAULT').features

        self.relu1_1 = vgg19[:2]
        self.relu2_1 = vgg19[2:7]
        self.relu3_1 = vgg19[7:12]
        self.relu4_1 = vgg19[12:21]

        #fix parameters
        self.requires_grad_(False)

    def forward(self, x):
        _output = namedtuple('output', ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
        relu1_1 = self.relu1_1(x)
        relu2_1 = self.relu2_1(relu1_1)
        relu3_1 = self.relu3_1(relu2_1)
        relu4_1 = self.relu4_1(relu3_1)
        output = _output(relu1_1, relu2_1, relu3_1, relu4_1)

        # torch.Size([1, 64, 256, 256])
        # torch.Size([1, 128, 128, 128])
        # torch.Size([1, 256, 64, 64])
        # torch.Size([1, 512, 32, 32])

        return output


class Decoder(nn.Module): 
    """
    starting from relu 4_1
    """
    def __init__(self, ckpt_path=None):
        super().__init__()
        
        self.layers = nn.Sequential(
            # nn.Conv2d(512, 256, 3, padding=1, padding_mode='reflect'),
            # nn.ReLU(),
            # nn.Upsample(scale_factor=2, mode='nearest'), # relu4-1
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu3-4
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu3-3
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu3-2
            nn.Conv2d(256, 128, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='nearest'),# relu3-1
            nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu2-2
            nn.Conv2d(128, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='nearest'),# relu2-1
            nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu1-2
            nn.Conv2d(64, 3, 3, padding=1, padding_mode='reflect'),
        )

        if ckpt_path is not None:
          self.load_state_dict(torch.load(ckpt_path))

    def forward(self, x):
        return self.layers(x)


### high-res unet feature map decoder


class DownBlock(nn.Module):

    def __init__(self, in_dim, out_dim, down='conv'):
        super(DownBlock, self).__init__()

        if down == 'conv':
            self.down_conv = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 3, 2, 1),
                nn.LeakyReLU(),
                nn.Conv2d(out_dim, out_dim, 3, 1, 1),
                nn.LeakyReLU(),
            )
        elif down == 'mean':
            self.down_conv = nn.AvgPool2d(2)
        else:
            raise NotImplementedError(
                '[ERROR] invalid downsampling operator: {:s}'.format(down)
            )

    def forward(self, x):
        x = self.down_conv(x)
        return x


class UpBlock(nn.Module):

    def __init__(self, in_dim, out_dim, skip_dim=None, up='nearest'):
        super(UpBlock, self).__init__()

        if up == 'conv':
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 3, 2, 1, 1),
                nn.ReLU(),
            )
        else:
            assert up in ('bilinear', 'nearest'), \
                '[ERROR] invalid upsampling mode: {:s}'.format(up)
            self.up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=up),
                nn.Conv2d(in_dim, out_dim, 3, 1, 1),
                nn.ReLU(),
            )
        
        in_dim = out_dim
        if skip_dim is not None:
            in_dim += skip_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.ReLU(),
        )

    def _pad(self, x, y):
        dh = y.size(-2) - x.size(-2)
        dw = y.size(-1) - x.size(-1)
        if dh == 0 and dw == 0:
            return x
        if dh < 0:
            x = x[..., :dh, :]
        if dw < 0:
            x = x[..., :, :dw]
        if dh > 0 or dw > 0:
            x = F.pad(
                x, 
                pad=(dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), 
                mode='reflect'
            )
        return x

    def forward(self, x, skip=None):
        x = self.up_conv(x)
        if skip is not None:
            x = torch.cat([self._pad(x, skip), skip], 1)
        x = self.conv(x)
        return x


class UNetDecoder(nn.Module):

    def __init__(self, in_dim=256):
        super(UNetDecoder, self).__init__()

        self.down_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        in_dim = in_dim
        self.n_levels = 2
        self.up = 1

        for i in range(self.n_levels):
            self.down_layers.append(
                DownBlock(
                    in_dim, in_dim,
                )
            )
            out_dim = in_dim // 2 ** (self.n_levels - i)
            self.skip_convs.append(nn.Conv2d(in_dim, out_dim, 1))
            self.up_layers.append(
                UpBlock(
                    out_dim * 2, out_dim, out_dim,
                )
            )

        out_dim = in_dim // 2 ** self.n_levels
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_dim, 3, 1, 1),
        )

    def forward(self, feats):
        skips = []
        for i in range(self.n_levels):
            skips.append(self.skip_convs[i](feats))
            feats = self.down_layers[i](feats)
        for i in range(self.n_levels - 1, -1, -1):
            feats = self.up_layers[i](feats, skips[i])
        rgb = self.out_conv(feats)
        return rgb


### high-res feature map decoder

class PlainDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu3-4
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu3-3
            nn.Conv2d(256, 256, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu3-2
            nn.Conv2d(256, 128, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), 
            nn.Conv2d(128, 128, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu2-2
            nn.Conv2d(128, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1, padding_mode='reflect'),
            nn.ReLU(), # relu1-2
            nn.Conv2d(64, 3, 3, padding=1, padding_mode='reflect'),
        )

    def forward(self, x):
        return self.layers(x)

class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
        return outputs


class CLIP(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.clip_model, self.preprocess = clip.load('ViT-B/32', device, jit=False)


def sliding_window(image, window_size, step=16):
    _, _, H, W = image.shape
    for i in range(0, H - window_size + 1, step):
        for j in range(0, W - window_size + 1, step):
            yield image[:, :, i:i+window_size, j:j+window_size], i, j