import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from filters_generator import WA_1Conv, WA_3GConv

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x_s, x_d):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x_s: input tensor with shape of [b h w c];
            x_d: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True;
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x_s = torch.roll(x_s, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x_s = rearrange(x_s, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)

        qkv = rearrange(x_s, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)

        q_s, k_s, v_s = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)

        # ======================================
        if self.type!='W': x_d = torch.roll(x_d, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x_d = rearrange(x_d, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x_d.size(1)
        w_windows = x_d.size(2)

        qkv = rearrange(x_d, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)

        q_d, k_d, v_d = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        #======================================

        sim_s = torch.einsum('hbwpc,hbwqc->hbwpq', q_s, k_s) * self.scale
        # Adding learnable relative embedding
        sim_s = sim_s + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim_s = sim_s.masked_fill_(attn_mask, float("-inf"))

        probs_s = nn.functional.softmax(sim_s, dim=-1)
        output_s = torch.einsum('hbwij,hbwjc->hbwic', probs_s, v_d)
        output_s = rearrange(output_s, 'h b w p c -> b w p (h c)')

        #============================================
        sim_d = torch.einsum('hbwpc,hbwqc->hbwpq', q_d, k_d) * self.scale
        # Adding learnable relative embedding
        sim_d = sim_d + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim_d = sim_d.masked_fill_(attn_mask, float("-inf"))

        probs_d = nn.functional.softmax(sim_d, dim=-1)
        output_d = torch.einsum('hbwij,hbwjc->hbwic', probs_d, v_s)
        output_d = rearrange(output_d, 'h b w p c -> b w p (h c)')

        output = torch.cat((output_s,output_d),-1)

        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]


# Haly Dynamic Multi-head Cross-attention (HDMC)
class HDMC(nn.Module):
    def __init__(self, dim, fgroup, head_dim, window_size, type='W'):
        super(HDMC, self).__init__()
        self.dim = dim
        self.heads = dim // head_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=dim * 3,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels=dim * 3, out_channels=dim * 3,
                      kernel_size=3, padding=1, stride=1, groups=self.heads * 3, bias=False)
        )
        self.dynamic = nn.Sequential(
            WA_1Conv(inp=dim, oup=dim * 3, fgroup=fgroup),
            nn.GELU(),
            WA_3GConv(inp=dim * 3, oup=dim * 3, fgroup=fgroup, cgroup=self.heads * 3)
        )
        self.att = WMSA(input_dim=dim, output_dim=dim * 2, head_dim=head_dim, window_size=window_size, type=type)
        self.conv1_1 = nn.Conv2d(in_channels=dim * 2, out_channels=dim,
                                 kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def forward(self, x):
        x_s = self.conv(x)
        x_d = self.dynamic(x)
        x_s = Rearrange('b c h w -> b h w c')(x_s)
        x_d = Rearrange('b c h w -> b h w c')(x_d)
        x = self.att(x_s, x_d)
        x = Rearrange('b h w c -> b c h w')(x)
        x = self.conv1_1(x)

        return x

#Haly Dynamic Feed-Forward Network (HDFN)
class HDFN(nn.Module):
    def __init__(self, dim, fgroup):
        super(HDFN, self).__init__()

        self.dim = dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim*2,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels=dim*2, out_channels=dim*2,
                      kernel_size=3, padding=1, stride=1, groups=dim, bias=False)
        )
        self.dynamic = nn.Sequential(
            WA_1Conv(inp=dim, oup=dim*2, fgroup=fgroup),
            nn.GELU(),
            WA_3GConv(inp=dim*2, oup=dim*2, fgroup=fgroup, cgroup=dim)
        )
        self.linear = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim // 16,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.Conv2d(in_channels=dim // 16, out_channels=1,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            nn.Sigmoid()
        )

        self.conv1_1 = nn.Conv2d(in_channels=dim*2, out_channels=dim,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=False)

    def forward(self, x):
        conv_x = self.conv(x)
        dynamic_x = self.dynamic(x)
        alpha = self.linear(F.adaptive_avg_pool2d(x, (1, 1)))
        x = conv_x * alpha + dynamic_x * (1-alpha)
        x = self.conv1_1(x)

        return x

# Weather Adaptive Transformer Block (WATB)
class WATB(nn.Module):
    def __init__(self, dim, fgroup, head_dim, window_size, type='W'):
        super(WATB, self).__init__()

        self.ln1 = LayerNorm2d(dim)
        self.hdmc = HDMC(dim=dim, fgroup=fgroup, head_dim=head_dim, window_size=window_size, type=type)
        self.ln2 = LayerNorm2d(dim)
        self.hdfn = HDFN(dim=dim, fgroup=fgroup)

    def forward(self, x):
        x = self.hdmc(self.ln1(x)) + x
        x = self.hdfn(self.ln2(x)) + x

        return x


#Weather Adaptive Transformer (WAformer)
class WAformer(nn.Module):

    def __init__(self, img_channel=3, width=32, fgroup=16, head_dim=32, window_size=8, enc_blk_nums=[4, 4, 6], middle_blk_num=8,
                 dec_blk_nums=[6, 4, 4]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()


        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(
                nn.Sequential(
                    *[WATB(chan, fgroup, head_dim, window_size, 'W' if not j%2 else 'SW') for j in range(num)]
                )
            )
            self.downs.append(
                nn.Sequential(
                nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                nn.Conv2d(chan, 2 * chan, 3, 1, 1)
                )
            )
            chan = chan * 2

        self.middle = \
            nn.Sequential(
                *[WATB(chan, fgroup, head_dim, window_size, 'W' if not j%2 else 'SW') for j in range(middle_blk_num)]
            )


        for i, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(chan, chan // 2, 3, 1, 1)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[WATB(chan, fgroup, head_dim, window_size, 'W' if not j%2 else 'SW') for j in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders) * window_size
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0.0, 0.02)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = inp -x

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


if __name__ == '__main__':
    net = WAformer().cuda()


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    # params = float(params[:-3])
    # macs = float(macs[:-4])

    print(macs, params)


