import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb 
from gaussian_kernel import Gaussian_Kernel
from utils import xavier_init, he_init
from unet import UNet

class Spectral_Modulation(nn.Module):
    def __init__(self, C, pos_dim=16):        
        super().__init__()
        self.pos_emb = nn.Sequential(
                nn.Conv2d(2, pos_dim, 1, 1, 0, bias=False),
                nn.Tanh(),
                )
        # self.pos_emb.apply(xavier_init)

        self.modulate = nn.Sequential(
                nn.Conv2d(C + pos_dim, 64, 1, 1, 0, bias=False),
                nn.GELU(),
                nn.Conv2d(64, C, 1, 1, 0, bias=False),
                nn.GELU(),
                )
        # self.modulate.apply(he_init)
    
    def get_pos_matrix(self, shape):
        h, w = shape
        y_values = torch.linspace(-1, 1, h)
        x_values = torch.linspace(-1, 1, w)
        y_matrix, x_matrix = torch.meshgrid(y_values, x_values)
        pos_matrix = torch.stack((y_matrix, x_matrix), dim=-1)
        pos_matrix = pos_matrix.unsqueeze(0).permute(0, 3, 1, 2).cuda()

        return pos_matrix

    def forward(self, x):
        b, _, h, w = x.shape
        pos_matrix = self.get_pos_matrix((h, w))
        pos_matrix = pos_matrix.repeat(b, 1, 1, 1)
        x_emb = self.pos_emb(pos_matrix)
        x_cat = torch.cat((x, x_emb), dim=1)
        x_modulate = self.modulate(x_cat)
        out = torch.clamp(x_modulate, 0.0, 1.0)
          
        return out

class Spectral_Degradation(nn.Module):
    def __init__(self, size_info, pos_dim=32):
        super().__init__()
        C, c = size_info
        mid_ch1 = 64
        pos_dim = pos_dim
        self.convs = nn.ModuleList()
        for i in range(c):
            tmp = nn.Sequential(
                nn.Conv2d(C, mid_ch1, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(mid_ch1),
                nn.GELU(),
                nn.Conv2d(mid_ch1, mid_ch1//2, 1, 1, 0, bias=False),
                nn.GELU(),
                nn.Conv2d(mid_ch1//2, 1, 1, 1, 0, bias=False),
                nn.GELU(),
                )
            # tmp.apply(he_init)
            self.convs.append(tmp)     
        self.modulate = Spectral_Modulation(C, pos_dim)
        self.new_bands = c
       
    def forward(self, x):
        n, _, h, w = x.shape
        x = self.modulate(x)
        x_speD = torch.zeros((n, self.new_bands, h, w)).cuda()
        for i in range(self.new_bands):
            mid = self.convs[i](x)
            x_speD[:, i:i+1, ...] = mid[:]
        x_speD = torch.clamp(x_speD, 0.0, 1.0)

        return x_speD

class Spatial_Warping(nn.Module):
    def __init__(self, size_info):        
        super().__init__()
        h, w, scale = size_info
        vectors = [torch.arange(0, s) for s in (h, w)]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        self.grid = torch.unsqueeze(grid, 0).cuda()  # add batch
        self.field_generator = UNet(dim=16, C_in=1, C_out=2)
        # self.field_generator.apply(he_init)
        self.shape = (h, w)
    
    def forward(self, x, use_type=''):
        b = x.shape[0]
        if use_type == 'train':
            field = self.field_generator(x.mean(1).unsqueeze(1))
            field[field > 3] = 0 
            field[field < -3] = 0 
            grid = self.grid.repeat(b, 1, 1, 1)
            field = field + grid
            # Need to normalize field values to [-1, 1]
            for i in range(2):
                field[:, i, ...] = 2 * (field[:, i, ...] / (self.shape[i] - 1) - 0.5)
            # field: (batch_size, height, width, 2) 2->(x, y)
            field = field.permute(0, 2, 3, 1)
            field = field[..., [1, 0]]
            torch.save(field.detach(), 'field.pt')
        elif use_type == 'test':
            field = torch.load('field.pt').cuda()
        x_warping = F.grid_sample(x, field, mode='bilinear', padding_mode='border', align_corners=False)
        out = torch.clamp(x_warping, 0.0, 1.0)
          
        return out

class Spatial_Degradation(nn.Module):
    def __init__(self, size_info, m_ksize=25):
        super().__init__()
        # size_info: h,w,scale
        self.scale = size_info[-1]
        self.pad = int((m_ksize - 1) / 2)
        self.warping = Spatial_Warping(size_info)
        # ['sigma_y', 'sigma_x', 'theta', 'offset_y', 'offset_x']
        self.kernel_generator = Gaussian_Kernel(m_ksize=m_ksize, \
            init_paras=[3.0, 3.0, 0., 0., 0.])

    def forward(self, x, use_type='train'):
        bands = x.shape[1]
        x = self.warping(x, use_type)
        self.kernel = self.kernel_generator()
        kernel = self.kernel.repeat(bands, 1, 1, 1)
        x_spaD = F.conv2d(x, kernel, stride=1, padding=self.pad, bias=None, groups=bands)
        x_spaD = x_spaD[..., ::self.scale, ::self.scale]
        x_spaD = torch.clamp(x_spaD, 0.0, 1.0)

        return x_spaD

class Estimate_Net(nn.Module):
    def __init__(self, lrhsi_size=(), hrmsi_size=(), pos_dim=32, m_ksize=25):
        super().__init__()
        C, h, w = lrhsi_size
        c, H, W = hrmsi_size
        scale = H // h
        size_info1 = (C, c)
        size_info2 = (H, W, scale)

        self.SpeD = Spectral_Degradation(size_info1, pos_dim=pos_dim)
        self.SpaD = Spatial_Degradation(size_info2, m_ksize=m_ksize)

    def forward(self, hrmsi, lrhsi):
        hrmsi_sapd = self.SpaD(hrmsi)
        lrhsi_sped = self.SpeD(lrhsi)

        return hrmsi_sapd, lrhsi_sped
       
