import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from estimate_net import Estimate_Net
from data import CustomDataset
from utils import ssim, seed_torch, gen_false_color_img
import torch.optim as optim
import pdb 
seed_torch()

class Estimate():
    def __init__(self, data_path='', max_iter=20, print_per_iter=100, pos_dim=32, m_ksize=25):
        super().__init__()
        print('Estimate Spectral-Degradation and Spatial-Degradation ...')
        # data init
        self.dataset = CustomDataset(data_path=data_path)
        # net init
        C, h, w = self.dataset.lrhsi.shape[1:]
        c, H, W = self.dataset.hrmsi.shape[1:]
        lrhsi_size = (C, h, w)
        hrmsi_size = (c, H, W)
        self.est_model = Estimate_Net(lrhsi_size=lrhsi_size, hrmsi_size=hrmsi_size, \
            pos_dim=pos_dim, m_ksize=m_ksize).cuda()
        # mkidr
        self.save_dir = 'results'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # other init
        lr = 5e-4 # 1e-3
        self.max_iter = max_iter
        self.print_per_iter = print_per_iter
        self.criterion = ssim
        self.optimizer = optim.Adam(self.est_model.parameters(), lr=lr) # 去掉正则化项 效果变好

    def run(self):
        print('-------------------')
        print('Start Estimate')
        print('-------------------')
        for epoch in range(self.max_iter):
            self.optimizer.zero_grad()
            hrmsi_spad, lrhsi_sped = self.est_model(self.dataset.hrmsi, self.dataset.lrhsi)
            loss = 1. - self.criterion(hrmsi_spad, lrhsi_sped)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % self.print_per_iter == 0:
                info1 = 'epoch:[%d/%d], loss:%.4f' % (epoch + 1, self.max_iter, loss)
                print(info1)
                info2 = 'SSIM:%.4f' % ssim(hrmsi_spad, lrhsi_sped)
                print(info2)
                hrmsi_spad = hrmsi_spad.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                lrhsi_sped = lrhsi_sped.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                gen_false_color_img(hrmsi_spad, self.save_dir+'/hrmsi_spad_%d.pdf' % (epoch+1), clist=[0,1,2])
                gen_false_color_img(lrhsi_sped, self.save_dir+'/lrhsi_sped_%d.pdf' % (epoch+1), clist=[0,1,2])
               
            torch.save(self.est_model.state_dict(), self.save_dir+'/est_model_last.pkl')

if __name__ == '__main__': 
    data_path = 'hypsen.mat'
    estimate = Estimate(data_path=data_path, max_iter=2000, print_per_iter=100, \
        pos_dim=32, m_ksize=25)
    estimate.run()
