import torch
from losses import lpips
import PIL
import os
from losses.style.style_loss import StyleLoss

class EmbeddingLossBuilder(torch.nn.Module):
    def __init__(self, opt):
        super(EmbeddingLossBuilder, self).__init__()

        self.opt = opt
        self.parsed_loss = [[opt.l2_lambda, 'l2'], [opt.percept_lambda, 'percep'], [opt.sp_hair_lambda, 'sp_hair']]
        self.l2 = torch.nn.MSELoss()
        if opt.device == 'cuda':
            use_gpu = True
        else:
            use_gpu = False
        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=use_gpu)
        self.percept.eval()
        # self.percept = VGGLoss()

        # style loss
        self.style = StyleLoss(distance="l2", VGG16_ACTIVATIONS_LIST=[3, 8, 15, 22], normalize=False).to(opt.device)
        self.style.eval()


    def _loss_l2(self, gen_im, ref_im, **kwargs):
        return self.l2(gen_im, ref_im)


    def _loss_lpips(self, gen_im, ref_im, **kwargs):

        return self.percept(gen_im, ref_im).sum()


    def _loss_sp_hair(self, gen_im, ref_im, sp_mask):
        return self.style(gen_im * sp_mask, ref_im * sp_mask, mask1=sp_mask, mask2=sp_mask)



    def forward(self, ref_im_H,ref_im_L, gen_im_H, gen_im_L, sp_mask=None):

        loss = 0
        loss_fun_dict = {
            'l2': self._loss_l2,
            'percep': self._loss_lpips,
            'sp_hair': self._loss_sp_hair,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            if loss_type == 'l2':
                var_dict = {
                    'gen_im': gen_im_H,
                    'ref_im': ref_im_H,
                }
            elif loss_type == 'percep':
                var_dict = {
                    'gen_im': gen_im_L,
                    'ref_im': ref_im_L,
                }
            elif loss_type == 'sp_hair':
                if weight == 0 or sp_mask is None:
                    continue
                var_dict = {
                    'gen_im': gen_im_L,
                    'ref_im': ref_im_L,
                    'sp_mask': sp_mask,
                }
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += weight*tmp_loss
        return loss, losses