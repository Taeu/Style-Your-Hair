import torch
from torch import nn
from models.Net import Net
import numpy as np
import os
from functools import partial
from utils.bicubic import BicubicDownSample
from tqdm import tqdm
import PIL
import torchvision
from PIL import Image
from utils.data_utils import convert_npy_code
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from losses.align_loss import AlignLossBuilder
import torch.nn.functional as F
import cv2
from utils.data_utils import load_FS_latent
from utils.seg_utils import save_vis_mask
from utils.model_utils import download_weight
from utils.data_utils import cuda_unsqueeze
from utils.image_utils import dilate_erosion_mask_tensor

import face_alignment


# added for perceptual loss
from losses import lpips
import torchvision.transforms as transforms

# added for blend with align
from losses import masked_lpips
from utils.image_utils import load_image, dilate_erosion_mask_path, dilate_erosion_mask_tensor

from torchvision.utils import save_image
from glob import glob
import matplotlib.pyplot as plt

# added
from utils.slic_utils import slic_custom
import shutil

import matplotlib.pyplot as plt

toPIL = torchvision.transforms.ToPILImage()
toTensor = torchvision.transforms.ToTensor()
resize256 = torchvision.transforms.Resize(256)

class Alignment(nn.Module):

    def __init__(self, opts, net=None):
        super(Alignment, self).__init__()
        self.opts = opts
        if not net:
            self.net = Net(self.opts)
        else:
            self.net = net

        self.load_segmentation_network()
        self.load_downsampling()
        self.setup_align_loss_builder()

        self.seg_mean = seg_mean.to(opts.device)
        self.seg_std = seg_std.to(opts.device)

        if self.opts.kp_loss:
            # self.setup_align_loss_builder(no_face=False)
            if self.opts.kp_type =='2D':
                self.kp_extractor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=opts.device)
            else:
                self.kp_extractor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=opts.device)
            for param in self.kp_extractor.face_alignment_net.parameters():
                param.requires_grad = False
            self.l2 = torch.nn.MSELoss()

        ### perceptual loss
        self.percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=self.opts.device == 'cuda')
        self.percept.eval()
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.image_transform1024 = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.image_transform256 = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # blend with alignment
        if self.opts.blend_with_align:
            self.percept_with_mask = masked_lpips.PerceptualLoss(
                model="net-lin", net="vgg", vgg_blocks=['1', '2', '3'], use_gpu=self.opts.device == 'cuda'
            )
            self.percept_with_mask.eval()

    def load_segmentation_network(self):
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.opts.device)

        if not os.path.exists(self.opts.seg_ckpt):
            download_weight(self.opts.seg_ckpt)
        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt, map_location=self.opts.device))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()

    def load_downsampling(self):
        self.downsample = BicubicDownSample(factor=self.opts.size // 512, cuda=self.opts.device == 'cuda')
        self.downsample_256 = BicubicDownSample(factor=self.opts.size // 256, cuda=self.opts.device == 'cuda')

    def setup_align_loss_builder(self, no_face=False):
        self.loss_builder = AlignLossBuilder(self.opts, no_face = no_face)

    def preprocess_img(self, img_path, is_downsampled = True):
        im = Image.open(img_path)
        return self.preprocess_PILImg(im, is_downsampled = is_downsampled)

    def preprocess_PILImg(self, im, is_downsampled = True):
        im = torchvision.transforms.ToTensor()(im)[:3].unsqueeze(0).to(self.opts.device)
        if is_downsampled:
            im = (self.downsample(im).clamp(0, 1) - self.seg_mean) / self.seg_std
        else:
            im = (im.clamp(0,1) - self.seg_mean) / self.seg_std
        return im

    def get_img_and_seg_from_path(self, img_path, is_downsampled=True):
        im = self.preprocess_img(img_path, is_downsampled=is_downsampled)
        if is_downsampled == False:  # upsample img to 512
            im = F.interpolate(im, size=(512, 512))
        down_seg, _, _ = self.seg(im)
        if is_downsampled == False:  # downsample img to original size
            down_seg = F.interpolate(down_seg, size=(self.opts.size, self.opts.size))
        seg_target = torch.argmax(down_seg, dim=1).long()
        return im, seg_target

    def create_target_segmentation_mask(self, img_path1, img_path2, sign, save_intermediate=True, is_downsampled = True):

        im1, seg_target1 = self.get_img_and_seg_from_path(img_path1, is_downsampled= is_downsampled)

        if self.opts.save_all:
            save_vis_mask(img_path1, img_path2, seg_target1[0].cpu(), self.opts.save_dir, count='0_initial_src_seg')
        ggg = torch.where(seg_target1 == 0, torch.zeros_like(seg_target1), torch.ones_like(seg_target1)) # 0 : background
        hair_mask1 = torch.where(seg_target1 == 10, torch.ones_like(seg_target1), torch.zeros_like(seg_target1)) # 10 : hair
        seg_target1 = seg_target1[0].byte().cpu().detach()
        seg_target1 = torch.where(seg_target1 == 10, torch.zeros_like(seg_target1), seg_target1) # hair 부분 제외한 나머지 segmap

        im2, seg_target2 = self.get_img_and_seg_from_path(img_path2, is_downsampled=is_downsampled)
        original_img_path2 = img_path2

        if self.opts.optimize_warped_trg_mask:
            # 220131 target warping optimization + 220204
            im1_for_kp = F.interpolate(im1, size=(256, 256))
            im1_for_kp = ((im1_for_kp + 1) / 2).clamp(0, 1) # [0, 1] 사이로
            src_kp_hm = self.kp_extractor.face_alignment_net(im1_for_kp)
            im2, warped_latent_2, warped_down_seg = self.warp_target(img_path2, src_kp_hm, None, img_path1) # Warping !!

            warped_down_seg, im2 = self.create_down_seg(warped_latent_2, is_downsampled=is_downsampled)
            if is_downsampled == False:
                warped_seg = F.interpolate(warped_down_seg, size=(self.opts.size, self.opts.size))
                seg_target2 = torch.argmax(warped_seg, dim=1).long() # todo : debug for k,  512 or 256
            else:
                seg_target2 = torch.argmax(warped_down_seg, dim=1).long()
            warped_down_seg = torch.argmax(warped_down_seg.clone().detach(), dim=1).long() # 512, 512

        hair_mask2 = torch.where(seg_target2 == 10, torch.ones_like(seg_target2), torch.zeros_like(seg_target2))
        seg_target2 = seg_target2[0].byte().cpu().detach()
        if self.opts.save_all:
            save_vis_mask(img_path1, img_path2, seg_target1.cpu(), self.opts.save_dir, count='0_erased_src_seg')
        new_target = torch.where(seg_target2 == 10, 10 * torch.ones_like(seg_target1), seg_target1) # put target hair on the target seg 1 (Here, seg_target1 has no hair region)
        if self.opts.save_all:
            save_vis_mask(img_path1, img_path2, new_target.cpu(), self.opts.save_dir, count='0_initial_target_seg')

        if self.opts.mean_seg:
            if self.opts.warped_seg: # mean_seg is the warped target img's seg
                mean_seg = warped_down_seg.cpu().squeeze().type(torch.ByteTensor) # 512, 512 or 256, 256
                if self.opts.save_all:
                    save_vis_mask(img_path1, img_path2, mean_seg.cpu(),self.opts.save_dir,count='1_warped_target_seg')

            new_target_mean_seg = torch.where((new_target == 0) * (mean_seg != 0), mean_seg, new_target) ## 220213 edited by taeu
            if self.opts.save_all:
                save_vis_mask(img_path1, img_path2, new_target_mean_seg.cpu(), self.opts.save_dir,count='1_warped_target+source_seg')
            target_mask = new_target_mean_seg.unsqueeze(0).long().to(self.opts.device)

        #####################  Save Visualization of Target Segmentation Mask
        if self.opts.save_all:
            save_vis_mask(img_path1, img_path2, target_mask.squeeze().cpu(),self.opts.save_dir, count='2_final_target_seg')

        hair_mask_target = torch.where(target_mask == 10, torch.ones_like(target_mask), torch.zeros_like(target_mask))
        if is_downsampled:
            hair_mask_target = F.interpolate(hair_mask_target.float().unsqueeze(0), size=(512, 512), mode='nearest')
        else:
            hair_mask_target = F.interpolate(hair_mask_target.float().unsqueeze(0), size=(self.opts.size, self.opts.size), mode='nearest')

        if self.opts.optimize_warped_trg_mask:
            im2, seg_target2 = self.get_img_and_seg_from_path(original_img_path2, is_downsampled=is_downsampled)
            hair_mask2 = torch.where(seg_target2 == 10, torch.ones_like(seg_target2), torch.zeros_like(seg_target2))
            return target_mask, hair_mask_target, hair_mask1, hair_mask2, warped_latent_2
        else:
            return target_mask, hair_mask_target, hair_mask1, hair_mask2, None

    def setup_align_optimizer(self, latent_path=None):
        if latent_path:
            latent_W = torch.from_numpy(convert_npy_code(np.load(latent_path))).to(self.opts.device).requires_grad_(True)
        else:
            latent_W = self.net.latent_avg.reshape(1, 1, 512).repeat(1, 18, 1).clone().detach().to(self.opts.device).requires_grad_(True)

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }

        optimizer_align = opt_dict[self.opts.opt_name]([latent_W], lr=self.opts.learning_rate)

        return optimizer_align, latent_W

    # 220208 warp target from FS space
    def setup_warp_F_optimizer(self, F, S):
        F = F.to(self.opts.device).requires_grad_(True)
        S = S.to(self.opts.device).requires_grad_(True)

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }

        optimizer_warp = opt_dict[self.opts.opt_name]([F, S], lr=self.opts.learning_rate)
        return optimizer_warp, F, S

    def create_down_seg(self, latent_in, is_downsampled=True):
        gen_im, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
                                       start_layer=0, end_layer=8)
        gen_im_0_1 = (gen_im + 1) / 2
        # get hair mask of synthesized image
        if is_downsampled:
            im = (self.downsample(gen_im_0_1) - self.seg_mean) / self.seg_std
        else:
            im = (F.interpolate(gen_im_0_1, size=(512,512)) - self.seg_mean) / self.seg_std
        down_seg, _, _ = self.seg(im)

        if is_downsampled == False:
            down_seg = F.interpolate(down_seg, size=(self.opts.size,self.opts.size))

        return down_seg, gen_im

    def dilate_erosion(self, free_mask, device, dilate_erosion=5):
        free_mask = F.interpolate(free_mask.cpu(), size=(256, 256), mode='nearest').squeeze()
        free_mask_D, free_mask_E = cuda_unsqueeze(dilate_erosion_mask_tensor(free_mask, dilate_erosion=dilate_erosion), device)
        return free_mask_D, free_mask_E

    def setup_align_with_blend_optimizer(self, latent_1, only_interpolation= False):
        if only_interpolation:
            pass
        else:
            latent_1 = latent_1.clone().detach().to(self.opts.device).requires_grad_(True)

        latent_interpolation = torch.zeros((18, 512), requires_grad=True, device=self.opts.device)
        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        if only_interpolation:
            optimizer_align_with_blend = opt_dict[self.opts.opt_name]([latent_interpolation], lr=self.opts.learning_rate)
        else:
            optimizer_align_with_blend = opt_dict[self.opts.opt_name]([latent_1, latent_interpolation], lr=self.opts.learning_rate)
        return optimizer_align_with_blend, latent_1, latent_interpolation

    def align_images(self, img_path1, img_path2, sign='realistic', align_more_region=False, smooth=5,
                     save_intermediate=True):

        ################## img_path1: Identity Image # Face
        ################## img_path2: Structure Image # Hair

        device = self.opts.device
        output_dir = self.opts.output_dir
        embedding_dir = self.opts.embedding_dir # output_dir # todo self.opts.embedding_dir
        is_downsampled = self.opts.size > 256

        # Step 2-1 : Warp target hair image with source pose
        target_mask, hair_mask_target, hair_mask1, hair_mask2, warped_latent_2 = \
            self.create_target_segmentation_mask(img_path1=img_path1, img_path2=img_path2, sign=sign,
                                                 save_intermediate=save_intermediate, is_downsampled = is_downsampled)

        im_name_1 = os.path.splitext(os.path.basename(img_path1))[0] # source image : identity
        im_name_2 = os.path.splitext(os.path.basename(img_path2))[0] # target image : hairstyle

        latent_FS_path_1 = os.path.join(embedding_dir, 'FS', f'{im_name_1}.npz')
        latent_FS_path_2 = os.path.join(embedding_dir, 'FS', f'{im_name_2}.npz')

        latent_1, latent_F_1 = load_FS_latent(latent_FS_path_1, device) # [1,18,512], [1, 512, 32, 32]
        latent_2, latent_F_2 = load_FS_latent(latent_FS_path_2, device)

        latent_W_path_1 = os.path.join(embedding_dir, 'W+', f'{im_name_1}.npy')
        latent_W_path_2 = os.path.join(embedding_dir, 'W+', f'{im_name_2}.npy')

        pbar = tqdm(range(self.opts.align_steps1), desc='Align Step 1', leave=False) # Optimize src latent with Aligned Mask

        # Step 2-2 : Blend Source Face Image and Warped Target Hair Image
        if self.opts.blend_with_align:

            ## get reasonable masks for perceptual loss
            I_1 = load_image(img_path1, downsample=is_downsampled).to(device).unsqueeze(0)
            I_2 = load_image(img_path2, downsample=is_downsampled).to(device).unsqueeze(0)  ## downsample True : 256, 256

            down_seg2, _, _ = self.seg(self.preprocess_img(img_path2, is_downsampled=is_downsampled))
            down_seg2 = F.interpolate(down_seg2, size=(256, 256))
            seg_target2 = torch.argmax(down_seg2, dim=1).long()
            hair_mask_2 = (seg_target2 == 10) * 1.0

            # HM_1D, _ = cuda_unsqueeze(dilate_erosion_mask_path(img_path1, self.seg, is_downsampled=is_downsampled), device)
            # HM_2D, HM_2E = cuda_unsqueeze(dilate_erosion_mask_path(img_path2, self.seg, is_downsampled=is_downsampled), device)
            if self.opts.align_src_first:
                aligned_latent_1 = self.optimize_src_latent_with_aligned_mask(latent_W_path_1, target_mask, latent_1, is_downsampled)
                aligned_latent_1 = aligned_latent_1.detach().clone().to(device)
                optimizer_align_with_blend, _, latent_interpolation = self.setup_align_with_blend_optimizer(None, only_interpolation=True)
            else:
                optimizer_align_with_blend, aligned_latent_1, latent_interpolation = self.setup_align_with_blend_optimizer(latent_1)
            warped_latent_2 = warped_latent_2.detach().clone().to(device)
            with torch.no_grad():
                warped_gen_im, _ = self.net.generator([warped_latent_2], input_is_latent=True, return_latents=False,
                                                      start_layer=0, end_layer=8)
                warped_gen_im_256 = self.downsample_256(warped_gen_im)
                warped_gen_im_256_0_1 = ((warped_gen_im_256 + 1) / 2).clamp(0, 1)

            # cur_check_dir = f'{self.opts.output_dir}Align_with_Blend/'
            # os.makedirs(cur_check_dir, exist_ok=True)

            target_hairmask = (target_mask == 10) * 1.0
            target_hairmask_down_32 = F.interpolate(target_hairmask.float().unsqueeze(0), size=(32, 32), mode='bicubic')

            free_mask = 1 - (1 - hair_mask1.unsqueeze(0)) * (1 - target_hairmask.unsqueeze(0))
            free_mask, _ = self.dilate_erosion(free_mask, device, dilate_erosion=smooth)
            free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode='bicubic')[0]
            interpolation_low = 1 - free_mask_down_32

            for step in pbar:
                optimizer_align_with_blend.zero_grad()
                if is_downsampled:
                    latent_mixed = aligned_latent_1 + latent_interpolation.unsqueeze(0) * (
                                warped_latent_2 - aligned_latent_1)
                else:
                    latent_mixed = aligned_latent_1 + latent_interpolation.unsqueeze(0)[:, :14] * (
                                warped_latent_2 - aligned_latent_1)


                # 각각 shape check

                aligned_F_1, _ = self.net.generator([aligned_latent_1], input_is_latent=True, return_latents=False,
                                                    start_layer=0, end_layer=3)
                warped_F_2, _ = self.net.generator([warped_latent_2], input_is_latent=True, return_latents=False,
                                                   start_layer=0, end_layer=3)

                latent_F_mixed = aligned_F_1 + target_hairmask_down_32 * (warped_F_2 - aligned_F_1)
                latent_F_mixed = latent_F_mixed + interpolation_low.unsqueeze(0) * (latent_F_1 - latent_F_mixed)
                ####

                I_G, _ = self.net.generator([latent_mixed], input_is_latent=True, return_latents=False, start_layer=4,
                                            end_layer=8, layer_in=latent_F_mixed)
                I_G_0_1 = ((I_G + 1) / 2).clamp(0, 1)  # for saving
                I_1_0_1 = ((I_1 + 1) / 2).clamp(0, 1)
                loss_dict = {}
                #### Loss1 : Target Segmentation Loss
                if self.opts.align_src_first:
                    ce_loss = 0
                else:
                    if is_downsampled:
                        im = (self.downsample(I_G_0_1) - self.seg_mean) / self.seg_std
                    else:
                        im = (F.interpolate(I_G_0_1, size=(512, 512)) - self.seg_mean) / self.seg_std
                    down_seg, _, _ = self.seg(im)
                    ce_loss = self.loss_builder.cross_entropy_loss(down_seg, target_mask)  # 1, 16, 512, 512, 1
                    loss_dict["ce_loss"] = ce_loss.item()

                target_hairmask_down_256 = \
                F.interpolate(target_hairmask.float().unsqueeze(0), size=(256, 256), mode='bicubic')[0]
                hair_mask1_down_256 = F.interpolate(hair_mask1.float().unsqueeze(0), size=(256, 256), mode='bicubic')[0]
                #### Loss2 : Source Face Perceptual Loss
                # I_1 * Hair 가 아닌 영역 Source_Face_Region , I_G * Target_hair 가 아닌 영역 교집합을 구해야한다.
                # Mask : Target_Hair 가 아닌 영역 * I_1 의 Hair 가 아닌 영역 을 마스크로 한 LPIPS Loss
                no_hair_region = (1 - target_hairmask_down_256) * (1 - hair_mask1_down_256)
                face_loss = self.percept_with_mask(self.downsample_256(I_G), I_1,
                                                   mask=no_hair_region)  # todo : check the range
                loss_dict['face_loss'] = face_loss.item()

                #### Loss3 : Target Hair Perceptual Loss
                # 만들어진 영역의 Hair 와 latent_2 로 부터 만들어지는 영역의 Hair 랑 동일해야한다.
                # Mask : Target_Hair Region

                hair_loss = self.percept_with_mask(self.downsample_256(I_G), warped_gen_im_256,
                                                   mask=target_hairmask_down_256)  # todo : check the range
                loss_dict['hair_loss'] = hair_loss.item()

                if self.opts.align_src_first:
                    loss = face_loss + hair_loss
                else:
                    loss = face_loss + hair_loss + ce_loss

                if self.opts.blend_with_gram:
                    hairstyle_loss = self.loss_builder.style_loss(self.downsample_256(I_G), I_2,
                                                                  mask1=F.interpolate(target_hairmask.unsqueeze(0),
                                                                                      size=(256, 256)),
                                                                  mask2=hair_mask_2.unsqueeze(0)) * 10
                    loss_dict['hairstyle_loss'] = hairstyle_loss.item()
                    loss += hairstyle_loss
                loss.backward()
                optimizer_align_with_blend.step()
                #print(loss_dict)
                # for debugging
                # if step % 100 == 0 :
                #     #print(loss_dict)
                #     save_im = toPIL(
                #         I_G_0_1.squeeze().cpu())  # save_im = toPIL(((I_G_0_1 + 1) / 2).clamp(0, 1).squeeze().cpu())
                #     toPIL((no_hair_region * I_1_0_1).squeeze().cpu()).save(
                #         cur_check_dir + f'{im_name_1}_with_{im_name_2}_nohairreigion_{step}.png')
                #     toPIL((target_hairmask_down_256 * warped_gen_im_256_0_1).squeeze().cpu()).save(
                #         cur_check_dir + f'{im_name_1}_with_{im_name_2}_hairreigion_{step}.png')
                #     save_im.save(cur_check_dir + f'{im_name_1}_with_{im_name_2}_hair_{step}.png')

            latent_in = latent_mixed
            if self.opts.save_all:
                gram_add = ''
                if self.opts.blend_with_gram:
                    gram_add = '_gram'

                save_im = toPIL(I_G_0_1.squeeze().cpu())
                if self.opts.save_all:
                    save_im.save(os.path.join(self.opts.save_dir, f'4_blend_and_alignment_img.png'))
            save_im.save(os.path.join(self.opts.output_dir, f'{im_name_1}_{im_name_2}.png'))
        else:
            pass

        #print('down_seg shape : ',down_seg.shape)

        # intermediate_align, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,start_layer=0, end_layer=3)
        # intermediate_align = intermediate_align.clone().detach()
        #
        # #############################################
        #
        # latent_F_out_new, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
        #                                          start_layer=0, end_layer=3) # Target Mask shape 을 만들면서, Target hair style 을 따르도록 배운 W 부터 만들어진 F (512, 32, 32)
        # latent_F_out_new = latent_F_out_new.clone().detach()
        #
        # free_mask = 1 - (1 - hair_mask1.unsqueeze(0)) * (1 - hair_mask_target)
        #
        # ##############################
        # free_mask, _ = self.dilate_erosion(free_mask, device, dilate_erosion=smooth)
        # ##############################
        #
        # free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode='bicubic')[0]
        # interpolation_low = 1 - free_mask_down_32
        #
        #
        # latent_F_mixed = intermediate_align + interpolation_low.unsqueeze(0) * (
        #         latent_F_1 - intermediate_align) # 1
        #
        # ## trg_mask * trg_img visualization code
        # if self.opts.size == 256: # size 256, 256
        #     img = Image.fromarray((np.array(Image.open(img_path2).convert('RGB')) * np.array(hair_mask2.cpu().squeeze().unsqueeze(-1))).astype(np.uint8))
        # else:
        #     img = Image.fromarray((np.array(Image.open(img_path2).convert('RGB').resize((512,512))) * np.array(hair_mask2.cpu().squeeze().unsqueeze(-1))).astype(np.uint8))
        # #img.save(os.path.join(self.opts.output_dir,'Vis_Mask',f'{im_name_1}_{im_name_2}_trg_mask^trg_img.png'))
        # if not align_more_region:
        #     free_mask = hair_mask_target * hair_mask2.unsqueeze(0)
        #     ##########################
        #     _, free_mask = self.dilate_erosion(free_mask, device, dilate_erosion=self.opts.smooth)
        #     ##########################
        #     free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode='bicubic')[0]
        #     interpolation_low = 1 - free_mask_down_32
        #
        # latent_F_mixed = latent_F_out_new + interpolation_low.unsqueeze(0) * (latent_F_mixed - latent_F_out_new) # 2
        #
        # free_mask = F.interpolate((hair_mask2.unsqueeze(0) * hair_mask_target).float(), size=(256, 256), mode='nearest').to(self.opts.device
        # ##########################
        # _, free_mask = self.dilate_erosion(free_mask, device, dilate_erosion=self.opts.smooth)
        # ##########################
        # free_mask_down_32 = F.interpolate(free_mask.float(), size=(32, 32), mode='bicubic')[0]
        # interpolation_low = 1 - free_mask_down_32
        #
        # latent_F_mixed = latent_F_2 + interpolation_low.unsqueeze(0) * (latent_F_mixed - latent_F_2) # 3
        #
        # with torch.no_grad():
        #     gen_im, _ = self.net.generator([latent_mixed], input_is_latent=True, return_latents=False, start_layer=4,end_layer=8, layer_in=latent_F_mixed)
        #self.save_align_results(im_name_1, im_name_2, sign, gen_im, latent_mixed, latent_F_mixed,save_intermediate=save_intermediate, save_name='latent_mixed')

    def optimize_src_latent_with_aligned_mask(self, latent_W_path_1, target_mask, latent_1, is_downsampled):

        # todo : self.opts.align_steps_n
        pbar = tqdm(range(140), desc='Align Step 1', leave=False)  # Optimize src latent with Aligned Mask
        optimizer_align, latent_align_1 = self.setup_align_optimizer(
            latent_W_path_1)  # latent_W_path_1 = os.path.join(output_dir, 'W+', f'{im_name_1}.npy')
        for step in pbar:
            optimizer_align.zero_grad()
            latent_in = torch.cat([latent_align_1[:, :6, :], latent_1[:, 6:, :]], dim=1)
            down_seg, _ = self.create_down_seg(latent_in, is_downsampled=is_downsampled)

            loss_dict = {}
            ##### Cross Entropy Loss
            ce_loss = self.loss_builder.cross_entropy_loss(down_seg, target_mask)
            loss_dict["ce_loss"] = ce_loss.item()
            loss = ce_loss
            #### TODO not finished

            loss.backward()
            optimizer_align.step()

        _, gen_im = self.create_down_seg(latent_in, is_downsampled=is_downsampled)
        save_im = toPIL(((gen_im + 1) / 2).clamp(0, 1).squeeze().cpu())
        if self.opts.save_all:
            save_im.save(os.path.join(self.opts.save_dir, '4_Aligned_src_img.png'))

        return latent_align_1

    def save_align_results(self, im_name_1, im_name_2, sign, gen_im, latent_in, latent_F, save_intermediate=False, save_name ='1'):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))

        save_dir = os.path.join(self.opts.output_dir, 'Align_{}'.format(sign))
        os.makedirs(save_dir, exist_ok=True)

        latent_path = os.path.join(save_dir, '{}_{}.npz'.format(im_name_1, im_name_2))
        if save_intermediate:
            image_path = os.path.join(save_dir, '{}_{}_{}.png'.format(im_name_1, im_name_2, save_name))
            save_im.save(image_path)
        if self.opts.save_all:
            save_im.save(os.path.join(self.opts.save_dir, '5_latent_F_mixed_output.png'))
        np.savez(latent_path, latent_in=latent_in.detach().cpu().numpy(), latent_F=latent_F.detach().cpu().numpy())

    def warp_target(self, img_path2, src_kp_hm, src_ypr, img_path1):

        im_name_1 =  os.path.splitext(os.path.basename(img_path1))[0]
        output_dir = self.opts.output_dir
        embedding_dir = self.opts.embedding_dir
        is_downsampled = self.opts.size > 256
        device = self.opts.device
        im_name_2 = os.path.splitext(os.path.basename(img_path2))[0]  # target image : hair

        latent_FS_path_2 = os.path.join(embedding_dir, 'FS', f'{im_name_2}.npz')
        latent_W_path_2 = os.path.join(embedding_dir, 'W+', f'{im_name_2}.npy')
        latent_2, latent_F_2 = load_FS_latent(latent_FS_path_2, device)  # [1,18,512], [1, 512, 32, 32]


        # todo : change 40 to self.opts.warp_steps
        optimizer_warp_w, latent_warped_2 = self.setup_align_optimizer(latent_W_path_2)
        pbar = tqdm(range(self.opts.warp_steps), desc='Warp Target Step 1', leave=False)
        latent_W_optimized = latent_warped_2
        latent_F_optimized = None
        mode = 'w+_total'
        if self.opts.warp_front_part:
            mode = 'w+_6'

        cur_check_dir = None
        # cur_check_dir = f'{self.opts.output_dir}warped_result_{mode}_{self.opts.kp_type}/'
        # if self.opts.warp_loss_with_prev_list is not None:
        #     cur_check_dir += f'{self.opts.warp_loss_with_prev_list}/'
        # os.makedirs(cur_check_dir, exist_ok=True)

        warped_down_seg = None
        latent_in, warped_down_seg = self.optimize_warping(pbar, optimizer_warp_w, latent_W_optimized, latent_F_optimized, mode, is_downsampled, src_kp_hm, im_name_1, im_name_2, cur_check_dir, img_path1, img_path2)
        latent_F = None

        ## save img
        gen_im = self.save_warp_result(latent_F, latent_in, is_downsampled, cur_check_dir,im_name_2, im_name_1)
        return gen_im, latent_in, warped_down_seg
       
    def save_warp_result(self, latent_F, latent_in, is_downsampled, cur_check_dir, im_name_2, im_name_1):
        if latent_F is not None:
            gen_im, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
                                           start_layer=4,
                                           end_layer=8, layer_in=latent_F)
        else:
            _, gen_im = self.create_down_seg(latent_in, is_downsampled=is_downsampled)
        if cur_check_dir is not None:
            save_im = toPIL(((gen_im + 1) / 2).clamp(0, 1).squeeze().cpu())
            save_im.save(cur_check_dir + f'{im_name_2}_with_{im_name_1}_pose.png')
        return gen_im

    def _loss_lpips(self, gen_im, ref_im, **kwargs): # added 220208
        return self.percept(gen_im, ref_im).sum()

    # 220303 added
    def get_sp_mask(self, ref_im256_slic, seg_hair_ref256_slic, prev_centroids=None, im_path=None, im1024=None):

        if im_path is not None:
            ref_im = Image.open(im_path).convert('RGB')
            ref_im1024 = self.image_transform1024(ref_im).unsqueeze(0).to(self.opts.device)
        else:
            ref_im1024 = im1024

        slic_segments, prev_centroids, closest_indices = slic_custom(ref_im256_slic, mask=seg_hair_ref256_slic,
                                                                     compactness=self.slic_compactness,
                                                                     n_segments=self.slic_numSegments if prev_centroids is None else prev_centroids.shape[1],
                                                                     sigma=5, previous_centroids=prev_centroids)
        n_seg = len(np.unique(slic_segments)) - 1

        grid_mask256 = torch.zeros(n_seg, 1, slic_segments.shape[0], slic_segments.shape[1])  # 256
        for idx in range(n_seg):
            grid_mask256[idx][0][slic_segments == idx + 1] = 1
        grid_mask256 = grid_mask256.to(self.opts.device)

        grid_mask1024 = F.interpolate(grid_mask256, size=(1024, 1024))  # 6, 1, 1024, 1024

        if ref_im1024.shape[2] == 256:
            crop_mask = grid_mask256.clone()
        elif ref_im1024.shape[2] == 1024:
            crop_mask = grid_mask1024.clone()

        grid_mask_large256 = []
        crop_indices = []
        for idx in range(n_seg):
            _, idx_y, idx_x = (crop_mask[idx] == 1).nonzero(as_tuple=True)
            min_x, min_y, max_x, max_y = torch.min(idx_x).item(), torch.min(idx_y).item(), torch.max(
                idx_x).item(), torch.max(idx_y).item()

            crop_indices.append([min_x, min_y, max_x, max_y])
            # grid_mask[idx][:,min_x:max_x,min_y:max_y]
            grid_mask_large256.append(
                F.interpolate(crop_mask[idx][:, min_y:max_y, min_x:max_x].unsqueeze(0), size=(256, 256))[0])

        grid_mask_large256 = torch.stack(grid_mask_large256)
        sp_ref_im = []

        for crop_idx in crop_indices:
            min_x, min_y, max_x, max_y = crop_idx
            sp_ref_im.append(
                F.interpolate(ref_im1024[0][:, min_y:max_y, min_x:max_x].unsqueeze(0), size=(256, 256))[0])
        sp_ref_im = torch.stack(sp_ref_im)

        return grid_mask_large256.to(self.opts.device), sp_ref_im.to(
            self.opts.device), slic_segments, prev_centroids, closest_indices

    def optimize_warping(self, pbar, optimizer_warp, latent_W_optimized, latent_F_optimized, mode, is_downsampled,
                         src_kp_hm,
                         im_name_1, im_name_2, cur_check_dir, img_path1, img_path2):

        if 'w+_6' == mode:
            latent_end = latent_W_optimized[:, 6:, :].clone().detach()

        # for style_loss
        ref_im = Image.open(img_path2).convert('RGB')
        ref_im256 = ref_im.resize((256, 256), PIL.Image.LANCZOS)
        ref_im256 = self.image_transform(ref_im256).unsqueeze(0).to(self.opts.device)

        self.seg_transform = transforms.Compose([transforms.Resize((512, 512)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
        ref_im512 = self.seg_transform(ref_im).unsqueeze(0).to(self.opts.device)
        down_seg_ref, _, _ = self.seg(ref_im512)  # 512 512
        ref_seg = torch.argmax(down_seg_ref.clone().detach(), dim=1).long()
        seg_hair_ref = torch.where((ref_seg == 10), torch.ones_like(ref_seg),
                                   torch.zeros_like(ref_seg))
        seg_hair_ref256 = F.interpolate(seg_hair_ref.unsqueeze(0).float(), size=(256, 256))

        prev_im = ref_im256
        prev_seg = ref_seg

        if 'delta_w' in self.opts.warp_loss_with_prev_list:
            latent_W_optimized_prev = latent_W_optimized[:, :6, :].clone().detach() # todo : changed, front 만 동작하게 되어있음

        if 'style_hair_slic_large' in self.opts.warp_loss_with_prev_list:
            self.slic_compactness = 20  # 100
            self.slic_numSegments = 5 # todo : opts
            lambda_hair = 1000 #  todo : opts

            # cur_check_dir += f'{self.slic_compactness}_{self.slic_numSegments}_{lambda_hair}/'
            # os.makedirs(cur_check_dir, exist_ok=True)

            ref_im256_slic = (((ref_im256[0] + 1) / 2).clamp(0, 1)).permute(1, 2, 0).detach().cpu().numpy()
            seg_hair_ref256_slic = seg_hair_ref256[0].detach().cpu().numpy()
            prev_slic_segments, prev_centroids, _ = slic_custom(ref_im256_slic, mask=seg_hair_ref256_slic,
                                                                compactness=self.slic_compactness,
                                                                n_segments=self.slic_numSegments, sigma=5)

        for step in pbar:
            optimizer_warp.zero_grad()
            if 'w+_total' == mode:
                latent_in = latent_W_optimized  # torch.cat([latent_warped_2[:, :6, :], latent_2[:, 6:, :]], dim=1) ## 220205
                # latent_in = torch.cat([latent_warped_2[:, :6, :], latent_2[:, 6:, :]], dim=1) # 220205
                down_seg, gen_im = self.create_down_seg(latent_in, is_downsampled=is_downsampled)
            elif 'w+_6' == mode:
                latent_in = torch.cat([latent_W_optimized[:, :6, :], latent_end], dim=1)
                down_seg, gen_im = self.create_down_seg(latent_in, is_downsampled=is_downsampled)
            else:
                # todo : implement cat latent vector some part fixed, the other part to be optimized
                pass

            loss_dict = {}
            loss = 0

            # 220303 added
            gen_im1024 = gen_im.clone()
            gen_im1024 = ((gen_im1024 + 1) / 2).clamp(0, 1)

            if self.opts.size > 256:
                gen_im = F.interpolate(gen_im, size=(256, 256))
            gen_im = ((gen_im + 1) / 2).clamp(0, 1)
            gen_kp_hm = self.kp_extractor.face_alignment_net(gen_im) # 1,68,64,64

            # keypoint loss
            kp_loss = self.l2(src_kp_hm[:, :], gen_kp_hm[:, :])  # no restriction
            lambda_kp = 1000 # todo opts
            loss_dict["kp_loss"] = kp_loss.item() * lambda_kp
            loss += kp_loss * lambda_kp

            # early stop : if Keypoint loss is below 0.1
            if kp_loss * lambda_kp < 0.05:
                print(f"Early stop, Key point loss below 0.05 : {kp_loss:.3f}")
                break

            # perceptual loss (lpips)
            curr_seg = torch.argmax(down_seg.clone().detach(), dim=1).long()

            if self.opts.warp_loss_with_prev_list is not None:
                # 220303 added
                try:
                    if 'style_hair_slic_large' in self.opts.warp_loss_with_prev_list:

                        seg_hair_gen = torch.where((curr_seg == 10), torch.ones_like(curr_seg),
                                                   torch.zeros_like(curr_seg))
                        seg_hair_gen256 = F.interpolate(seg_hair_gen.unsqueeze(0).float(), size=(256, 256))

                        gen_im256_slic = gen_im[0].permute(1, 2, 0).detach().cpu().numpy()
                        seg_hair_gen256_slic = seg_hair_gen256[0].detach().cpu().numpy()

                        if step == 0:
                            prev_centroids_ref = prev_centroids.copy()

                        sp_gen_mask_large256, sp_gen_im, prev_slic_segments, prev_centroids, closest_indices \
                            = self.get_sp_mask(gen_im256_slic, seg_hair_gen256_slic, prev_centroids=prev_centroids,
                                               im_path=None, im1024=gen_im1024)

                        if step == 0:
                            sp_ref_mask_large256, sp_ref_im, ref_slic_segments, ref_centroids, _ \
                                = self.get_sp_mask(ref_im256_slic, seg_hair_ref256_slic, prev_centroids=prev_centroids_ref,
                                                   im_path=img_path2)
                            points = prev_centroids[0].copy()  # n, 2
                            points_prev = ref_centroids[0].copy()  # 6, 2
                            points_repeat = np.repeat(np.array(points)[:, np.newaxis], ref_centroids.shape[0],
                                                      axis=1)  # 3, 1, 2 -> 3, 6, 2
                            closest_indices = np.argmin(np.linalg.norm(points_repeat - points_prev[np.newaxis,], axis=2),
                                                        axis=1)  # 1 6 2

                        sp_ref_im, sp_ref_mask_large256 = sp_ref_im[closest_indices], sp_ref_mask_large256[closest_indices]
                        hair_loss = self.loss_builder.style_loss(sp_gen_im, sp_ref_im, mask1=sp_gen_mask_large256,
                                                                 mask2=sp_ref_mask_large256)

                        loss_dict["style_loss_prev_hair_large_slic"] = hair_loss.item() * lambda_hair
                        loss += hair_loss * lambda_hair  # 0.001
                except :
                    pass

                if 'delta_w' in self.opts.warp_loss_with_prev_list:  # 1-hair 의 교집합
                    delta_w_loss = self.l2(latent_W_optimized[:, :6, :], latent_W_optimized_prev)
                    lambda_delta_w = 1000
                    loss_dict["delta_w"] = delta_w_loss.item() * lambda_delta_w
                    loss += delta_w_loss * lambda_delta_w

                if 'style_hair' in self.opts.warp_loss_with_prev_list:
                    seg_hair_gen = torch.where((curr_seg == 10), torch.ones_like(curr_seg),
                                               torch.zeros_like(curr_seg))
                    seg_hair_gen256 = F.interpolate(seg_hair_gen.unsqueeze(0).float(), size=(256, 256))

                    hair_loss = self.loss_builder.style_loss(gen_im, ref_im256, mask1=seg_hair_gen256,
                                                             mask2=seg_hair_ref256)

                    lambda_hair = 100
                    loss += hair_loss / lambda_hair
                    loss_dict["hair_loss"] = hair_loss.item() / lambda_hair



            latent_W_optimized_prev = latent_W_optimized[:, :6, :].clone().detach() # todo :이것도 6 기준 opt 로

            loss.backward()
            optimizer_warp.step()
            # if step % 10 == 0 : ### warped result save step size
            #     cur_check_dir = f'{self.opts.output_dir}check_hair/'
            #     os.makedirs(cur_check_dir, exist_ok=True)
            #     print(f'{step}: ', loss_dict)
            #     save_im = toPIL(gen_im.squeeze().cpu())
            #     aaa = torch.zeros((3, 256, 256))
            #     kp_prob =  F.interpolate(torch.max(gen_kp_hm, dim=1)[0].unsqueeze(0).cpu(), size=(256, 256))
            #     aaa[0] = kp_prob[0][0]
            #     kp_im = toPIL(torch.cat((gen_im.squeeze().cpu(), aaa), dim=-1))
            #
            #     save_im.save(cur_check_dir + f'{im_name_2}_with_{im_name_1}_pose_{step}.png')
            #     # added for debug
            #     if 'style_hair_slic_large' in self.opts.warp_loss_with_prev_list:
            #         save_image(torch.cat([sp_gen_im * sp_gen_mask_large256, sp_ref_im * sp_ref_mask_large256]),
            #                    cur_check_dir + f'{im_name_2}_with_{im_name_1}_sp_gen_ref_{step}.png', normalize=True,
            #                    nrow=sp_gen_im.shape[0])

            prev_im = gen_im.clone().detach()
            prev_seg = torch.argmax(down_seg.clone().detach(), dim=1).long()

        if self.opts.save_all:
            save_im = toPIL(gen_im.squeeze().cpu())
            save_im.save(os.path.join(self.opts.save_dir, '1_warped_img.png'))
        if 'F' in mode:
            return latent_F_optimized, latent_W_optimized
        if self.opts.warped_seg:
            return latent_in, prev_seg
        else:
            return latent_in, None