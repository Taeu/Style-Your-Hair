from typing import Tuple

import torch
from torch import nn
from models.Net import Net
import numpy as np
import os
from functools import partial
from utils.bicubic import BicubicDownSample
from datasets.image_dataset import ImagesDataset
from losses.embedding_loss import EmbeddingLossBuilder
from torch.utils.data import DataLoader
from tqdm import tqdm
import PIL
import torchvision
from utils.data_utils import convert_npy_code, load_FS_latent # 0224 added

import face_recognition
from models.face_parsing.model import BiSeNet
from torchvision import transforms
from skimage import io
import cv2
from PIL import Image

import torch.nn.functional as F

toPIL = torchvision.transforms.ToPILImage()


class Embedding(nn.Module):

    def __init__(self, opts):
        super(Embedding, self).__init__()
        self.opts = opts
        self.net = Net(self.opts)
        self.load_downsampling()
        self.setup_embedding_loss_builder()

    def load_downsampling(self):
        factor = self.opts.size // 256
        self.downsample = BicubicDownSample(factor=factor, cuda=self.opts.device == 'cuda')

    def setup_W_optimizer(self):

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        latent = []
        if (self.opts.tile_latent):
            tmp = self.net.latent_avg.clone().detach().to(self.opts.device)
            tmp.requires_grad = True
            for i in range(self.net.layer_num):
                latent.append(tmp)
            optimizer_W = opt_dict[self.opts.opt_name]([tmp], lr=self.opts.learning_rate)
        else:
            for i in range(self.net.layer_num):
                tmp = self.net.latent_avg.clone().detach().to(self.opts.device)
                tmp.requires_grad = True
                latent.append(tmp)
            optimizer_W = opt_dict[self.opts.opt_name](latent, lr=self.opts.learning_rate)

        return optimizer_W, latent


    def setup_FS_optimizer(self, latent_W, F_init):

        latent_F = F_init.clone().detach().requires_grad_(True)
        latent_S = []
        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        for i in range(self.net.layer_num):

            tmp = latent_W[0, i].clone()

            if i < self.net.S_index:
                tmp.requires_grad = False
            else:
                tmp.requires_grad = True

            latent_S.append(tmp)

        optimizer_FS = opt_dict[self.opts.opt_name](latent_S[self.net.S_index:] + [latent_F], lr=self.opts.learning_rate)

        return optimizer_FS, latent_F, latent_S


    def setup_dataloader(self, image_path=None):

        self.dataset = ImagesDataset(opts=self.opts,image_path=image_path)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        print("Number of images: {}".format(len(self.dataset)))

    def setup_embedding_loss_builder(self):
        self.loss_builder = EmbeddingLossBuilder(self.opts)

    def match_scale(self, source_path: str, target_path: str) -> Tuple[np.array, np.array]:

        background = io.imread('background.jpeg')
        background = cv2.resize(background, (1024, 1024))

        def _setup_segmentation_network():
            seg = BiSeNet(n_classes=16)
            seg.to(self.opts.device)
                
            seg.load_state_dict(torch.load('pretrained_models/seg.pth', map_location=self.opts.device))
            for param in seg.parameters():
                param.requires_grad = False
            seg.eval()

            return seg

        def _get_scale(image: np.array) -> int:
            box = face_recognition.face_locations(image)[0]
            return box[2] - box[0]

        def _get_center(image: np.array) -> Tuple[float, float]:
            landmarks = face_recognition.face_landmarks(image)[0]

            chin = landmarks['chin']
            xc = np.mean([p[0] for p in chin])
            yc = np.mean([p[1] for p in chin])

            return xc, yc

        seg = _setup_segmentation_network()
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        _target = io.imread(target_path)
        _source = io.imread(source_path)

        source_scale = _get_scale(_source)
        target_scale = _get_scale(_target)
        ratio = source_scale / target_scale

        target = cv2.resize(_target, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)

        if ratio < 1:
            sh, sw, _ = _target.shape
            th, tw, _ = target.shape
            tp = int((sh - th) / 2)
            bp = sh - th - tp
            lp = int((sw - tw) / 2)
            rp = sw - tw - lp

            target = cv2.copyMakeBorder(target, tp, bp, lp, rp, borderType=cv2.BORDER_CONSTANT, value=[127, 127, 127])
                    
            target_pt = to_tensor(Image.fromarray(target).resize((512, 512), Image.BILINEAR)).unsqueeze(0).to(self.opts.device)
            out = seg(target_pt)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            parsing = cv2.resize(parsing, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)[..., np.newaxis]
            parsing = np.repeat(parsing, 3, axis=-1)

            target  = np.where(parsing == [0, 0, 0], background, target)
            
        elif ratio > 1:

            oh, ow, _ = _target.shape
            h, w, _ = target.shape
            xc, yc = _get_center(target)

            dx = w - ow
            dy = h - oh

            if (xc / w) > 0.8:  # 왼쪽 crop
                if (yc / h) > 0.8:  # 위쪽 crop
                    target = target[dy:, dx:, :]
                elif (yc / h) < 0.2:  # 아래쪽 crop
                    target = target[:-dy, dx:, :]
                else:
                    tdy = dy // 2
                    bdy = dy - tdy
                    target = target[tdy:-bdy, dx:, :]

            elif (xc / w) < 0.2:  # 오른쪽 crop
                if (yc / h) > 0.8:  # 위쪽 crop
                    target = target[dy:, :-dx, :]
                elif (yc / h) < 0.2:  # 아래쪽 crop
                    target = target[:-dy, :-dx:, :]
                else:
                    tdy = dy // 2
                    bdy = dy - tdy
                    target = target[tdy:-bdy, :-dx, :]

            else:
                ldx = dx // 2
                rdx = dx - ldx

                if (yc / h) > 0.8:  # 위쪽 crop
                    target = target[dy:, ldx:-rdx, :]
                elif (yc / h) < 0.2:  # 아래쪽 crop
                    target = target[:-dy, ldx:-rdx:, :]
                else:
                    tdy = dy // 2
                    bdy = dy - tdy
                    target = target[tdy:-bdy, ldx:-rdx, :]

        assert _source.shape == target.shape
        
        return _source, target


    def invert_images_in_W(self, image_path=None):
        self.setup_dataloader(image_path=image_path)
        device = self.opts.device
        ibar = tqdm(self.dataloader, desc='Images')

        for ref_im_H, ref_im_L, ref_name in ibar:
            print(ref_name[0]) # todo : erease
            optimizer_W, latent = self.setup_W_optimizer()
            pbar = tqdm(range(self.opts.W_steps), desc='Embedding', leave=False)
            if self.opts.size == 256:
                ref_im_H = F.interpolate(ref_im_H, size=(256,256))

            for step in pbar:
                optimizer_W.zero_grad()
                latent_in = torch.stack(latent).unsqueeze(0)

                gen_im, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False)
                im_dict = {
                    'ref_im_H': ref_im_H.to(device),
                    'ref_im_L': ref_im_L.to(device),
                    'gen_im_H': gen_im,
                    'gen_im_L': self.downsample(gen_im)
                }
                #import pdb ; pdb.set_trace()
                loss, loss_dic = self.cal_loss(im_dict, latent_in)

                loss.backward()
                optimizer_W.step()

                if self.opts.verbose:
                    pbar.set_description('Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}'
                                         .format(loss, loss_dic['l2'], loss_dic['percep'], loss_dic['p-norm']))

                if self.opts.save_intermediate and step % self.opts.save_interval== 0:
                    self.save_W_intermediate_results(ref_name, gen_im, latent_in, step)

            self.save_W_results(ref_name, gen_im, latent_in)

    def invert_images_in_W_with_pre_align(self, images):
        device = self.opts.device
        ibar = tqdm(self.dataloader, desc='Images')
        for ref_im_H, ref_im_L, ref_name in ibar:
            optimizer_W, latent = self.setup_W_optimizer()
            pbar = tqdm(range(self.opts.W_steps), desc='Embedding', leave=False)
            for step in pbar:
                optimizer_W.zero_grad()
                latent_in = torch.stack(latent).unsqueeze(0)

                gen_im, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False)
                im_dict = {
                    'ref_im_H': ref_im_H.to(device),
                    'ref_im_L': ref_im_L.to(device),
                    'gen_im_H': gen_im,
                    'gen_im_L': self.downsample(gen_im)
                }

                loss, loss_dic = self.cal_loss(im_dict, latent_in)
                loss.backward()
                optimizer_W.step()

                if self.opts.verbose:
                    pbar.set_description('Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}'
                                         .format(loss, loss_dic['l2'], loss_dic['percep'], loss_dic['p-norm']))

                if self.opts.save_intermediate and step % self.opts.save_interval== 0:
                    self.save_W_intermediate_results(ref_name, gen_im, latent_in, step)

            self.save_W_results(ref_name, gen_im, latent_in)

    def invert_images_in_FS(self, image_path=None, trg_name = ''):
        self.setup_dataloader(image_path=image_path)
        output_dir = self.opts.output_dir
        embedding_dir = self.opts.embedding_dir
        device = self.opts.device
        ibar = tqdm(self.dataloader, desc='Images')
        for ref_im_H, ref_im_L, ref_name in ibar:
            latent_W_path = os.path.join(embedding_dir, 'W+', f'{ref_name[0]}.npy')
            latent_W = torch.from_numpy(convert_npy_code(np.load(latent_W_path))).to(device)
            F_init, _ = self.net.generator([latent_W], input_is_latent=True, return_latents=False, start_layer=0, end_layer=3)
            optimizer_FS, latent_F, latent_S = self.setup_FS_optimizer(latent_W, F_init)


            pbar = tqdm(range(self.opts.FS_steps), desc='Embedding', leave=False)
            for step in pbar:

                optimizer_FS.zero_grad()
                latent_in = torch.stack(latent_S).unsqueeze(0)
                gen_im, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False,
                                               start_layer=4, end_layer=8, layer_in=latent_F)
                im_dict = {
                    'ref_im_H': ref_im_H.to(device),
                    'ref_im_L': ref_im_L.to(device),
                    'gen_im_H': gen_im,
                    'gen_im_L': self.downsample(gen_im)
                }

                loss, loss_dic = self.cal_loss(im_dict, latent_in)
                loss.backward()
                optimizer_FS.step()

                if self.opts.verbose:
                    pbar.set_description(
                        'Embedding: Loss: {:.3f}, L2 loss: {:.3f}, Perceptual loss: {:.3f}, P-norm loss: {:.3f}, L_F loss: {:.3f}'
                        .format(loss, loss_dic['l2'], loss_dic['percep'], loss_dic['p-norm'], loss_dic['l_F']))

            self.save_FS_results(ref_name, gen_im, latent_in, latent_F)




    def cal_loss(self, im_dict, latent_in, latent_F=None, F_init=None):
        loss, loss_dic = self.loss_builder(**im_dict)
        p_norm_loss = self.net.cal_p_norm_loss(latent_in)
        loss_dic['p-norm'] = p_norm_loss
        loss += p_norm_loss

        if latent_F is not None and F_init is not None:
            l_F = self.net.cal_l_F(latent_F, F_init)
            loss_dic['l_F'] = l_F
            loss += l_F

        return loss, loss_dic



    def save_W_results(self, ref_name, gen_im, latent_in):
        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()

        embedding_dir = os.path.join(self.opts.embedding_dir, 'W+')
        os.makedirs(embedding_dir, exist_ok=True)

        latent_path = os.path.join(embedding_dir, f'{ref_name[0]}.npy')
        image_path = os.path.join(embedding_dir, f'{ref_name[0]}.png')

        save_im.save(image_path)
        np.save(latent_path, save_latent)



    def save_W_intermediate_results(self, ref_name, gen_im, latent_in, step):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))
        save_latent = latent_in.detach().cpu().numpy()


        intermediate_folder = os.path.join(self.opts.embedding_dir, 'W+', ref_name[0])
        os.makedirs(intermediate_folder, exist_ok=True)

        latent_path = os.path.join(intermediate_folder, f'{ref_name[0]}_{step:04}.npy')
        image_path = os.path.join(intermediate_folder, f'{ref_name[0]}_{step:04}.png')

        save_im.save(image_path)
        np.save(latent_path, save_latent)


    def save_FS_results(self, ref_name, gen_im, latent_in, latent_F):

        save_im = toPIL(((gen_im[0] + 1) / 2).detach().cpu().clamp(0, 1))

        embedding_dir = os.path.join(self.opts.embedding_dir, 'FS')
        os.makedirs(embedding_dir, exist_ok=True)

        latent_path = os.path.join(embedding_dir, f'{ref_name[0]}.npz')
        image_path = os.path.join(embedding_dir, f'{ref_name[0]}.png')

        save_im.save(image_path)
        np.savez(latent_path, latent_in=latent_in.detach().cpu().numpy(),
                 latent_F=latent_F.detach().cpu().numpy())


    def set_seed(self):
        if self.opt.seed:
            torch.manual_seed(self.opt.seed)
            torch.cuda.manual_seed(self.opt.seed)
            torch.backends.cudnn.deterministic = True
