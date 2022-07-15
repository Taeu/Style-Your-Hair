import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.functional import grid_sample

from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import cv2

def gabor_fn(kernel_size, channel_in, channel_out, theta):
    # sigma_x = sigma
    # sigma_y = sigma.float() / gamma
    sigma_x = nn.Parameter(torch.ones(channel_out) * 2.0, requires_grad=False).cuda()
    sigma_y = nn.Parameter(torch.ones(channel_out) * 3.0, requires_grad=False).cuda()
    Lambda = nn.Parameter(torch.ones(channel_out) * 4.0, requires_grad=False).cuda()
    psi = nn.Parameter(torch.ones(channel_out) * 0.0, requires_grad=False).cuda()

    # Bounding box
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax+1).cuda()
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
    x_0 = torch.arange(xmin, xmax+1).cuda()
    x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()   # [channel_out, channelin, kernel, kernel]

    # Rotation
    # don't need to expand, use broadcasting, [64, 1, 1, 1] + [64, 3, 7, 7]
    x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
    y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

    # [channel_out, channel_in, kernel, kernel]
    gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2)) \
         * torch.cos(2 * math.pi / Lambda.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))

    return gb

def DoG_fn(kernel_size, channel_in, channel_out, theta):
    # params
    sigma_h = nn.Parameter(torch.ones(channel_out) * 1.0, requires_grad=False).cuda()
    sigma_l = nn.Parameter(torch.ones(channel_out) * 2.0, requires_grad=False).cuda()
    sigma_y = nn.Parameter(torch.ones(channel_out) * 2.0, requires_grad=False).cuda()

    # Bounding box
    xmax = kernel_size // 2
    ymax = kernel_size // 2
    xmin = -xmax
    ymin = -ymax
    ksize = xmax - xmin + 1
    y_0 = torch.arange(ymin, ymax+1).cuda()
    y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
    x_0 = torch.arange(xmin, xmax+1).cuda()
    x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()   # [channel_out, channelin, kernel, kernel]

    # Rotation
    # don't need to expand, use broadcasting, [64, 1, 1, 1] + [64, 3, 7, 7]
    x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
    y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

    gb = (torch.exp(-.5 * (x_theta ** 2 / sigma_h.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2))/sigma_h \
        - torch.exp(-.5 * (x_theta ** 2 / sigma_l.view(-1, 1, 1, 1) ** 2 + y_theta ** 2 / sigma_y.view(-1, 1, 1, 1) ** 2))/sigma_l) \
         / (1.0/sigma_h - 1.0/sigma_l)

    return gb

# L1 loss of orientation map
class L1OLoss(nn.Module):
    def __init__(self, channel_in=1, channel_out=1, stride=1, padding=8, orient_filter = 'gabor'): #, opt
        super(L1OLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        # self.opt = opt
        self.mode = orient_filter
        self.Tensor = torch.cuda.FloatTensor

        self.numKernels = 32
        self.kernel_size = 17
        # self.sigma_x = nn.Parameter(torch.ones(channel_out)*2.0, requires_grad=False).cuda()
        # self.sigma_y = nn.Parameter(torch.ones(channel_out)*3.0, requires_grad=False).cuda()
        # self.Lambda = nn.Parameter(torch.ones(channel_out)*4.0, requires_grad=False).cuda()
        # self.psi = nn.Parameter(torch.ones(channel_out)*0.0, requires_grad=False).cuda()

    def calOrientationGabor(self, image):
        resArray = []
        # filter the image with different orientations

        for iOrient in range(self.numKernels):
            theta = nn.Parameter(torch.ones(self.channel_out)*(math.pi*iOrient/self.numKernels), requires_grad=False).cuda()
            GaborKernel = gabor_fn(self.kernel_size, self.channel_in, self.channel_out, theta).float() #1, 1, 17, 17
            response = F.conv2d(image, GaborKernel, stride=self.stride, padding=self.padding) # 1 1 1024 1024
            resArray.append(response.clone())

        resTensor = resArray[0]
        for iOrient in range(1, self.numKernels):
            resTensor = torch.cat([resTensor, resArray[iOrient]], dim=1)

        # argmax the response
        resTensor[resTensor < 0] = 0
        maxResTensor = torch.argmax(resTensor, dim=1).float() # 0~31
        confidenceTensor = torch.max(resTensor, dim=1)[0]
        confidenceTensor = (F.tanh(confidenceTensor)+1)/2.0 # [0, 1]
        confidenceTensor = torch.unsqueeze(confidenceTensor, 1) # 1 1 1024 1024
        # cal the angle a
        orientTensor = maxResTensor * math.pi / self.numKernels
        orientTensor = torch.unsqueeze(orientTensor, 1) # 0~ 3?
        # cal the sin2a and cos2a
        orientTwoChannel = torch.cat([torch.sin(2*orientTensor), torch.cos(2*orientTensor)], dim=1) * confidenceTensor # 1 2 1024 1024 -1 ~ 1
        return orientTwoChannel, confidenceTensor

    def calOrientationDoG(self, image, mask):
        resArray = []
        # filter the image with different orientations
        for iOrient in range(self.numKernels):
            theta = nn.Parameter(torch.ones(self.channel_out)*(math.pi*iOrient/self.numKernels), requires_grad=False).cuda()
            DoGKernel = DoG_fn(self.kernel_size, self.channel_in, self.channel_out, theta)
            DoGKernel = DoGKernel.float()
            response = F.conv2d(image, DoGKernel, stride=self.stride, padding=self.padding)
            resArray.append(response.clone())

        resTensor = resArray[0]
        for iOrient in range(1, self.numKernels):
            resTensor = torch.cat([resTensor, resArray[iOrient]], dim=1)

        # argmax the response
        resTensor[resTensor < 0] = 0
        maxResTensor = torch.argmax(resTensor, dim=1).float()
        confidenceTensor = torch.max(resTensor, dim=1)[0]
        # confidenceTensor = (F.tanh(confidenceTensor)+1)/2.0 # [0, 1]
        confidenceTensor = torch.unsqueeze(confidenceTensor, 1)

        confidenceTensor = confidenceTensor * mask
        confidenceTensor = confidenceTensor / torch.max(confidenceTensor) # [0, 1]
        mask = confidenceTensor <= 0 # 0인 부분
        confidenceTensor = confidenceTensor * (~mask).float() # 0 초과인 부분 #1 - mask
        # cal the angle a
        orientTensor = maxResTensor * math.pi / self.numKernels
        orientTensor = torch.unsqueeze(orientTensor, 1)
        # cal the sin2a and cos2a
        orientTwoChannel = torch.cat([torch.sin(2*orientTensor), torch.cos(2*orientTensor)], dim=1) * confidenceTensor
        return orientTwoChannel, confidenceTensor


    def forward(self, gen_im, ref_orient, hair_mask):
        # constraint the area of hair, input_semantics is one-hot map

        # hair_mask = input_semantics[:,1,:,:]
        # hair_mask = torch.unsqueeze(hair_mask, 1)
        # RGB to gray

        # trans_image = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # gen_im = trans_image(gen_im).unsqueeze(0) # -1~ 1 / 1 3 1024 1204
        # fake_image = (gen_im+1)/2.0*255 # 0~ 255
        fake_image = gen_im * 255 # 0~ 255
        gray = 0.299*fake_image[:,0,:,:] + 0.587*fake_image[:,1,:,:] + 0.144*fake_image[:,2,:,:] # 1 1024 1024
        gray = torch.unsqueeze(gray, 1) # 1 1 1024 1024
        gray = gray.cuda()

        # cal orientation map with two channels via Gabor
        orientation_fake, confidence = self.calOrientationGabor(gray)  # [n, 2, h, w]
        # orientation_fake, confidence = self.calOrientationDoG(gray, hair_mask)
        # if True:
        #     # orientation_fake[:,1,:,:] = torch.from_numpy(cv2.GaussianBlur(orientation_fake[:,1,:,:].cpu().numpy().squeeze(), (0, 0), 4)).unsqueeze(0)
        #     # orientation_fake[:,0,:,:] = torch.from_numpy(cv2.GaussianBlur(orientation_fake[:,0,:,:].cpu().numpy().squeeze(), (0, 0), 4)).unsqueeze(0)
        #     import pdb;
        #     pdb.set_trace()
        #
        #     orient2save = (orientation_fake + 1) / 2.0
        #     # import pdb; pdb.set_trace()
        #     orient2save = torch.cat([orient2save, torch.zeros(1, 1, orient2save.shape[2], orient2save.shape[3]).cuda()], dim=1)
        #     save_image(orient2save, '/home/nas3_userM/chaeyeonchung/Barbershop_0213/vis_orient16_DoG.png')

        # if 'gabor' in self.mode:
        #     orientation_fake, confidence = self.calOrientationGabor(gray) # [n, 2, h, w]
        # else:
        #     orientation_fake, confidence = self.calOrientationDoG(gray, hair_mask)  # [n, 2, h, w]
        # transfor the label from one channel to two channels
        # if not self.opt.use_ig: # inpainting
        #     orientation_label = ref_orient / 255 * math.pi
        #     orientation_mask = torch.cat([torch.sin(2*orientation_label), torch.cos(2*orientation_label)], dim=1)
        # else:
        #   orientation_mask = ref_orient
        # print(hair_mask.shape, orientation_fake.shape)

        # cal L1 loss and the log confidence loss
        orient_loss = self.l1(orientation_fake * hair_mask, ref_orient.detach() * hair_mask)
        # if 'gabor' in self.mode:
        confidence = torch.clamp(confidence, 0.001, 1)
        confidence_loss = -torch.sum(torch.log(confidence)*hair_mask)/torch.sum(hair_mask)
        # else:
        #     confidence_gt = (hair_mask * 0 + 1) * hair_mask
        #     confidence_gt.requires_grad_(False)
        #     confidence = confidence*hair_mask
        #     confidence_loss = torch.sum(torch.abs(confidence-confidence_gt.detach())) / (torch.sum(hair_mask) + 1e-5)

        return orient_loss, confidence_loss, orientation_fake


if __name__ == '__main__':
    image_path = '/home/nas3_userM/chaeyeonchung/Barbershop_0213/input/face/16.png'
    image = Image.open(image_path)
    orient_loss = L1OLoss()
    orient_loss, confidence_loss = orient_loss(image, image, image)