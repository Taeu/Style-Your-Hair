import argparse


parser = argparse.ArgumentParser(description='Style Your Hair')
parser.add_argument('--flip_check', default=True, help='image2 might be flipped')
# warping and alignment
parser.add_argument('--warp_front_part', default=True,help='optimize warped_trg img from W+ space and only optimized [:6] part')
parser.add_argument('--warped_seg', default=True, help='create aligned mask from warped seg')
parser.add_argument('--align_src_first', default=True, help='align src with trg mask before blending')
parser.add_argument('--optimize_warped_trg_mask', default=True, help='optimize warped_trg_mask')
parser.add_argument('--mean_seg', default=True, help='use mean seg when alignment')
parser.add_argument('--kp_type', type=str, default='3D', help='kp_type')
parser.add_argument('--kp_loss', default=True, help='use keypoint loss when alignment')
parser.add_argument('--kp_loss_lambda', type=float, default=1000, help='kp_loss_lambda')
# blending
parser.add_argument('--blend_with_gram', default=True, help='add gram matrix loss in blending step')
parser.add_argument('--blend_with_align', default=True, help='optimization of alignment process with blending')
# hair related loss
parser.add_argument('--warp_loss_with_prev_list', nargs='+', help='select among delta_w, style_hair_slic_large',default="delta_w style_hair_slic_large")
parser.add_argument('--sp_hair_lambda', type=float, default=5.0, help='Super pixel hair loss when embedding')
# utils
parser.add_argument('--version', type=str, default='final', help='version name')
parser.add_argument('--save_all', default=True, help='save all output from whole process')
parser.add_argument('--embedding_dir', type=str, default='output', help='embedding vector directory')
# I/O arguments
parser.add_argument('--input_dir', type=str, default='image',help='The directory of the images to be inverted')
parser.add_argument('--output_dir', type=str, default='output',help='The directory to save the output images')
parser.add_argument('--im_path1', type=str, default='00001.png', help='Identity image')
parser.add_argument('--im_path2', type=str, default='02602.jpg', help='Structure image')
parser.add_argument('--sign', type=str, default='realistic', help='realistic or fidelity results')
parser.add_argument('--smooth', type=int, default=5, help='dilation and erosion parameter')
# StyleGAN2 setting
parser.add_argument('--size', type=int, default=1024)
parser.add_argument('--ckpt', type=str, default="pretrained_models/ffhq.pt")
parser.add_argument('--channel_multiplier', type=int, default=2)
parser.add_argument('--latent', type=int, default=512)
parser.add_argument('--n_mlp', type=int, default=8)
# Arguments
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--tile_latent', action='store_true', help='Whether to forcibly tile the same latent N times')
parser.add_argument('--opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate to use during optimization')
parser.add_argument('--lr_schedule', type=str, default='fixed', help='fixed, linear1cycledrop, linear1cycle')
parser.add_argument('--save_intermediate', action='store_true',help='Whether to store and save intermediate HR and LR images during optimization')
parser.add_argument('--save_interval', type=int, default=300, help='Latent checkpoint interval')
parser.add_argument('--verbose', action='store_true', help='Print loss information')
parser.add_argument('--seg_ckpt', type=str, default='pretrained_models/seg.pth')
# Embedding loss options
parser.add_argument('--percept_lambda', type=float, default=1.0, help='Perceptual loss multiplier factor')
parser.add_argument('--l2_lambda', type=float, default=1.0, help='L2 loss multiplier factor')
parser.add_argument('--p_norm_lambda', type=float, default=0.001, help='P-norm Regularizer multiplier factor')
parser.add_argument('--l_F_lambda', type=float, default=0.1, help='L_F loss multiplier factor')
parser.add_argument('--W_steps', type=int, default=1100, help='Number of W space optimization steps')
parser.add_argument('--FS_steps', type=int, default=250, help='Number of W space optimization steps')
# Alignment loss options
parser.add_argument('--ce_lambda', type=float, default=1.0, help='cross entropy loss multiplier factor')
parser.add_argument('--style_lambda', type=str, default=4e4, help='style loss multiplier factor')
parser.add_argument('--align_steps1', type=int, default=400, help='')
parser.add_argument('--align_steps2', type=int, default=100, help='')
parser.add_argument('--warp_steps', type=int, default=100, help='')
# Blend loss options
parser.add_argument('--face_lambda', type=float, default=1.0, help='')
parser.add_argument('--hair_lambda', type=str, default=1.0, help='')
parser.add_argument('--blend_steps', type=int, default=400, help='')

ARGS = parser.parse_args()