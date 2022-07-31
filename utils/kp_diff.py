import os
from skimage import io
from skimage.transform import resize
import time
import face_alignment
import numpy as np
from PIL import Image

def flip_check(im_path1, im_path2, device):


    im_name_2 = os.path.splitext(os.path.basename(im_path2))[0]

    kp_extractor = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=device)

    im1 = io.imread(im_path1)
    im2 = io.imread(im_path2)
    im2_flip = im2[:, ::-1]

    start_time = time.time()
    kp1 = kp_extractor.get_landmarks(im1)[0]
    kp2 = kp_extractor.get_landmarks(im2)[0]
    kp2_flip = kp_extractor.get_landmarks(im2_flip)[0]

    kp_diff = np.mean(np.abs(kp1 - kp2))
    kp_diff_flip = np.mean(np.abs(kp1 - kp2_flip))
    if kp_diff > kp_diff_flip:
        print(f'flip is better, kp_diff : {kp_diff} >  kp_diff_flip : {kp_diff_flip}')
        im2_flip = Image.fromarray(im2_flip)
        save_im_path2 = im_path2.replace(im_name_2, im_name_2 + '_flip')
        im2_flip.save(save_im_path2)
        return save_im_path2
    print(f'cal. kp. diff. time : {time.time() - start_time}')
    return im_path2

if __name__ == '__main__':
    input_dir =''
    output_dir = ''
    im_path1 = f'{input_dir}09172.png'
    im_path2 = f'{input_dir}21665.png'
    os.environ['CUDA_VISIBLE_DEIVCES'] = '5'
    flip_check(im_path1, im_path2)
