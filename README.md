# Style-Your-Hair
Official Pytorch implementation of "Style Your Hair: Latent Optimization for Pose-Invariant Hairstyle Transfer via Local-Style-Aware Hair Alignment (ECCV 2022)"

![teaser](docs/assets/teaser.png)
![qualitative result](docs/assets/sup-qualitative.png)
> **Style Your Hair: Latent Optimization for Pose-Invariant Hairstyle Transfer via Local-Style-Aware Hair Alignment**<br/>
[Taewoo Kim*](https://github.com/Taeu),
[Chaeyeon Chung*](https://github.com/ChennyTech),
[Yoonseo Kim*](https://github.com/myoons),
[Sunghyun Park](https://psh01087.github.io/),
[Kangyeol Kim](https://kangyeolk.github.io/), and 
[Jaegul Choo](https://sites.google.com/site/jaegulchoo/)<br/>
`*` indicates equal contributions.

> [arXiv](https://arxiv.org/abs/2208.07765) | [BibTeX](#bibtex) |


> **Abstract** Editing hairstyle is unique and challenging due to the complexity and delicacy of hairstyle.
Although recent approaches significantly improved the hair details, this is achieved under the assumption that a target hair and a source image are aligned.
HairFIT, a pose-invariant hairstyle transfer model, alleviates this assumption, yet it still shows unsatisfactory quality in preserving delicate hair textures.
To solve these limitations, we propose a high-performing pose-invariant hairstyle transfer model equipped with a latent optimization and a newly presented local-style-matching loss.
In the StyleGAN2 latent space, we first explore a pose-aligned latent code of a target hair with the detailed textures preserved based on local-style-matching.
Then, our model inpaints the occlusions of the source considering the aligned target hair and blends both images to produce a final output.
The experimental results demonstrate that our model has strengths in transferring a hairstyle under higher pose differences and preserving local hairstyle textures.


## Description
Official Implementation of Style Your Hair. KEEP UPDATING! Please Git Pull the latest version.


## Installation
- Clone the repository:
``` 
git clone https://github.com/Taeu/Style-Your-Hair.git
cd Style-Your-Hair
```
- Install dependencies:
```
conda create -n {env_name} python=3.7.9
conda activate {env_name}
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install face_alignment face-recognition gdown ipython matplotlib
```


## Download example images
Please download the [example images](https://drive.google.com/drive/folders/1RxzbNcKb3bPDKccyo300YXCJ8EvZSaIL?usp=sharing).
And put the images in `./ffhq_image/` folder.

## Getting Started  

Produce the results:
```
python main.py --input_dir ./ffhq_image/ --im_path1 source.png --im_path2 target.png \
    --output_dir ./style_your_hair_output/ \
    --warp_loss_with_prev_list delta_w style_hair_slic_large \
    --save_all --version final --flip_check
```


## Acknowledgments
This code borrows heavily from [Barbershop](https://github.com/ZPdesu/Barbershop).

## BibTeX

```
@article{kim2022style,
  title={Style Your Hair: Latent Optimization for Pose-Invariant Hairstyle Transfer via Local-Style-Aware Hair Alignment},
  author={Kim, Taewoo and Chung, Chaeyeon and Kim, Yoonseo and Park, Sunghyun and Kim, Kangyeol and Choo, Jaegul},
  journal={arXiv preprint arXiv:2208.07765},
  year={2022}
}
```
