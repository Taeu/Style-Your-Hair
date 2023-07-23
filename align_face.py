import dlib
from pathlib import Path
import argparse
import torchvision
from utils.shape_predictor import align_face
import PIL


def preprocessing_align_face(unprocessed_dir, output_dir, output_size=1024, seed=None, cache_dir='cache', inter_method='bicubic'):

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)

    print("Downloading Shape Predictor")
    # f=open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=cache_dir, return_path=True)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    for im in Path(args.unprocessed_dir).glob("*.*"):
        faces = align_face(str(im), predictor)

        for i, face in enumerate(faces):
            if output_size:
                factor = 1024//output_size
                assert output_size*factor == 1024
                face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
                face_tensor_lr = face_tensor[0].cpu().detach().clamp(0, 1)
                face = torchvision.transforms.ToPILImage()(face_tensor_lr)
                if factor != 1:
                    face = face.resize((output_size, output_size), PIL.Image.LANCZOS)
            if len(faces) > 1:
                face.save(Path(args.output_dir) / (im.stem+f"_{i}.png"))
            else:
                face.save(Path(args.output_dir) / (im.stem + f".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocessing_align_face')

    parser.add_argument('-unprocessed_dir', type=str, default='unprocessed', help='directory with unprocessed images')
    parser.add_argument('-output_dir', type=str, default='ffhq_image', help='output directory')

    parser.add_argument('-output_size', type=int, default=1024,
                        help='size to downscale the input images to, must be power of 2')
    parser.add_argument('-seed', type=int, help='manual seed to use')
    parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')

    ###############
    parser.add_argument('-inter_method', type=str, default='bicubic')

    args = parser.parse_args()
    preprocessing_align_face(args.unprocessed_dir, args.output_dir, args.output_size, args.seed, args.cache_dir, args.inter_method)