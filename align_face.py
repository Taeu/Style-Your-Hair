import dlib
from pathlib import Path
import argparse
import torchvision
from utils.shape_predictor import align_face
import PIL


def preprocessing_align_face(unprocessed_dir, output_dir, output_size=1024, seed=None, cache_dir='cache', inter_method='bicubic'):
    print("preprocessing ......")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)

    print("Downloading Shape Predictor")
    # f=open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=cache_dir, return_path=True)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    for im in Path(unprocessed_dir).glob("*.*"):
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
                face.save(Path(output_dir) / (im.stem+f"_{i}.png"))
            else:
                face.save(Path(output_dir) / (im.stem + f".png"))
    
    print("finish preprocessing works.... next!!!!!!")
