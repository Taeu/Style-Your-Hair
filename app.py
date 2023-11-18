from pathlib import Path
import sys
parent = Path(__file__).parents
path_root = parent[0]
sys.path.append(str(path_root))

import os
from PIL import Image
import subprocess
import gradio as gr
import shutil
import sys
from utils.settings import ARGS
from main import main
from time import time


def prediction(img1, img2):
    ARGS.input_dir = os.path.join(path_root, ARGS.input_dir)
    ARGS.output_dir = os.path.join(path_root, ARGS.output_dir)
    ARGS.embedding_dir = os.path.join(path_root, ARGS.embedding_dir)
    
    if os.path.isdir(ARGS.input_dir):
        shutil.rmtree(ARGS.input_dir)
    if os.path.isdir(ARGS.output_dir):
        shutil.rmtree(ARGS.output_dir)
    os.makedirs(ARGS.input_dir, exist_ok=True)
    img1_path = os.path.join(ARGS.input_dir, "01.png")
    img2_path = os.path.join(ARGS.input_dir, "02.png")
    img1.convert("RGB").save(img1_path)
    img2.convert("RGB").save(img2_path)
    ARGS.im_path1 = img1_path
    ARGS.im_path2 = img2_path
    output = main(ARGS)
    return output
    # return Image.open(ARGS.im_path2)
  



css = '''
#image_upload{min-height:256}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 800px}
'''
with gr.Blocks(css=css) as demo:
    gr.Markdown(
    """
    # Generate your Hair Style
    """)
    with gr.Row():
        generate_btn = gr.Button("Generate").style(full_width=True)
    with gr.Row():
        input_img1 = gr.Image(label="Source Image", elem_id="image_upload1",type='pil').style(width=256, height=256)
        input_img2 = gr.Image(label="Target Image", elem_id="image_upload2",type='pil').style(width=256, height=256)
    
    output = gr.Image(label="Image", elem_id="output_image",type='pil').style(width=1024, height=1024)
    
    generate_btn.click(fn=prediction, inputs=[input_img1, input_img2], outputs=output, api_name="Generative you WIG")

demo.launch(share=False)
