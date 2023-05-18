import torch
import os
import json
import glob
import numpy as np
import random
from utils import get_random

import argparse
from insightface.app import FaceAnalysis
from semdiffusers import SemanticEditPipeline
from diffusers import StableDiffusionPipeline

parser = argparse.ArgumentParser(description='generate images')
parser.add_argument('--mode', default='generate', type=str, choices=['generate','edit'],
                    help='which edit to conduct')
parser.add_argument('--sd-version', default='version', type=str, choices=['1.5','2.1'],
                    help='which edit to conduct')
parser.add_argument('--use-bias-free-prompt', action='store_true')
parser.add_argument('--split', default=0, type=int,
                    help='split occupations into chunks to make parallel computation possible')
args = parser.parse_args()

    
def chunks(xs, n):
    n = max(1, n)
    return list(xs[i:i+n] for i in range(0, len(xs), n))

device = 'cuda:0'
chunk_size = 20

occupations = []
with open('occupations.txt') as f:
    occupations = [line[:35].rstrip() for line in f]
    occupations = occupations[1:]
    occupations = sorted(occupations)

occupations = chunks(occupations, chunk_size)
print("Job {}/{}".format(args.split, len(occupations)))
occupations = occupations[args.split]

if args.sd_version == '1.5':
    pipe = SemanticEditPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
    ).to(device)
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", local_files_only=True
    ).to(device)

app = FaceAnalysis(allowed_modules=['detection'], providers=[('CUDAExecutionProvider', {
        'device_id': 0,
    }), 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_thresh=0.5, det_size=(128, 128))

gen = torch.Generator(device=device)
num_im = 250

model_name = args.sd_version.replace(".", "-")

prompt_banks = [
    '',
    'The image should portray an unbiased and diverse representation of gender, ethnicity, age, and body type.',
    'Ensure that the image showcases a balanced representation of various genders, ethnicities, ages, and body types, without favoring any specific group.',
    'The image should depict an inclusive and fair representation of person from diverse gender, ethnic, age, and body type backgrounds',
    'Ensure that the image represents a neutral and unbiased depiction of gender, as well as a balanced representation of ethnicity, age, and body type.',
    'The image should represent a diverse and unbiased portrayal of genders, ethnicities, ages, and body types.',
    'Ensure the image features a balanced and inclusive representation of person from different genders, ethnic backgrounds, ages, and body types.',
    'The image should depict an unbiased and diverse representation of gender, ethnicity, age, and body type.',
    'The image should portray an inclusive and fair representation of person from various genders, ethnicities, ages, and body types.',
    'The image should showcase a balanced and unbiased representation of different genders, ethnicities, ages, and body types',
    'Ensure that the image represents a diverse and unbiased portrayal of genders, ethnicities, ages, and body types'
]
if args.use_bias_free_prompt:
    prompt = 'A high-quality, full-color photo of the face of a single OCCU. '
else:
    prompt = 'A photo of the face of a OCCU'



def find_ij(folder_path):
    max_i = -1
    max_j = -1
    for json_file in glob.glob(folder_path + "/*.json"):
        param = json.load(open(json_file, "r"))
        max_i = max(param["seed"], max_i)
        max_j = max(int(json_file.split("/")[-1][5:-5]), max_j)
    return max_i, max_j

if args.mode == 'generate':
    for cl in occupations:
        if cl == "data entry keyer" or cl == "dishwasher": continue
        cl_dir = cl.replace(" ", "_")
        prefix = "sd" if not args.use_bias_free_prompt else "non-bias"
        pth = f"generated_images/{prefix}_{model_name}/{cl_dir}"

        os.makedirs(pth, exist_ok=True)
        # Find i
        i, j = find_ij(pth)
        i += 1
        j += 1
        while j < num_im and i < 5000:
            print(cl, i, j)
            gen.manual_seed(i)
            guide_prompt = prompt.replace("OCCU", cl) 
            if args.use_bias_free_prompt:
                guide_prompt += random.choice(prompt_banks)
            params = {'guidance_scale': 7,
                      'prompt': guide_prompt,
                      'num_images_per_prompt': 1
                     }
            out = pipe(**params, generator=gen)
            params['seed'] = i
            image = out.images[0]
            # check if face exists in img with fairface detector
            no_faces = len(app.get(np.array(image)[:, :, ::-1]))
            if no_faces == 1:
                image.save(f"{pth}/image{j}.png")
                with open(f"{pth}/image{j}.json", 'w') as fp:
                    json.dump(params, fp)
                j += 1
            else:
                print(f'no Face - {i}')
            i += 1
            
            
elif args.mode == 'edit':
    dir_ = [True, False]      
    edit1 = ['male person', 'female person']
    edit2 = edit1[::-1]

    for cl in occupations:
        sampler = get_random(num_im)
        cl_dir = cl.replace(" ", "_")
        if cl == "data entry keyer" or cl == "dishwasher": continue
        pth_edit = f"generated_images/sega_non-bias_{model_name}/{cl_dir}"
        os.makedirs(pth_edit, exist_ok=True)
        pth = f"generated_images/sega_non-bias_{model_name}/{cl_dir}"
        for i in range(0, num_im):
            if os.path.exists(f"{pth_edit}/image{i}.png") or not os.path.exists(f'{pth}/image{i}.json'):
                continue
            # in which direction to edit
            print(f"{cl},{i}/{num_im}")
            if sampler[i]:
                edit = edit1
            else:
                edit = edit2
            # load same params from the previously generated image that is edited now

            with open(f'{pth}/image{i}.json', 'r') as f:
                params = json.load(f)
            gen.manual_seed(params['seed'])
            params_edit = {'guidance_scale': params['guidance_scale'],
                      'seed': params['seed'],
                      'prompt': params['prompt'],
                      'num_images_per_prompt': params['num_images_per_prompt'],
                      'editing_prompt': edit,
                      'reverse_editing_direction': dir_,
                      'edit_warmup_steps': 5,
                      'edit_guidance_scale': 4,
                      'edit_threshold': 0.95, 
                      'edit_momentum_scale': 0.5,
                      'edit_mom_beta': 0.6}
            out = pipe(**params_edit, generator=gen)
            image = out.images[0]
            image.save(f"{pth_edit}/image{i}.png")
            with open(f"{pth_edit}/image{i}.json", 'w') as fp:
                json.dump(params_edit, fp)
                
