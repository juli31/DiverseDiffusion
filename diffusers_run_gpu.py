#@title Import required libraries
import argparse
import itertools
import math

import os
from contextlib import nullcontext
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import diffusers
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import bitsandbytes as bnb

from diffusers import LMSDiscreteScheduler

scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
 
# Option for using LMS scheduler
#LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
 

model_id = "runwayml/stable-diffusion-v1-5"

# hugging face access token, specific to the user
access_token = "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"
torch_device = "cuda"
num_images = 1000
min_dist = torch.ones((1)) * 182
min_dist_pooling = torch.ones((1)) * 3.1
num_iterations = 100



# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", use_auth_token=access_token)

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# make sure you're logged in with `huggingface-cli login`
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=access_token) 

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = diffusers.UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", use_auth_token=access_token)

vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device) 

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def sch_images(text_embeddings, scheduler, latents, guidance_scale, mod_name):
            m_path = experiment_name + mod_name 
            m_path_sm = experiment_name + mod_name.split('/')[0]
            
            if not os.path.exists(m_path_sm):
                os.mkdir(m_path_sm)
            
            m_path += '/'
            if not os.path.exists(m_path):
                os.mkdir(m_path)

            for t in tqdm(scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                # latent_model_input = torch.cat([latents] * 2)    
            
                latent_model_input = torch.cat([latents, latents], 0)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents

            with torch.no_grad():
                image = vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
        
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        
            images = (image * 255).round().astype("uint8")
            
            f_name = prompt + '_' + str(random.randint(1000000000000000, 9000000000000000)) + '.png'
            
            for im in images:
                img = Image.fromarray(im) 
                img.save(m_path + f_name)
                print(m_path + f_name)

def gen_image(prompt, num_images=5, height=512, width=512,
              weight=512, num_inference_steps = 100,
              guidance_scale = 7.5,
              batch_size = 1):

    text_embeddings_list = []
    r_num =  str(random.randint(10000000000, 90000000000))
    for i in range(num_images):
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")


        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8))
    
        latents = latents.to(torch_device)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma
        
        sch_images(text_embeddings, scheduler, latents, guidance_scale, 'f_mod/' + r_num )


def gen_image_big_dist_1(prompt, num_images=5, height=512, width=512,
                         weight=512, num_inference_steps = 100,
                         guidance_scale = 7.5,
                         batch_size = 1):
        
    old_latents = []

    text_embeddings_list = []
    r_num =  str(random.randint(10000000000, 90000000000))
    for i in range(num_images):
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")


        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]     
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        

        latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8))
        
        if len(old_latents) != 0:
                #r_n = random.randint(0, len(old_latents)-1)
                dist = torch.min(torch.stack([(latents - old_latents[r_n]).pow(2).sum().sqrt() for r_n in range(len(old_latents))]))

                while dist < min_dist:
                        latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8))
                        dist = torch.min(torch.stack([(latents - old_latents[r_n]).pow(2).sum().sqrt() for r_n in range(len(old_latents))]))

                        
        
        old_latents.append(latents)
        latents = latents.to(torch_device)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma

        text_embeddings_list.append(text_embeddings)
        
        
        sch_images(text_embeddings, scheduler, latents ,guidance_scale, 'f_mod_cap/' + r_num )



def gen_image_big_dist_2(prompt, num_images=5, height=512, width=512,
                            weight=512, num_inference_steps = 100,
                            guidance_scale = 7.5,
                            batch_size = 1):
        
    old_latents = []
    r_num =  str(random.randint(10000000000, 90000000000))
    for i in range(num_images):
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")


        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]     
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        
        latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8),)
        
        if len(old_latents) != 0:
                #r_n = random.randint(0, len(old_latents)-1)
                dist = torch.min(torch.stack([(latents - old_latents[r_n]).pow(2).sum().sqrt() for r_n in range(len(old_latents))]))
                max_dist = dist
                for i in range(num_iterations):
                        latents_new = torch.randn((batch_size, unet.in_channels, height // 8, width // 8),)
                        dist_new = torch.min(torch.stack(
                                [(latents_new - old_latents[r_n]).pow(2).sum().sqrt() for r_n in range(len(old_latents))]))
                        if dist_new > max_dist:
                                latents = latents_new
                                max_dist = dist_new
                                

                        
        
        old_latents.append(latents)
        latents = latents.to(torch_device)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma
        sch_images(text_embeddings, scheduler, latents ,guidance_scale, 'f_mod_max_real/' + r_num )
        
        
def gen_image_big_dist_3(prompt, num_images=5, height=512, width=512,
                            weight=512, num_inference_steps = 100,
                            guidance_scale = 7.5, 
                            batch_size = 1):
        
    old_latents = []
    m = torch.nn.AvgPool3d((1, 8, 8))
    r_num =  str(random.randint(10000000000, 90000000000))
    for i in range(num_images):
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")


        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]     
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        
        latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8),)
        latents_pooling = m(latents)
        
        if len(old_latents) != 0:
                #r_n = random.randint(0, len(old_latents)-1)
   
                dist = torch.min(torch.stack([(latents_pooling - old_latents[r_n]).pow(2).sum().sqrt() for r_n in range(len(old_latents))]))
                max_dist = dist
                for i in range(num_iterations):
                        latents_new = torch.randn((batch_size, unet.in_channels, height // 8, width // 8),)
                        latents_pooling_new = m(latents_new)
                        dist_new = torch.min(torch.stack(
                                [(latents_pooling_new - old_latents[r_n]).pow(2).sum().sqrt() for r_n in range(len(old_latents))]))
                        if dist_new > max_dist:
                                latents = latents_new
                                max_dist = dist_new
                                latents_pooling = latents_pooling_new
                                

                        
        
        old_latents.append(latents_pooling)
        latents = latents.to(torch_device)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma
        sch_images(text_embeddings, scheduler, latents ,guidance_scale, 'f_mod_max_pooling/' + r_num )

def gen_image_big_dist_4(prompt, num_images=5, height=512, width=512,
                            weight=512, num_inference_steps = 100,
                            guidance_scale = 7.5, 
                            batch_size = 1):
        
    old_latents = []
    m = torch.nn.AvgPool3d((1, 8, 8))
    r_num =  str(random.randint(10000000000, 90000000000))
    for i in range(num_images):
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")


        with torch.no_grad():
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]     
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        
        latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8), )
        latents_pooling = m(latents)
        
        if len(old_latents) != 0:
                #r_n = random.randint(0, len(old_latents)-1)
   
                dist = torch.min(torch.stack([(latents_pooling - old_latents[r_n]).pow(2).sum().sqrt() for r_n in range(len(old_latents))]))
                max_dist = dist

                while max_dist < min_dist_pooling:
                        latents_new = torch.randn((batch_size, unet.in_channels, height // 8, width // 8), )
                        latents_pooling_new = m(latents_new)
                        dist_new = torch.min(torch.stack(
                                [(latents_pooling_new - old_latents[r_n]).pow(2).sum().sqrt() for r_n in range(len(old_latents))]))
                        if dist_new > max_dist:
                                latents = latents_new
                                max_dist = dist_new
                                latents_pooling = latents_pooling_new
                                print("Distance:", max_dist)
                                

                        
        
        old_latents.append(latents_pooling)
        latents = latents.to(torch_device)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma
        sch_images(text_embeddings, scheduler, latents, guidance_scale, 'f_mod_cap_pooling/' + r_num )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="Prompt passed to the stable diffusion", required=True)
    parser.add_argument("--batch_size", help="Size of image batches for improving diversity", type=int, required=True)
    args = parser.parse_args()
            
    _batch_size = args.batch_size
    prompt = args.prompt
    prompt = prompt.replace('_', ' ')
    
    experiment_name = '/private/home/m/FACE/stable-diffusion/ee/' + prompt + '_mindistpooling_' + str(min_dist_pooling) + '_batch_' + str(_batch_size) + '/'  
    if not os.path.exists(experiment_name):
        os.mkdir(experiment_name)   

    pipe = diffusers.StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=access_token, revision="fp16", torch_dtype=torch.float16)
    
    num_batches = num_images // _batch_size
    
    for i in range(num_batches):
        f_mod_pooling_far = gen_image_big_dist_4(prompt, _batch_size)
        f_mod_pooling = gen_image_big_dist_3(prompt, _batch_size)
        f_mod_farthest = gen_image_big_dist_2(prompt, _batch_size)
        f_mod_far = gen_image_big_dist_1(prompt, _batch_size)
        f_mod = gen_image(prompt, _batch_size)

