import os
import math
import argparse
import numpy as np
import torch
import safetensors.torch as sf
import db_examples

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file

# Constants
sd15_name = 'stablediffusionapi/realistic-vision-v51'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load necessary models
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# Modify UNet
with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward

def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)

unet.forward = hooked_unet_forward

# Load model weights
model_path = './models/iclight_sd15_fc.safetensors'
if not os.path.exists(model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged

# Move models to the appropriate device
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# Set up attention processors
unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Define schedulers
ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Define pipelines
t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None
)

# Define BGSource Enum
class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"

# Helper functions remain unchanged, e.g., encode_prompt_pair, numpy2pytorch, pytorch2numpy, resize_and_center_crop

@torch.inference_mode()
def process_relight(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source):
    input_fg, matting = run_rmbg(input_fg)
    results = process(input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source)
    return input_fg, results

# CLI argument parsing
def main():
    parser = argparse.ArgumentParser(description="IC-Light: CLI version for relighting with foreground conditioning.")
    parser.add_argument('--input_fg', required=True, help="Path to the input foreground image.")
    parser.add_argument('--prompt', required=True, help="Text prompt for the model.")
    parser.add_argument('--bg_source', choices=[e.value for e in BGSource], default=BGSource.NONE.value, help="Background source for relighting.")
    parser.add_argument('--image_width', type=int, default=512, help="Width of the output image.")
    parser.add_argument('--image_height', type=int, default=640, help="Height of the output image.")
    parser.add_argument('--num_samples', type=int, default=1, help="Number of samples to generate.")
    parser.add_argument('--seed', type=int, default=12345, help="Seed for random number generation.")
    parser.add_argument('--steps', type=int, default=25, help="Number of diffusion steps.")
    parser.add_argument('--cfg', type=float, default=2.0, help="CFG scale for model guidance.")
    parser.add_argument('--highres_scale', type=float, default=1.5, help="Scale for high-res generation.")
    parser.add_argument('--highres_denoise', type=float, default=0.5, help="Denoise strength for high-res.")
    parser.add_argument('--lowres_denoise', type=float, default=0.9, help="Denoise strength for low-res.")
    parser.add_argument('--a_prompt', default='best quality', help="Additional prompt details.")
    parser.add_argument('--n_prompt', default='lowres, bad anatomy, bad hands, cropped, worst quality', help="Negative prompt details.")
    
    args = parser.parse_args()
    
    input_fg = np.array(Image.open(args.input_fg))
    bg_source = BGSource(args.bg_source)
    input_fg, results = process_relight(input_fg, args.prompt, args.image_width, args.image_height, args.num_samples, args.seed, args.steps, args.a_prompt, args.n_prompt, args.cfg, args.highres_scale, args.highres_denoise, args.lowres_denoise, bg_source)
    
    for i, result in enumerate(results):
        output_path = os.path.join(output_dir, f"output_{i}.png")
        Image.fromarray(result).save(output_path)
        print(f"Saved output to {output_path}")

if __name__ == "__main__":
    main()
