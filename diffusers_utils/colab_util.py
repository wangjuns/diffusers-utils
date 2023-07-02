from typing import Dict
from PIL.Image import Image
import pprint
from diffusers import StableDiffusionPipeline

from diffusers_utils.civitai import resovler_schduler, to_diffusers_params, to_latents_params
from diffusers_utils.latent_util import generate_latents

pp = pprint.PrettyPrinter(indent=4)

def run_predict(pipe: StableDiffusionPipeline, prompt:Dict[str, str]) -> Image:
    pp.pprint(prompt)

    scheduler = resovler_schduler(prompt.get('Sampler', None))
    if scheduler is not None:
        pipe.scheduler = scheduler.from_config(pipe.scheduler.config)


    (seed, latents) = generate_latents(pipe, 'cpu', **to_latents_params(prompt))
    image = pipe(**to_diffusers_params(prompt), latents=latents).images[0]
    return image
