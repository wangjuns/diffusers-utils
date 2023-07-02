import torch

def generate_latents(pipe, device, the_seed, size: tuple[int, int]|None = None):
    if size is None:
        width = pipe.unet.config.sample_size * pipe.vae_scale_factor
        height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    else:
        width, height = size

    # init generator
    generator = torch.Generator(device=device)

    # Get a new random seed, store it and use it as the generator state
    if the_seed is None:
        seed = generator.seed()
    else:
        seed = the_seed
    generator = generator.manual_seed(seed)

    image_latents = torch.randn(
        (1, pipe.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device
    )
    #image_latents = image_latents.half()
    return (seed, image_latents)