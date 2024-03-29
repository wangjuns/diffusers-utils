from typing import Any, Dict
from diffusers import DDIMScheduler,DPMSolverMultistepScheduler


def parse_civitai_prompt(text: str) -> Dict[str, str]:
    segments = text.split('Negative prompt:')

    prompt = {}
    prompt['prompt'] = segments[0].strip()

    segments = segments[1].split('\n')
    prompt['negative_prompt'] = segments[0].strip()

    segments = segments[1].split(',')

    for i in range(0, len(segments)):
        segment = segments[i]
        kv = segment.split(':')
        assert len(kv) == 2, f"worng format {segment}"
        k, v = kv
        prompt[k.strip()] = v.strip()
    return prompt


def to_diffusers_params(prompt:Dict[str, str]) -> Dict[str, Any]:
    step = prompt.get('Steps', '50')
    step = int(step)

    guidance_scale = prompt.get('CFG scale', '7')
    guidance_scale = float(guidance_scale)

    negative_prompt=prompt.get('negative_prompt', None)
    
    result =  {
        "prompt" : prompt['prompt'],
        "num_inference_steps" : step,
        "guidance_scale" : guidance_scale,
        "negative_prompt" : negative_prompt
    }
    
    if 'Size' in prompt:
        size = prompt['Size'].split('x')
        assert len(size) == 2, f"wrong size format: {prompt['Size']}"
        result['width'] = size[0]
        result['height'] = size[1]
        
    return result
        

def to_latents_params(prompt:Dict[str, str]) -> Dict[str, Any]:
    seed = prompt.get('Seed', None)
    if seed is not None:
        seed = int(seed)

    image_size = prompt.get('Size', None)
    if image_size is not None:
        image_size = (int(x) for x in image_size.split('x'))
    return {
        "the_seed" : seed,
        "size" : image_size
    }
    

def resovler_schduler(name:str):
    if name is None:
        return None

    if name == 'DPM++ SDE Karras' or name == 'DPM++ 2M Karras' or name == 'DPM++ 2S a Karras':
        return DPMSolverMultistepScheduler

    print(f"Not supported sampler: {name}. Using default")
    return None