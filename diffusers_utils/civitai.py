from typing import Any, Dict

def parse_civitai_prompt(text: str) -> Dict[str, str]:
    segments = text.split('Negative prompt:')

    prompt = {}
    prompt['prompt'] = segments[0].strip()

    segments = segments[1].split('\n')
    prompt['negtive_prompt'] = segments[0].strip()

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

    negative_prompt=prompt.get('negtive_prompt', None)

    return {
        "prompt" : prompt['prompt'],
        "num_inference_steps" : step,
        "guidance_scale" : guidance_scale,
        "negative_prompt" : negative_prompt
    }

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