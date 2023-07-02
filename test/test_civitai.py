import unittest
from PIL import Image
import cv2

from diffusers_utils import civitai

# https://meik2333.com/posts/unit-testing-in-python/


class TestCivitai(unittest.TestCase):
    prompt = '''
(finely detailed beautiful face), (finely detailed beautiful widening eyes:1.1), (8k eyes), (8k pupils),(photorealistic:1.4), masterpiece, ultra quality, insanely detailed, (ultra high res), absurdres,raw,(standing, full body:1.2), girl, (18 yo:1.2), (_ hanfu,ming style),(tulle,translucent, see-through),stifled laugh, (pureerosface ,ulzzang-6500:0.5), Submissive,Innocent, seductive face,kawaii, (red hanfu:1.5),tang style,(full body:1.2), <lora:hanfu_v30:0.5>
Negative prompt: Unspeakable-Horrors-Composition-4v,(EasyNegative:1),(bad_prompt_version2:0.8), bad-hands-5, (low quality, worst quality, normal quality:1.7), sketch, illustration, (monotone:1.2), (monochrome:1.2), (grayscale:1.4), watermark,muscular,
ENSD: 31337, Size: 600x800, Seed: 1294305483, Model: braBeautifulRealistic_brav5, Steps: 10, Sampler: DPM++ SDE Karras, CFG scale: 8, Clip skip: 2, Model hash: ac68270450, Hires steps: 15, Hires upscale: 2, Hires upscaler: R-ESRGAN 4x+, Denoising strength: 0.3
'''

    def test_parse_civitai_prompt(self):
        result = civitai.parse_civitai_prompt(self.prompt)

        self.assertIn('prompt', result)

    


if __name__ == '__main__':
    unittest.main()
