import unittest
from PIL import Image
import cv2

from diffusers_utils import images_util

# https://meik2333.com/posts/unit-testing-in-python/


class TestDict(unittest.TestCase):
    def test_grid(self):
        images = []
        for i in range(1, 3):
            img = Image.open(f'test/p{i}.jpeg')
            images.append(img)

        result = images_util.image_grid(images, 1, 2)

        # result.show()

    def test_crop(self):
        img = Image.open(f'test/temp_image1.jpeg')
        img = images_util.crop_img(img, (512, 512))
        
        assert img is not None, "detect failed"
        
        cv2.imwrite("test/test_corp_out.jpg", img)


if __name__ == '__main__':
    unittest.main()
