import unittest
from PIL import Image

from diffusers_utils import images_util

# https://meik2333.com/posts/unit-testing-in-python/


class TestDict(unittest.TestCase):
    def test_init(self):
        images = []
        for i in range(1, 3):
            img = Image.open(f'test/p{i}.jpeg')
            images.append(img)

        result = images_util.image_grid(images, 1, 2)
        # result.show()


if __name__ == '__main__':
    unittest.main()
