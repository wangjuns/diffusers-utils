from PIL import Image
import cv2
import numpy


def image_grid(imgs: list[Image.Image], rows: int, cols: int) -> Image.Image:
    # assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def crop_img(img, target_size: tuple[int, int] = (512, 512)):
    '''
    img: str | Image.Image
    '''
    if isinstance(img, str):
        img = cv2.imread(img)  # 读取图片
    else:
        img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)

    (height, width, _) = img.shape

    t_img_len = height if height < width else width

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # 加载人脸检测模型
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 检测人脸
    if len(faces) == 0:
        print(f"no face detected in image {img}")
        return
    (x, y, w, h) = faces[0]
    if height < width:
        y = 0
        h = height
        x = max(0, (x + w//2) - height // 2)
        w = height
    else:
        x = 0
        w = width
        y = max(0, (y + h // 2) - width // 2)
        h = width

    crop_img = img[y:y+h, x:x+w]  # 裁剪人脸区域
    crop_img = cv2.resize(crop_img, target_size)  # 将人脸缩放为正方形
    return crop_img
