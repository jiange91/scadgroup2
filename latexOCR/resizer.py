import numpy as np
from PIL import Image
import cv2
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
import torch

def minmax_size(img, max_dimensions=None, min_dimensions=None):
    if max_dimensions is not None:
        ratios = [a/b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size)//max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        if any([s < min_dimensions[i] for i, s in enumerate(img.size)]):
            padded_im = Image.new('L', min_dimensions, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img

def pad(img: Image, divable=32):
    """Pad an Image to the next full divisible value of `divable`. Also normalizes the image and invert if needed.

    Args:
        img (PIL.Image): input image
        divable (int, optional): . Defaults to 32.

    Returns:
        PIL.Image
    """
    data = np.array(img.convert('LA'))
    data = (data-data.min())/(data.max()-data.min())*255
    if data[..., 0].mean() > 128:
        gray = 255*(data[..., 0] < 128).astype(np.uint8)  # To invert the text to white
    else:
        gray = 255*(data[..., 0] > 128).astype(np.uint8)
        data[..., 0] = 255-data[..., 0]

    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = data[b:b+h, a:a+w]
    if rect[..., -1].var() == 0:
        im = Image.fromarray((rect[..., 0]).astype(np.uint8)).convert('L')
    else:
        im = Image.fromarray((255-rect[..., -1]).astype(np.uint8)).convert('L')
    dims = []
    for x in [w, h]:
        div, mod = divmod(x, divable)
        dims.append(divable*(div + (1 if mod > 0 else 0)))
    padded = Image.new('L', dims, 255)
    padded.paste(im, im.getbbox())
    return padded

test_transform = alb.Compose(
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        # alb.Sharpen()
        ToTensorV2(),
    ]
)

def resizer(*, img, args):
    img = minmax_size(pad(img), args.max_dimensions, args.min_dimensions)
    image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=max(args.max_dimensions)//32, global_pool='avg', in_chans=1, drop_rate=.05,
                                 preact=True, stem_type='same', conv_layer=StdConv2dSame).to(args.device)
    image_resizer.load_state_dict(torch.load('/Users/zijian/Desktop/ucsd/291_Vir/project/playaround/latexOCR/checkpoints/image_resizer.pth', map_location=args.device))
    image_resizer.eval()
    with torch.no_grad():
        input_image = img.convert('RGB').copy()
        r, w = 1, input_image.size[0]
        for _ in range(10):
            img = pad(minmax_size(input_image.resize((w, int(input_image.size[1]*r)), Image.BILINEAR if r > 1 else Image.LANCZOS), args.max_dimensions, args.min_dimensions))
            t = test_transform(image=np.array(img.convert('RGB')))['image'][:1].unsqueeze(0)
            w = (image_resizer(t.to(args.device)).argmax(-1).item()+1)*32
            if (w == img.size[0]):
                break
            r = w/img.size[0]
    im = t.to(args.device)
    print(f'Tensor shape after resize: {im.shape}')
    return {'img': im}