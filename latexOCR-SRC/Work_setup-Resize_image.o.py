#@ type: compute
#@ corunning:
#@  mem:
#@     trans: mem
#@     type: mem
#@ parents:
#@   - Dummy_authentication
#@ dependents:
#@   - ML_inference

from disagg import *
from obj_pool import *
import base64
import copy
import math
from typing import *
from munch import Munch
import yaml
from PIL import Image
import os


def workSetup(*, imgPath):
    img = Image.open(imgPath)
    output = {'img': img, 'imgName': os.path.splitext(os.path.basename(
        imgPath))[0]}
    with open(
        '/Users/zijian/Desktop/ucsd/291_Vir/project/playaround/latexOCR/settings/config.yaml'
        , 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        args = Munch(params)
        args.max_dimensions = [args.max_width, args.max_height]
        args.min_dimensions = [args.get('min_width', 32), args.get(
            'min_height', 32)]
        args.device = 'cpu'
        if 'decoder_args' not in args or args.decoder_args is None:
            args.decoder_args = {}
        output['args'] = args
    return output


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
        ratios = [(a / b) for a, b in zip(img.size, max_dimensions)]
        if any([(r > 1) for r in ratios]):
            size = np.array(img.size) // max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        if any([(s < min_dimensions[i]) for i, s in enumerate(img.size)]):
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
    data = (data - data.min()) / (data.max() - data.min()) * 255
    if data[..., 0].mean() > 128:
        gray = 255 * (data[..., 0] < 128).astype(np.uint8)
    else:
        gray = 255 * (data[..., 0] > 128).astype(np.uint8)
        data[..., 0] = 255 - data[..., 0]
    coords = cv2.findNonZero(gray)
    a, b, w, h = cv2.boundingRect(coords)
    rect = data[b:b + h, a:a + w]
    if rect[..., -1].var() == 0:
        im = Image.fromarray(rect[..., 0].astype(np.uint8)).convert('L')
    else:
        im = Image.fromarray((255 - rect[..., -1]).astype(np.uint8)).convert(
            'L')
    dims = []
    for x in [w, h]:
        div, mod = divmod(x, divable)
        dims.append(divable * (div + (1 if mod > 0 else 0)))
    padded = Image.new('L', dims, 255)
    padded.paste(im, im.getbbox())
    return padded


test_transform = alb.Compose([alb.ToGray(always_apply=True), alb.Normalize(
    (0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)), ToTensorV2()])


def resizer(*, img, args):
    img = minmax_size(pad(img), args.max_dimensions, args.min_dimensions)
    image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=max(args.
        max_dimensions) // 32, global_pool='avg', in_chans=1, drop_rate=
        0.05, preact=True, stem_type='same', conv_layer=StdConv2dSame).to(args
        .device)
    image_resizer.load_state_dict(torch.load(
        '/Users/zijian/Desktop/ucsd/291_Vir/project/playaround/latexOCR/checkpoints/image_resizer.pth'
        , map_location=args.device))
    image_resizer.eval()
    with torch.no_grad():
        input_image = img.convert('RGB').copy()
        r, w = 1, input_image.size[0]
        for _ in range(10):
            img = pad(minmax_size(input_image.resize((w, int(input_image.
                size[1] * r)), Image.BILINEAR if r > 1 else Image.LANCZOS),
                args.max_dimensions, args.min_dimensions))
            t = test_transform(image=np.array(img.convert('RGB')))['image'][:1
                ].unsqueeze(0)
            w = (image_resizer(t.to(args.device)).argmax(-1).item() + 1) * 32
            if w == img.size[0]:
                break
            r = w / img.size[0]
    im = t.to(args.device)
    print(f'Tensor shape after resize: {im.shape}')
    return {'img': im}


def main(params, action):
    context_dict_in_b64 = params['Dummy_authentication'][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = serial_context.loads(context_dict_in_byte)
    objPool: ObjPool = context_dict['objPool']
    objPool.registerTrans(actionLib=action)
    localPool = {}
    uploads = {'ML_inference': [('imgName', False, True), ('args', False, 
        True), ('img', False, True)]}
    Work_setupPullR = {'imgPath'}
    Work_setupBeC = set()
    for name in Work_setupPullR:
        localPool[name] = objPool.materialize(name)
    Work_setupInMap = {'imgPath'}
    Work_setupInDict = {}
    for inName in Work_setupInMap:
        if inName in Work_setupBeC:
            Work_setupInDict[inName] = copy.deepcopy(localPool[inName])
        else:
            Work_setupInDict[inName] = localPool[inName]
    Work_setupOutDict = workSetup(**Work_setupInDict)
    localPool.update(Work_setupOutDict)
    Resize_imagePullR = set()
    Resize_imageBeC = set()
    for name in Resize_imagePullR:
        localPool[name] = objPool.materialize(name)
    Resize_imageInMap = {'args', 'img'}
    Resize_imageInDict = {}
    for inName in Resize_imageInMap:
        if inName in Resize_imageBeC:
            Resize_imageInDict[inName] = copy.deepcopy(localPool[inName])
        else:
            Resize_imageInDict[inName] = localPool[inName]
    Resize_imageOutDict = resizer(**Resize_imageInDict)
    localPool.update(Resize_imageOutDict)
    selectedUploads = list(uploads.values())[0]
    nextLocal = set()
    for vname, isLocal, isRemote in selectedUploads:
        if isLocal:
            objPool.upload_local(localPool[vname], vname)
            nextLocal.add(vname)
        if isRemote:
            objPool.upload_remote(localPool[vname], vname)
    objPool.filterLocal(nextLocal)
    clearRemote = {'imgPath'}
    if clearRemote:
        for vname in clearRemote:
            objPool.deleteRemote(vname)
    context_dict = {}
    context_dict['objPool'] = objPool
    context_dict_in_byte = serial_context.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode('ascii')}
