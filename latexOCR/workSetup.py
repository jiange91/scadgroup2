from munch import Munch
import yaml
from PIL import Image
import os

def workSetup(*, imgPath):
    img = Image.open(imgPath)
    output = {'img': img, 'imgName': os.path.splitext(os.path.basename(imgPath))[0]}
    with open('/Users/zijian/Desktop/ucsd/291_Vir/project/playaround/latexOCR/settings/config.yaml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader) 
        args = Munch(params)
        args.max_dimensions = [args.max_width, args.max_height]
        args.min_dimensions = [args.get('min_width', 32), args.get('min_height', 32)]
        args.device = 'cpu'
        if 'decoder_args' not in args or args.decoder_args is None:
            args.decoder_args = {}
        output['args'] = args
    return output