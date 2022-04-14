#@ type: compute
#@ corunning:
#@  mem:
#@     trans: mem
#@     type: mem
#@ parents:
#@   - Work_setup-Resize_image
#@ dependents:
#@   - Cache_render

from disagg import *
from obj_pool import *
import base64
import copy
import math
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import *
from x_transformers.autoregressive_wrapper import *
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_hybrid import HybridEmbed
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
from transformers import PreTrainedTokenizerFast
from einops import rearrange, repeat
import re
operators = '|'.join(['arccos', 'arcsin', 'arctan', 'arg', 'cos', 'cosh',
    'cot', 'coth', 'csc', 'deg', 'det', 'dim', 'exp', 'gcd', 'hom', 'inf',
    'injlim', 'ker', 'lg', 'lim', 'liminf', 'limsup', 'ln', 'log', 'max',
    'min', 'Pr', 'projlim', 'sec', 'sin', 'sinh', 'sup', 'tan', 'tanh'])
ops = re.compile('\\\\operatorname{(%s)}' % operators)


class CustomARWrapper(AutoregressiveWrapper):

    def __init__(self, *args, **kwargs):
        super(CustomARWrapper, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=
        1.0, filter_logits_fn=top_k, filter_thres=0.9, **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)
        if num_dims == 1:
            start_tokens = start_tokens[(None), :]
        b, t = start_tokens.shape
        self.net.eval()
        out = start_tokens
        mask = kwargs.pop('mask', None)
        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.
                device)
        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]
            logits = self.net(x, mask=mask, **kwargs)[:, (-1), :]
            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
            elif filter_logits_fn is entmax:
                probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, dim=-1
                    )
            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)
            if eos_token is not None and (torch.cumsum(out == eos_token, 1)
                [:, (-1)] >= 1).all():
                break
        out = out[:, t:]
        if num_dims == 1:
            out = out.squeeze(0)
        self.net.train(was_training)
        return out


class CustomVisionTransformer(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, *args, **kwargs):
        super(CustomVisionTransformer, self).__init__(*args, img_size=
            img_size, patch_size=patch_size, **kwargs)
        self.height, self.width = img_size
        self.patch_size = patch_size

    def forward_features(self, x):
        B, c, h, w = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        h, w = h // self.patch_size, w // self.patch_size
        pos_emb_ind = repeat(torch.arange(h) * (self.width // self.
            patch_size - w), 'h -> (h w)', w=w) + torch.arange(h * w)
        pos_emb_ind = torch.cat((torch.zeros(1), pos_emb_ind + 1), dim=0).long(
            )
        x += self.pos_embed[:, (pos_emb_ind)]
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class Model(nn.Module):

    def __init__(self, encoder: CustomVisionTransformer, decoder:
        CustomARWrapper, args, temp: float=0.333):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bos_token = args.bos_token
        self.eos_token = args.eos_token
        self.max_seq_len = args.max_seq_len
        self.temperature = temp

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        device = x.device
        encoded = self.encoder(x.to(device))
        dec = self.decoder.generate(torch.LongTensor([self.bos_token] * len
            (x))[:, (None)].to(device), self.max_seq_len, eos_token=self.
            eos_token, context=encoded, temperature=self.temperature)
        return dec


def get_model(args, training=False):
    backbone = ResNetV2(layers=args.backbone_layers, num_classes=0,
        global_pool='', in_chans=args.channels, preact=False, stem_type=
        'same', conv_layer=StdConv2dSame)
    min_patch_size = 2 ** (len(args.backbone_layers) + 1)

    def embed_layer(**x):
        ps = x.pop('patch_size', min_patch_size)
        assert ps % min_patch_size == 0 and ps >= min_patch_size, 'patch_size needs to be multiple of %i with current backbone configuration' % min_patch_size
        return HybridEmbed(**x, patch_size=ps // min_patch_size, backbone=
            backbone)
    encoder = CustomVisionTransformer(img_size=(args.max_height, args.
        max_width), patch_size=args.patch_size, in_chans=args.channels,
        num_classes=0, embed_dim=args.dim, depth=args.encoder_depth,
        num_heads=args.heads, embed_layer=embed_layer).to(args.device)
    decoder = CustomARWrapper(TransformerWrapper(num_tokens=args.num_tokens,
        max_seq_len=args.max_seq_len, attn_layers=Decoder(dim=args.dim,
        depth=args.num_layers, heads=args.heads, **args.decoder_args)),
        pad_value=args.pad_token).to(args.device)
    model = Model(encoder, decoder, args)
    return model


def token2str(tokens, tokenizer):
    if len(tokens.shape) == 1:
        tokens = tokens[(None), :]
    dec = [tokenizer.decode(tok) for tok in tokens]
    return [''.join(detok.split(' ')).replace('Ġ', ' ').replace('[EOS]', ''
        ).replace('[BOS]', '').replace('[PAD]', '').strip() for detok in dec]


def post_process(s: str):
    """Remove unnecessary whitespace from LaTeX code.

    Args:
        s (str): Input string

    Returns:
        str: Processed image
    """
    text_reg = '(\\\\(operatorname|mathrm|text|mathbf)\\s?\\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\\W_^\\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub('(?!\\\\ )(%s)\\s+?(%s)' % (noletter, noletter),
            '\\1\\2', s)
        news = re.sub('(?!\\\\ )(%s)\\s+?(%s)' % (noletter, letter),
            '\\1\\2', news)
        news = re.sub('(%s)\\s+?(%s)' % (letter, noletter), '\\1\\2', news)
        if news == s:
            break
    return s


def predictor(*, img, args):
    model = get_model(args)
    model.load_state_dict(torch.load(
        '/Users/zijian/Desktop/ucsd/291_Vir/project/playaround/latexOCR/checkpoints/weights.pth'
        , map_location=args.device))
    encoder, decoder = model.encoder, model.decoder
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    with torch.no_grad():
        model.eval()
        device = args.device
        encoded = encoder(img.to(device))
        dec = decoder.generate(torch.LongTensor([args.bos_token])[:, (None)
            ].to(device), args.max_seq_len, eos_token=args.eos_token,
            context=encoded.detach(), temperature=args.get('temperature', 0.25)
            )
        pred = post_process(token2str(dec, tokenizer)[0])
    print('Most likely latex: ' + pred)
    return {'pred': pred}


def main(params, action):
    context_dict_in_b64 = params['Work_setup-Resize_image'][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = serial_context.loads(context_dict_in_byte)
    objPool: ObjPool = context_dict['objPool']
    objPool.registerTrans(actionLib=action)
    localPool = {}
    uploads = {'Cache_render': [('pred', False, True)]}
    ML_inferencePullR = {'args', 'img'}
    ML_inferenceBeC = set()
    for name in ML_inferencePullR:
        localPool[name] = objPool.materialize(name)
    ML_inferenceInMap = {'args', 'img'}
    ML_inferenceInDict = {}
    for inName in ML_inferenceInMap:
        if inName in ML_inferenceBeC:
            ML_inferenceInDict[inName] = copy.deepcopy(localPool[inName])
        else:
            ML_inferenceInDict[inName] = localPool[inName]
    ML_inferenceOutDict = predictor(**ML_inferenceInDict)
    localPool.update(ML_inferenceOutDict)
    selectedUploads = list(uploads.values())[0]
    nextLocal = set()
    for vname, isLocal, isRemote in selectedUploads:
        if isLocal:
            objPool.upload_local(localPool[vname], vname)
            nextLocal.add(vname)
        if isRemote:
            objPool.upload_remote(localPool[vname], vname)
    objPool.filterLocal(nextLocal)
    clearRemote = {'args', 'img'}
    if clearRemote:
        for vname in clearRemote:
            objPool.deleteRemote(vname)
    context_dict = {}
    context_dict['objPool'] = objPool
    context_dict_in_byte = serial_context.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode('ascii')}
