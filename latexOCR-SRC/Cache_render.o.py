#@ type: compute
#@ corunning:
#@  mem:
#@     trans: mem
#@     type: mem
#@ parents:
#@   - ML_inference

from disagg import *
from obj_pool import *
import base64
import copy
import math
from typing import *
import os
import re
import sys
import io
import glob
import tempfile
import subprocess
from PIL import Image


class Latex:
    BASE = """
\\documentclass[varwidth]{standalone}
\\usepackage{fontspec,unicode-math}
\\usepackage[active,tightpage,displaymath,textmath]{preview}
\\begin{document}
\\thispagestyle{empty}
%s
\\end{document}
"""

    def __init__(self, math, dpi=250, font='Latin Modern Math'):
        """takes list of math code. `returns each element as PNG with DPI=`dpi`"""
        self.math = math
        self.dpi = dpi
        self.font = font

    def write(self, return_bytes=False):
        try:
            workdir = tempfile.gettempdir()
            fd, texfile = tempfile.mkstemp('.tex', 'eq', workdir, True)
            document = ''
            with os.fdopen(fd, 'w+') as f:
                document = self.BASE % '\n'.join(self.math)
                f.write(document)
                print('Generated tex content: ')
                print(document)
            png = self.convert_file(texfile, workdir, return_bytes=return_bytes
                )
            return png
        finally:
            if os.path.exists(texfile):
                try:
                    os.remove(texfile)
                except PermissionError:
                    pass

    def convert_file(self, infile, workdir, return_bytes=False):
        try:
            cmd = 'xelatex -halt-on-error -output-directory %s %s' % (workdir,
                infile)
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            sout, serr = p.communicate()
            if p.returncode != 0:
                raise Exception('latex error', serr, sout)
            pdffile = infile.replace('.tex', '.pdf')
            pngfile = os.path.join(workdir, infile.replace('.tex', '.png'))
            cmd = (
                'magick convert -density %i -colorspace gray %s -quality 90 %s'
                 % (self.dpi, pdffile, pngfile))
            p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            sout, serr = p.communicate()
            if p.returncode != 0:
                raise Exception('PDFpng error', serr, cmd, os.path.exists(
                    pdffile), os.path.exists(infile))
            if return_bytes:
                if len(self.math) > 1:
                    png = [open(pngfile.replace('.png', '') + '-%i.png' % i,
                        'rb').read() for i in range(len(self.math))]
                else:
                    png = [open(pngfile.replace('.png', '') + '.png', 'rb')
                        .read()]
                return png
            elif len(self.math) > 1:
                return [(pngfile.replace('.png', '') + '-%i.png' % i) for i in
                    range(len(self.math))]
            else:
                return pngfile.replace('.png', '') + '.png'
        finally:
            basefile = infile.replace('.tex', '')
            tempext = ['.aux', '.pdf', '.log']
            if return_bytes:
                ims = glob.glob(basefile + '*.png')
                for im in ims:
                    os.remove(im)
            for te in tempext:
                tempfile = basefile + te
                if os.path.exists(tempfile):
                    os.remove(tempfile)


__cache = {}


def tex2png(eq, **kwargs):
    if not eq in __cache:
        __cache[eq] = Latex(eq, **kwargs).write(return_bytes=True)
    return __cache[eq]


def tex2pil(tex, **kwargs):
    pngs = Latex(tex, **kwargs).write(return_bytes=True)
    images = [Image.open(io.BytesIO(d)) for d in pngs]
    return images


def persist(*, userID, imgName, pred):
    imgs = tex2pil([f'$${pred}$$'])
    outpath = f'user-{userID}'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    imgs[0].save(os.path.join(outpath, imgName) + '.png')
    return {'status':
        f"Saving render result to {os.path.join(outpath, imgName) + '.png'}"}


def main(params, action):
    context_dict_in_b64 = params['ML_inference'][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = serial_context.loads(context_dict_in_byte)
    objPool: ObjPool = context_dict['objPool']
    objPool.registerTrans(actionLib=action)
    localPool = {}
    uploads = {'FinalOutput': [('status', False, True)]}
    Cache_renderPullR = {'imgName', 'userID', 'pred'}
    Cache_renderBeC = set()
    for name in Cache_renderPullR:
        localPool[name] = objPool.materialize(name)
    Cache_renderInMap = {'imgName', 'userID', 'pred'}
    Cache_renderInDict = {}
    for inName in Cache_renderInMap:
        if inName in Cache_renderBeC:
            Cache_renderInDict[inName] = copy.deepcopy(localPool[inName])
        else:
            Cache_renderInDict[inName] = localPool[inName]
    Cache_renderOutDict = persist(**Cache_renderInDict)
    localPool.update(Cache_renderOutDict)
    selectedUploads = list(uploads.values())[0]
    nextLocal = set()
    for vname, isLocal, isRemote in selectedUploads:
        if isLocal:
            objPool.upload_local(localPool[vname], vname)
            nextLocal.add(vname)
        if isRemote:
            objPool.upload_remote(localPool[vname], vname)
    objPool.filterLocal(nextLocal)
    clearRemote = {'imgName', 'userID', 'pred', 'status'}
    if clearRemote:
        for vname in clearRemote:
            objPool.deleteRemote(vname)
    context_dict = {}
    context_dict['objPool'] = objPool
    context_dict_in_byte = serial_context.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode('ascii')}
