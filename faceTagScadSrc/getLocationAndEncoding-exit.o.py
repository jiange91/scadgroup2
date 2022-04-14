#@ type: compute
#@ corunning:
#@  getLocationAndEncoding-worker-0-mem:
#@     trans: getLocationAndEncoding-worker-0-mem
#@     type: mem
#@  getLocationAndEncoding-worker-1-mem:
#@     trans: getLocationAndEncoding-worker-1-mem
#@     type: mem
#@  getLocationAndEncoding-worker-2-mem:
#@     trans: getLocationAndEncoding-worker-2-mem
#@     type: mem
#@  getLocationAndEncoding-worker-3-mem:
#@     trans: getLocationAndEncoding-worker-3-mem
#@     type: mem
#@  getLocationAndEncoding-worker-4-mem:
#@     trans: getLocationAndEncoding-worker-4-mem
#@     type: mem
#@  getLocationAndEncoding-worker-5-mem:
#@     trans: getLocationAndEncoding-worker-5-mem
#@     type: mem
#@  getLocationAndEncoding-worker-6-mem:
#@     trans: getLocationAndEncoding-worker-6-mem
#@     type: mem
#@  getLocationAndEncoding-worker-7-mem:
#@     trans: getLocationAndEncoding-worker-7-mem
#@     type: mem
#@  getLocationAndEncoding-worker-8-mem:
#@     trans: getLocationAndEncoding-worker-8-mem
#@     type: mem
#@  getLocationAndEncoding-worker-9-mem:
#@     trans: getLocationAndEncoding-worker-9-mem
#@     type: mem
#@  mem:
#@     trans: mem
#@     type: mem
#@ parents:
#@   - getLocationAndEncoding-worker-0
#@   - getLocationAndEncoding-worker-1
#@   - getLocationAndEncoding-worker-2
#@   - getLocationAndEncoding-worker-3
#@   - getLocationAndEncoding-worker-4
#@   - getLocationAndEncoding-worker-5
#@   - getLocationAndEncoding-worker-6
#@   - getLocationAndEncoding-worker-7
#@   - getLocationAndEncoding-worker-8
#@   - getLocationAndEncoding-worker-9
#@   - getLocationAndEncoding
#@ dependents:
#@   - tagNameOnFrames

from disagg import *
from obj_pool import *
import base64
import copy
import math
from typing import *


def main(params, action):
    context_dict_in_b64 = params['getLocationAndEncoding'][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = serial_context.loads(context_dict_in_byte)
    objPool: ObjPool = context_dict['objPool']
    objPool.registerTrans(actionLib=action)
    localPool = {}
    workerPools = []
    for p in range(10):
        context_dict_in_b64 = params[f'getLocationAndEncoding-worker-{p}'][0][
            'meta']
        context_dict_in_byte = base64.b64decode(context_dict_in_b64)
        context_dict = serial_context.loads(context_dict_in_byte)
        workerPools.append(context_dict[
            f'getLocationAndEncoding-worker-{p}-pool'])
        workerPools[p].registerTrans(actionLib=action)
    uploads = {'tagNameOnFrames': [('face_encodingsPF', False, True), (
        'face_locationsPF', False, True)]}
    selectedUploads = list(uploads.values())[0]
    ustkMap = {'face_locationsPF': True, 'face_encodingsPF': True}
    curOffsets = {vname: (0) for vname, b in ustkMap.items() if b}
    for vname, isLocal, isRemote in selectedUploads:
        needUstk = ustkMap[vname]
        for i in range(10):
            obj = workerPools[i].materialize(vname)
            if not needUstk:
                objPool.upload_remote(obj, f'{vname}-idx{i}')
            else:
                ofst = curOffsets[vname]
                for l in range(len(obj)):
                    objPool.upload_remote(obj[l], f'{vname}-idx{l + ofst}')
                curOffsets[vname] += len(obj)
        if not needUstk:
            objPool.upload_remote(('List', 10), vname)
        else:
            objPool.upload_remote(('List', curOffsets[vname]), vname)
    context_dict = {}
    context_dict['objPool'] = objPool
    context_dict_in_byte = serial_context.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode('ascii')}
