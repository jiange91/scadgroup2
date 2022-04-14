#@ type: compute
#@ corunning:
#@  tagNameOnFrames-worker-0-mem:
#@     trans: tagNameOnFrames-worker-0-mem
#@     type: mem
#@  tagNameOnFrames-worker-1-mem:
#@     trans: tagNameOnFrames-worker-1-mem
#@     type: mem
#@  mem:
#@     trans: mem
#@     type: mem
#@ parents:
#@   - tagNameOnFrames-worker-0
#@   - tagNameOnFrames-worker-1
#@   - tagNameOnFrames
#@ dependents:
#@   - BoxTargetName

from disagg import *
from obj_pool import *
import base64
import copy
import math
from typing import *


def main(params, action):
    context_dict_in_b64 = params['tagNameOnFrames'][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = serial_context.loads(context_dict_in_byte)
    objPool: ObjPool = context_dict['objPool']
    objPool.registerTrans(actionLib=action)
    localPool = {}
    workerPools = []
    for p in range(2):
        context_dict_in_b64 = params[f'tagNameOnFrames-worker-{p}'][0]['meta']
        context_dict_in_byte = base64.b64decode(context_dict_in_b64)
        context_dict = serial_context.loads(context_dict_in_byte)
        workerPools.append(context_dict[f'tagNameOnFrames-worker-{p}-pool'])
        workerPools[p].registerTrans(actionLib=action)
    uploads = {'BoxTargetName': [('face_namesPF', False, True)]}
    selectedUploads = list(uploads.values())[0]
    ustkMap = {'face_namesPF': True}
    curOffsets = {vname: (0) for vname, b in ustkMap.items() if b}
    for vname, isLocal, isRemote in selectedUploads:
        needUstk = ustkMap[vname]
        for i in range(2):
            obj = workerPools[i].materialize(vname)
            if not needUstk:
                objPool.upload_remote(obj, f'{vname}-idx{i}')
            else:
                ofst = curOffsets[vname]
                for l in range(len(obj)):
                    objPool.upload_remote(obj[l], f'{vname}-idx{l + ofst}')
                curOffsets[vname] += len(obj)
        if not needUstk:
            objPool.upload_remote(('List', 2), vname)
        else:
            objPool.upload_remote(('List', curOffsets[vname]), vname)
    context_dict = {}
    context_dict['objPool'] = objPool
    context_dict_in_byte = serial_context.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode('ascii')}
