#@ type: compute
#@ corunning:
#@  mem:
#@     trans: mem
#@     type: mem
#@ parents:
#@   - loadVT
#@ dependents:
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
#@   - getLocationAndEncoding-exit

from disagg import *
from obj_pool import *
import base64
import copy
import math
from typing import *


def main(params, action):
    context_dict_in_b64 = params['loadVT'][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = serial_context.loads(context_dict_in_byte)
    objPool: ObjPool = context_dict['objPool']
    objPool.registerTrans(actionLib=action)
    localPool = {}
    stkTargets = {'frames'}
    iterName = list(stkTargets)[0]
    localPool[iterName] = objPool.materialize(iterName)
    iterLen = localPool[iterName][1]
    paral = 10
    workloads = math.ceil(iterLen / paral)
    i = 0
    p = 0
    context_dict = {}
    while i < iterLen:
        if i + workloads < iterLen:
            begin, end = i, i + workloads
        else:
            begin, end = i, iterLen
        targetsWithRange = {tname: (begin, end, 1) for tname in stkTargets}
        context_dict[f'getLocationAndEncoding-worker-{p}'] = targetsWithRange
        i += workloads
        p += 1
    context_dict['objPool'] = objPool
    context_dict['workloads'] = workloads
    context_dict_in_byte = serial_context.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode('ascii')}
