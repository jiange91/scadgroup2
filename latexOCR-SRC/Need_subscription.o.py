#@ type: compute
#@ corunning:
#@  mem:
#@     trans: mem
#@     type: mem
#@ parents:
#@   - Dummy_authentication

from disagg import *
from obj_pool import *
import base64
import copy
import math
from typing import *
import time


def subscription(*, userID):
    print('Please buy me a coffee')
    time.sleep(2)
    return {'userID': userID}


def main(params, action):
    context_dict_in_b64 = params['Dummy_authentication'][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = serial_context.loads(context_dict_in_byte)
    objPool: ObjPool = context_dict['objPool']
    objPool.registerTrans(actionLib=action)
    localPool = {}
    uploads = {'FinalOutput': [('userID', False, True)]}
    Need_subscriptionPullR = {'userID'}
    Need_subscriptionBeC = set()
    for name in Need_subscriptionPullR:
        localPool[name] = objPool.materialize(name)
    Need_subscriptionInMap = {'userID'}
    Need_subscriptionInDict = {}
    for inName in Need_subscriptionInMap:
        if inName in Need_subscriptionBeC:
            Need_subscriptionInDict[inName] = copy.deepcopy(localPool[inName])
        else:
            Need_subscriptionInDict[inName] = localPool[inName]
    Need_subscriptionOutDict = subscription(**Need_subscriptionInDict)
    localPool.update(Need_subscriptionOutDict)
    selectedUploads = list(uploads.values())[0]
    nextLocal = set()
    for vname, isLocal, isRemote in selectedUploads:
        if isLocal:
            objPool.upload_local(localPool[vname], vname)
            nextLocal.add(vname)
        if isRemote:
            objPool.upload_remote(localPool[vname], vname)
    objPool.filterLocal(nextLocal)
    clearRemote = {'userID'}
    if clearRemote:
        for vname in clearRemote:
            objPool.deleteRemote(vname)
    context_dict = {}
    context_dict['objPool'] = objPool
    context_dict_in_byte = serial_context.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode('ascii')}
