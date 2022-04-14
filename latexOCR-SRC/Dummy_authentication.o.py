#@ type: compute
#@ corunning:
#@  mem:
#@     trans: mem
#@     type: mem
#@ parents:
#@   - None
#@ dependents:
#@   - Need_subscription
#@   - Work_setup-Resize_image

from disagg import *
from obj_pool import *
import base64
import copy
import math
from typing import *
import time


def auth(*, userID, imgPath):
    time.sleep(1)
    privilege = 2 * userID
    if privilege < 10:
        print('Not enough credits')
        return 0, {'imgPath': None, 'userID': userID}
    else:
        print('Proceeding')
        return 1, {'imgPath': imgPath, 'userID': userID}


def main(params, action):
    objPool = ObjPool(name='mem', memSize=536870912, pageSize=16384)
    objPool.registerTrans(actionLib=action)
    localPool = {'userID': 100, 'imgPath':
        '/Users/zijian/Desktop/ucsd/291_Vir/project/playaround/latexOCR/imgs/eq1.png'
        }
    uploads = {'Need_subscription': [('userID', False, True)],
        'Work_setup-Resize_image': [('imgPath', False, True), ('userID', 
        False, True)]}
    nexts = ['Need_subscription', 'Work_setup-Resize_image']
    Dummy_authenticationPullR = {}
    Dummy_authenticationBeC = set()
    for name in Dummy_authenticationPullR:
        localPool[name] = objPool.materialize(name)
    Dummy_authenticationInMap = {'userID', 'imgPath'}
    Dummy_authenticationInDict = {}
    for inName in Dummy_authenticationInMap:
        if inName in Dummy_authenticationBeC:
            Dummy_authenticationInDict[inName] = copy.deepcopy(localPool[
                inName])
        else:
            Dummy_authenticationInDict[inName] = localPool[inName]
    nid, Dummy_authenticationOutDict = auth(**Dummy_authenticationInDict)
    localPool.update(Dummy_authenticationOutDict)
    selectedUploads = uploads[nexts[nid]]
    nextLocal = set()
    for vname, isLocal, isRemote in selectedUploads:
        if isLocal:
            objPool.upload_local(localPool[vname], vname)
            nextLocal.add(vname)
        if isRemote:
            objPool.upload_remote(localPool[vname], vname)
    objPool.filterLocal(nextLocal)
    clearRemote = set()
    if clearRemote:
        for vname in clearRemote:
            objPool.deleteRemote(vname)
    context_dict = {}
    context_dict['objPool'] = objPool
    context_dict_in_byte = serial_context.dumps(context_dict)
    return {'meta': base64.b64encode(context_dict_in_byte).decode('ascii')}
