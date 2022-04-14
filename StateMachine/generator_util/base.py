from multiprocessing import pool
from ..primitives import *
from .. import profUtil
import ast
from ruamel.yaml import YAML
import copy
import astor
from ..console import KB, MB, ElementInDAG

def genObjPoolMeta(size, dagRoot: ElementInDAG):
    poolConfig = {'type': 'memory'}
    memSizeInMB = math.ceil(size / MB)
    baseSize = 512
    while baseSize < memSizeInMB:
        baseSize += 128
    print(baseSize)
    poolConfig['limits'] = {'mem': str(baseSize) + ' MB'}
    poolConfig['corunning'] = []
    stack = [dagRoot]
    while stack:
        cur = stack.pop()
        if cur.nType == 'Lambda' or cur.nType == 'SwitchFlow':
            poolConfig['corunning'].append(cur.name)
        elif cur.nType == 'Paral':
            poolConfig['corunning'].append(cur.name)
            for p in range(cur.paral):
                poolConfig['corunning'].append(cur.name + f'-worker-{p}')
            poolConfig['corunning'].append(cur.name + '-exit')
        stack += list(cur.children.values())
    return poolConfig, baseSize * MB

# Get meta data for compute element
def compMetaConfig(poolNames: List[str], parents: List[str], children: List[str]) -> str:
    metaComment = []
    metaComment.append('#@ type: compute')
    metaComment.append('#@ corunning:')
    for p in poolNames:
        metaComment.append(f'#@  {p}:')
        metaComment.append(f'#@     trans: {p}')
        metaComment.append('#@     type: mem')
    if parents:
        metaComment.append('#@ parents:')
        for p in parents:
            metaComment.append(f'#@   - {p}')
    if children:
        metaComment.append('#@ dependents:')
        for cname in children:
            metaComment.append(f'#@   - {cname}')
    return '\n'.join(metaComment) + '\n\n'

def baseImports():
    code = \
"""
from disagg import *
from obj_pool import *
import base64
import copy
import math
from typing import *
"""
    return ast.parse(code).body
    

def mainBlock() -> Tuple[ast.Module, List]:
    mainSkeletion = \
"""
def main(params, action):
    pass
"""
    mainDef = ast.parse(mainSkeletion)
    mainBody = mainDef.body[0].body
    return mainDef, mainBody

def setPoolFromPrev(prevName, poolName='objPool'):
    code = \
f"""
context_dict_in_b64 = params["{prevName}"][0]['meta']
context_dict_in_byte = base64.b64decode(context_dict_in_b64)
context_dict = serial_context.loads(context_dict_in_byte)
{poolName}: ObjPool = context_dict['objPool']
{poolName}.registerTrans(actionLib=action)
localPool = {{}}
"""
    return ast.parse(code).body

def setRootObjPool(name, memSize = 512 * MB, pageSize = 4 * KB, appInput=None):
    code = \
f"""
objPool = ObjPool(
    name="{name}",
    memSize= {memSize},
    pageSize= {pageSize} 
)
objPool.registerTrans(actionLib=action)
localPool = {appInput}
"""        
    return ast.parse(code).body

# def setInOutPool():
#     code = \
# """
# inPool = {{}}
# outPool = {{}}
# """
#     return ast.parse(code).body       

def getNodeInput(isRoot: bool, node: profUtil.Node, pullR: Set, beCareful: Set):
    pullR = {} if isRoot else pullR
    prepareInMap = \
f"""
{node.name}PullR = {pullR}
{node.name}BeC = {beCareful}
"""
    if node.stackerTargets:
        prepareInMap += \
f"""
{node.name}STKTargets = {node.stackerTargets}
{node.name}PullR, {node.name}STKTargets = stackedPullMap({node.name}PullR, {node.name}STKTargets, objPool)
"""        

    populateInPool = \
f"""
for name in {node.name}PullR:
    localPool[name] = objPool.materialize(name)
"""
    if node.stackerTargets:
        populateInPool += \
f"""
localPool, {node.name}Restore = stackedInPool(localPool, {node.name}STKTargets)
"""

    getInputForNode = \
f"""
{node.name}InMap = {node.inMap}
{node.name}InDict = {{}}
for inName in {node.name}InMap:
    if inName in {node.name}BeC:
        {node.name}InDict[inName] = copy.deepcopy(localPool[inName])
    else:
        {node.name}InDict[inName] = localPool[inName]
"""
    code = prepareInMap + populateInPool + getInputForNode
    return ast.parse(code).body

def updateOutput(node: profUtil.Node):
    code = ''
    if node.stackerTargets:
        code += \
f"""
localPool = restoreFromSTK(localPool, {node.name}Restore) 
"""
    if node.unstackerTargets:
        code += \
f"""
{node.name}USTKTargets = {node.unstackerTargets}
uploads = unstackedUploads({node.name}OutDict, uploads, {node.name}USTKTargets)
{node.name}OutDict = unstackedOutDict({node.name}OutDict, {node.name}USTKTargets)
"""
    code += \
f"""
localPool.update({node.name}OutDict)
"""
    return ast.parse(code).body

def baseUploads(ele: ElementInDAG):
    uploads = []
    for cname, objList in ele.uploads.items():
        uploads.append(f'"{cname}": {objList}')
    uploadsDict = '{' + ', '.join(uploads)+ '}'
    nexts = ''
    if ele.nType == 'SwitchFlow':
        nexts = 'nexts = ' + ele.nexts.__repr__()
    code = \
f"""
uploads = {uploadsDict}
{nexts}
"""
    return ast.parse(code).body

def UpdateLR(poolName='objPool'):
    code = \
f"""
nextLocal = set()
for vname, isLocal, isRemote in selectedUploads:
    if isLocal:
        {poolName}.upload_local(localPool[vname], vname)
        nextLocal.add(vname)
    if isRemote:
        {poolName}.upload_remote(localPool[vname], vname)
"""
    return ast.parse(code).body
    
def clearLR(curEle: ElementInDAG):
    code = \
f"""
objPool.filterLocal(nextLocal)
clearRemote = {curEle.clearRemote}
if clearRemote:
    for vname in clearRemote:
        objPool.deleteRemote(vname)
"""
    return ast.parse(code).body

def toDependents():
    code = \
"""
context_dict = {}
context_dict['objPool'] = objPool
context_dict_in_byte = serial_context.dumps(context_dict)
return {'meta': base64.b64encode(context_dict_in_byte).decode("ascii")}
"""
    return ast.parse(code).body

def stackedPullMap():
    code = \
"""
def stackedPullMap(inMap: Set, targets: Dict, objPool):
    if not targets:
        return inMap, targets
    else:
        stkMap = set()
        for name, (start, end, step) in targets.items():
            start = start or 0
            step = step or 1
            if not end:
                empiricalEnd = start
                while True:
                    curUstk = f'{name}-idx{empiricalEnd}'
                    if objPool.contains(curUstk):
                        stkMap.add(curUstk)
                    else:
                        break
                    empiricalEnd += step
                # update targets
                targets[name] = (start, empiricalEnd, step)
            else:
                for i in range(start, end, step):
                    curUstk = f'{name}-idx{i}'
                    stkMap.add(curUstk)
    return inMap.union(stkMap), targets
"""
    return ast.parse(code).body

def stackedInPool():
    code = \
"""
def stackedInPool(inPool: Dict, targets: Dict):
    if not targets:
        return inPool, None
    targetBackUp = {k: inPool[k] for k in targets.keys()}
    output: Dict[str, List] = {}
    for name, (start, end, step) in targets.items():
        output[name] = [inPool[f'{name}-idx{i}'] for i in range(start, end, step)]
    return {**inPool, **output}, targetBackUp
"""
    return ast.parse(code).body

def restoreFromSTK():
    code = \
"""
def restoreFromSTK(localPool: Dict, backUp: Dict):
    if backUp:
        localPool.update(backUp)
    return localPool
"""
    return ast.parse(code).body

def unstackedOutDict():
    code = \
"""
def unstackedOutDict(outDict: Dict, targets: Dict):
    if not targets:
        return input
    output = {}
    for name, offset in targets.items():
        l = outDict[name]
        for i in range(len(l)):
            output[f'{name}-idx{i+offset}'] = l[i]
        outDict[name] = ('List', len(l))
    return {**outDict, **output}
"""
    return ast.parse(code).body

# Convert uploads to unstacked version
# old-uploads = [need, not need]
# new-uploads = [need, need-idx0, ...need-idxn, not need]
def unstackedUploads():
    code = \
"""
def unstackedUploads(outDict: Dict, uploads: Dict, targets: Dict):
    if not targets:
        return uploads
    newUploads = {}
    for nextNode, objList in uploads.items():
        newUploads[nextNode] = []
        for objName, toL, toR in objList:
            if objName in targets:
                offset = targets[objName]
                l = len(outDict[objName])
                newUploads[nextNode] += [(f'{objName}-idx{offset+i}', toL, toR) for i in range(l)]
            newUploads[nextNode].append((objName, toL, toR))
    return newUploads
"""
    return ast.parse(code).body