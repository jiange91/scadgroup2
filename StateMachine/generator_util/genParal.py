from StateMachine import profUtil
from . import base
from ruamel.yaml import YAML
import copy, os
import astor, ast
from typing import List
from ..console import MB, KB, ElementInDAG
import math

class ParalGenerator:
    def __init__(self, paralEle: ElementInDAG, services: List[ast.Module]):
        self.paralEle = paralEle
        self.astServices = services
    
    def to_soure(self, poolSize=None, pageSize=None, outPath:str=None):
        paral = self.paralEle.paral

        # Prepare worker memory elements 
        yaml = YAML()
        yaml.indent(mapping=4, sequence=4, offset=2)
        for p in range(paral):
            memWorker, workerPoolSize = self.initWorkersMem(p)
            with open(os.path.join(outPath, f'{self.paralEle.name}-worker-{p}-mem.o.yaml'), 'w') as memf:
                yaml.dump(memWorker, memf)
        
        # Prepare worker targets
        self.entry_to_source(outPath)

        # Generate source code for each worker
        for p in range(self.paralEle.paral):
            self.worker_to_source(p, workerPoolSize, pageSize, outPath)
        
        # Generate exit point for collection and synchronization
        self.exit_to_source(outPath)

    def entry_to_source(self, outPath: str=None):
        # Create AST module
        eleModule = ast.Module()
        eleModule.body = []
        
        # Meta config
        parent = self.paralEle.parent.name
        if self.paralEle.parent.nType == 'Paral':
            parent = self.paralEle.parent.name+'-exit'
        metaConfig = base.compMetaConfig(['mem'], [parent], [self.paralEle.name+f'-worker-{p}' for p in range(self.paralEle.paral)] + [self.paralEle.name+'-exit'])

        # Scad support
        for n in base.baseImports():
            eleModule.body.append(n)
        
        # Add main function
        mainDef, mainBody = base.mainBlock()

        # Regist Pool
        pool = base.setPoolFromPrev(parent)
        for n in pool:
            mainBody.insert(-1, n)
        
        # For each worker, get target information
        workerTargets = self.getWorkerSTKTargets()
        for n in workerTargets:
            mainBody.insert(-1, n)
        
        # Pass to parallel children
        toWorker = self.toWorker()
        for n in toWorker:
            mainBody.insert(-1, n)
        
        # Append main and metaConfig
        mainBody.pop()
        eleModule.body.append(mainDef)
        codeStr = astor.to_source(eleModule)
        codeStr = metaConfig + codeStr
        with open(os.path.join(outPath, self.paralEle.name+'.o.py'), 'w') as f:
            f.write(codeStr)

        
    def getWorkerSTKTargets(self):
        targets = self.paralEle.composition[0].stackerTargets
        code = \
f"""
stkTargets = {targets}
iterName = list(stkTargets)[0]
localPool[iterName] = objPool.materialize(iterName)
iterLen = localPool[iterName][1]

paral = {self.paralEle.paral}
workloads = math.ceil(iterLen / paral)
i = 0
p = 0
context_dict = {{}}
while i < iterLen:
    if i + workloads < iterLen:
        begin, end = i, i + workloads
    else:
        begin, end = i, iterLen
    targetsWithRange = {{tname: (begin,end,1) for tname in stkTargets}}
    context_dict[f'{self.paralEle.name}-worker-{{p}}'] = targetsWithRange
    i += workloads
    p += 1
"""
        return ast.parse(code).body

    def toWorker(self):
        code = \
"""
context_dict['objPool'] = objPool
context_dict['workloads'] = workloads
context_dict_in_byte = serial_context.dumps(context_dict)
return {'meta': base64.b64encode(context_dict_in_byte).decode("ascii")}
"""
        return ast.parse(code).body

    def initWorkersMem(self, pid):
        size = self.paralEle.totalOutSize
        poolConfig = {'type': 'memory'}
        memSizeInMB = math.ceil(size / MB)
        baseSize = 512
        while baseSize < memSizeInMB:
            baseSize += 128
        poolConfig['limits'] = {'mem': str(baseSize) + ' MB'}
        poolConfig['corunning'] = [
            self.paralEle.name+'-exit',
            self.paralEle.name+f'-worker-{pid}']
        return poolConfig, baseSize * MB
    
    # Currently worker does not unstack the result, leave it to exit point
    def worker_to_source(self, pid, poolSize, pageSize, outPath):
        # Create AST module
        eleModule = ast.Module()
        eleModule.body = []

        # Meta config
        parents = [self.paralEle.name]
        mems = [f'{self.paralEle.name}-worker-{pid}-mem', 'mem']
        metaConfig = base.compMetaConfig(mems, parents, [f'{self.paralEle.name}-exit'])
        
        # Scad support
        scadImports = base.baseImports()
        for n in scadImports:
            eleModule.body.append(n)
        stkUstkSupport = []
        stkUstkSupport += base.stackedPullMap()
        stkUstkSupport += base.stackedInPool()
        stkUstkSupport += base.restoreFromSTK()
        for n in stkUstkSupport:
            eleModule.body.append(n)
        
        # Import whole service file
        astSvc = []
        for s in self.astServices:
            astSvc += s.body
        for n in astSvc:
            eleModule.body.append(n)
        
        # Add main function
        mainDef, mainBody = base.mainBlock()

        # Register prevPool and self Pool
        prevPool = base.setPoolFromPrev(f'{self.paralEle.name}')
        selfPool = self.workerSetSelfPool(pid, poolSize, pageSize)
        for n in prevPool:
            mainBody.insert(-1, n)
        for n in selfPool:
            mainBody.insert(-1, n)
        
        # Initial uploads and targets
        uploadsBeforeUSTK = self.constructWorkerUploads()
        for n in uploadsBeforeUSTK:
            mainBody.insert(-1, n)
        
        # load input for worker
        workerInput = self.getWorkerInput(pid)
        for n in workerInput:
            mainBody.insert(-1, n)
        
        # forward
        forward = self.forward(self.paralEle.composition[0])
        for n in forward:
            mainBody.insert(-1, n)
        
        # update using output
        afterForward = self.updateOutput()
        for n in afterForward:
            mainBody.insert(-1, n)

        # directly push output to selfpool
        directPush = self.directPush()
        for n in directPush:
            mainBody.insert(-1, n)
        
        # to dependent
        for n in self.toDependents(pid):
            mainBody.insert(-1, n)
        
        # Append main and metaConfig
        mainBody.pop()
        eleModule.body.append(mainDef)
        codeStr = astor.to_source(eleModule)
        codeStr = metaConfig + codeStr
        with open(os.path.join(outPath, self.paralEle.name+f'-worker-{pid}.o.py'), 'w') as f:
            f.write(codeStr)
    
    
    def workerSetSelfPool(self, pid, poolSize, pageSize):
        code = \
f"""
selfPool = ObjPool(
    name="{self.paralEle.name}-worker-{pid}-mem",
    memSize= {poolSize},
    pageSize= {pageSize} 
)
selfPool.registerTrans(actionLib=action)
"""  
        return ast.parse(code).body
    
    def constructWorkerUploads(self):
        objList = list(self.paralEle.uploads.values())[0]
        uploadsDict = f"{{'{self.paralEle.name}-exit': {objList}}}"
        code = \
f"""
uploads = {uploadsDict}
"""
        return ast.parse(code).body

    def getWorkerInput(self, pid):
        code = \
f"""
pullR = {self.paralEle.inMap}
workloads = context_dict['workloads']
stkTargets = context_dict['{self.paralEle.name}-worker-{pid}']
pullR, stkTargets = stackedPullMap(pullR, stkTargets, objPool)
for name in pullR:
    localPool[name] = objPool.materialize(name)
localPool, restore = stackedInPool(localPool, stkTargets)
inMap = {self.paralEle.inMap}
inDict = {{}}
for inName in inMap:
    inDict[inName] = localPool[inName]
"""
        return ast.parse(code).body
    
    def exit_to_source(self, outPath):
        # Create AST module
        eleModule = ast.Module()
        eleModule.body = []
        
        #Meta Config
        parents = [self.paralEle.name + f'-worker-{p}' for p in range(self.paralEle.paral)] + [self.paralEle.name]
        mems = [self.paralEle.name+f'-worker-{p}-mem' for p in range(self.paralEle.paral)] + ['mem']
        metaConfig = base.compMetaConfig(mems, parents, list(self.paralEle.children.keys()))

        # Scad support
        scadImports = base.baseImports()
        for n in scadImports:
            eleModule.body.append(n)
        
        # Add main function
        mainDef, mainBody = base.mainBlock()
        
        # Regist pools from entry and workers
        poolStepup = self.initExitPool()
        for n in poolStepup:
            mainBody.insert(-1, n)
        
        # Download from worker pool and update mainstream pool
        exitForward = self.exitUpdate()
        for n in exitForward:
            mainBody.insert(-1, n)
        
        # To next element
        for n in base.toDependents():
            mainBody.insert(-1, n)
        
        # Append main and metaConfig
        mainBody.pop()
        eleModule.body.append(mainDef)
        codeStr = astor.to_source(eleModule)
        codeStr = metaConfig + codeStr
        with open(os.path.join(outPath, self.paralEle.name+'-exit.o.py'), 'w') as f:
            f.write(codeStr) 
            
        
    def initExitPool(self):
        # Get mainstream pool
        appPoolSetup = base.setPoolFromPrev(self.paralEle.name, 'objPool')

        # Get worker pools
        paral = self.paralEle.paral
        code = \
f"""
workerPools = []
for p in range({paral}):
    context_dict_in_b64 = params[f'{self.paralEle.name}-worker-{{p}}'][0]['meta']
    context_dict_in_byte = base64.b64decode(context_dict_in_b64)
    context_dict = serial_context.loads(context_dict_in_byte)
    workerPools.append(context_dict[f'{self.paralEle.name}-worker-{{p}}-pool'])
    workerPools[p].registerTrans(actionLib=action)
"""
        return appPoolSetup + ast.parse(code).body
    
    def exitUpdate(self):
        getUploads = base.baseUploads(self.paralEle) + self.chooseUploads()
        code = \
f"""
ustkMap = {self.paralEle.composition[0].unstkMap}
curOffsets = {{vname: 0 for vname, b in ustkMap.items() if b}}
for vname, isLocal, isRemote in selectedUploads:
    needUstk = ustkMap[vname]
    # pull values iteratives and upload
    for i in range({self.paralEle.paral}):
        obj = workerPools[i].materialize(vname)
        if not needUstk:
            objPool.upload_remote(obj, f'{{vname}}-idx{{i}}')
        else:
            ofst = curOffsets[vname]
            for l in range(len(obj)):
                objPool.upload_remote(obj[l], f'{{vname}}-idx{{l+ofst}}')
            curOffsets[vname] += len(obj)
    # upload meta data for output
    if not needUstk:
        objPool.upload_remote(('List', {self.paralEle.paral}), vname)
    else:
        objPool.upload_remote(('List', curOffsets[vname]), vname)
"""
        return getUploads + ast.parse(code).body

    def forward(self, node: profUtil.Node):
        code = \
f"""
outDict = {node.funcName}(**inDict)
"""
        return ast.parse(code).body
    
    def updateOutput(self):
        code = \
f"""
localPool.update(outDict)
"""
        return ast.parse(code).body
    
    def directPush(self):
        code = \
"""
for vname, obj in outDict.items():
    selfPool.upload_remote(obj, vname)
"""
        return ast.parse(code).body

    def chooseUploads(self):
        code = \
"""
selectedUploads = list(uploads.values())[0]
"""
        return ast.parse(code).body
    
    def toDependents(self, pid):
        code = \
f"""
context_dict = {{}}
context_dict['{self.paralEle.name}-worker-{pid}-pool'] = selfPool
context_dict_in_byte = serial_context.dumps(context_dict)
return {{'meta': base64.b64encode(context_dict_in_byte).decode("ascii")}}
"""
        return ast.parse(code).body
    