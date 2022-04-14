import enum
from StateMachine import profUtil
from . import base
from ruamel.yaml import YAML
import copy, os
import astor, ast
from typing import List
from ..console import MB, KB, ElementInDAG

class SwitchFlowGenerator:
    def __init__(self, switchEle: ElementInDAG, services: List[ast.Module]):
        self.switchEle = switchEle
        self.astServices = services
    
    def to_soure(self, poolSize=None, pageSize=None, appInput=None, outPath:str = None):
        # Creat AST module
        eleModule = ast.Module()
        eleModule.body = []

        # Meta config
        parent = None
        if self.switchEle.parent:
            parent = self.switchEle.parent.name
            if self.switchEle.parent.nType == 'Paral':
                parent = self.switchEle.parent.name+'-exit'
        metaConfig = base.compMetaConfig(['mem'], [parent], list(self.switchEle.children.keys()))

        # scad support
        scadImports = base.baseImports()
        for n in scadImports:
            eleModule.body.append(n)
        stkUstkSupport = []
        if self.switchEle.stackerSupport:
            stkUstkSupport += base.stackedPullMap()
            stkUstkSupport += base.stackedInPool()
            stkUstkSupport += base.restoreFromSTK()
        if self.switchEle.unstackerSupport:
            stkUstkSupport += base.unstackedOutDict()
            stkUstkSupport += base.unstackedUploads()
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

        # Register pool
        if self.switchEle.parent:
            pool = base.setPoolFromPrev(parent)
        else:
            poolSize = poolSize or 512 * MB
            pageSize = pageSize or 4 * KB
            pool = base.setRootObjPool('mem', poolSize, pageSize, appInput)
        for n in pool:
            mainBody.insert(-1, n)

        # Initial uploads and nexts
        initUploads = base.baseUploads(self.switchEle)
        for n in initUploads:
            mainBody.insert(-1, n)
    
        # For each lambda unit
        for i, lu in enumerate(self.switchEle.composition):
            unitBody = []
            # Load input for current node
            self.switchEle.downloadOnNeeds()
            downloads = self.switchEle.inGuide[lu.name]
            curRemote = downloads.pullFromRemote
            curCareful = downloads.beCareful
            if not self.switchEle.parent and i == 0:
                unitBody += base.getNodeInput(True, lu, curRemote, curCareful)
            else:
                unitBody += base.getNodeInput(False, lu, curRemote, curCareful)
            
            # Forward
            unitBody += self.forward(lu)

            # Update pool and uploads
            unitBody += base.updateOutput(lu)

            for n in unitBody:
                mainBody.insert(-1, n)
        
        # set uploads
        for n in self.chooseUploads():
            mainBody.insert(-1, n)

        # update objpool 
        objPoolUpdate = base.UpdateLR('objPool') + base.clearLR(self.switchEle)
        for n in objPoolUpdate:
            mainBody.insert(-1, n)
        
        # To dependent
        for n in base.toDependents():
            mainBody.insert(-1, n)
            
        # Append main and metaConfig
        mainBody.pop()
        eleModule.body.append(mainDef)
        codeStr = astor.to_source(eleModule)
        codeStr = metaConfig + codeStr
        with open(os.path.join(outPath, self.switchEle.name+'.o.py'), 'w') as f:
            f.write(codeStr)

    def forward(self, node: profUtil.Node):
        if node.nType == 'Lambda':
            code = \
f"""
{node.name}OutDict = {node.funcName}(**{node.name}InDict)
"""
        if node.nType == 'SwitchFlow':
            code = \
f"""
nid, {node.name}OutDict = {node.funcName}(**{node.name}InDict)
"""
        return ast.parse(code).body
    
    def chooseUploads(self):
        code = \
"""
selectedUploads = uploads[nexts[nid]]
"""
        return ast.parse(code).body