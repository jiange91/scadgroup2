from StateMachine import profUtil
from . import base
from ruamel.yaml import YAML
import copy, os
import astor, ast
from typing import Dict, List
from ..console import MB, KB, ElementInDAG

class LambdaGenerator:
    def __init__(self, lambdaEle: ElementInDAG, services: List[ast.Module]):
        self.lambdaEle = lambdaEle
        self.astServices = services
    
    def to_soure(self, poolSize=None, pageSize=None, appInput=None, outPath:str=None):
        # Create AST module
        eleModule = ast.Module()
        eleModule.body = []

        # Meta config
        parent = None
        if self.lambdaEle.parent:
            parent = self.lambdaEle.parent.name
            if self.lambdaEle.parent.nType == 'Paral':
                parent = self.lambdaEle.parent.name+'-exit'
        metaConfig = base.compMetaConfig(['mem'], [parent], list(self.lambdaEle.children.keys()))

        # scad support
        scadImports = base.baseImports()
        for n in scadImports:
            eleModule.body.append(n)
        stkUstkSupport = []
        if self.lambdaEle.stackerSupport:
            stkUstkSupport += base.stackedPullMap()
            stkUstkSupport += base.stackedInPool()
            stkUstkSupport += base.restoreFromSTK()
        if self.lambdaEle.unstackerSupport:
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

        # Registe pool
        if self.lambdaEle.parent:
            pool = base.setPoolFromPrev(parent)
        else:
            poolSize = poolSize or 512 * MB
            pageSize = pageSize or 4 * KB
            pool = base.setRootObjPool('mem', poolSize, pageSize, appInput)
        for n in pool:
            mainBody.insert(-1, n)

        # Initial uploads
        initUploads = base.baseUploads(self.lambdaEle)
        for n in initUploads:
            mainBody.insert(-1, n)
    
        # For each lambda unit
        for i, lu in enumerate(self.lambdaEle.composition):
            unitBody = []
            # Load input for current node
            self.lambdaEle.downloadOnNeeds()
            downloads = self.lambdaEle.inGuide[lu.name]
            curRemote = downloads.pullFromRemote
            curCareful = downloads.beCareful
            if not self.lambdaEle.parent and i == 0:
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
        objPoolUpdate = base.UpdateLR('objPool') + base.clearLR(self.lambdaEle)
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
        with open(os.path.join(outPath, self.lambdaEle.name+'.o.py'), 'w') as f:
            f.write(codeStr)
        

    def forward(self, node: profUtil.Node):
        code = \
f"""
{node.name}OutDict = {node.funcName}(**{node.name}InDict)
"""
        return ast.parse(code).body
    
    def chooseUploads(self):
        code = \
"""
selectedUploads = list(uploads.values())[0]
"""
        return ast.parse(code).body