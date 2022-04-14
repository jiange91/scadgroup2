import ast
from ruamel.yaml import YAML
import copy, os
import astor
from .console import KB, MB, ElementInDAG
from .generator_util import base, genLambda, genSwitch, genParal

class Generator:
    def __init__(self, dagRoot: ElementInDAG, lambdas: str, dest: str):
        self.dagRoot = dagRoot
        self.destPath = dest
        if not os.path.exists(dest):
            os.makedirs(dest)
        # Get ast of each block task
        stack = [dagRoot]
        self.astLambda = {}
        while stack:
            e = stack.pop()
            self.astLambda[e.name] = []
            for b in e.composition:
                with open(os.path.join(lambdas, b.funcName)+'.py', 'r') as s:
                    self.astLambda[e.name].append(ast.parse(s.read()))
            stack += list(e.children.values())

    def build(self, poolSize=512*MB, pageSize=1*MB, appInput=None):
        # memory for objpool
        poolConfig, memSize = base.genObjPoolMeta(poolSize, self.dagRoot)
        yaml = YAML()
        yaml.indent(mapping=4, sequence=4, offset=2)
        with open(os.path.join(self.destPath, 'mem.o.yaml'), 'w') as memf:
            yaml.dump(poolConfig, memf)
        
        stack = [self.dagRoot]
        while stack:
            dagEle = stack.pop()
            if dagEle.nType == 'Lambda':
                lamGen = genLambda.LambdaGenerator(dagEle, self.astLambda[dagEle.name])
                lamGen.to_soure(memSize, pageSize, appInput, self.destPath)
            elif dagEle.nType == 'SwitchFlow':
                switchGen = genSwitch.SwitchFlowGenerator(dagEle, self.astLambda[dagEle.name])
                switchGen.to_soure(memSize, pageSize, appInput, self.destPath)
            elif dagEle.nType == 'Paral':
                paralGen = genParal.ParalGenerator(dagEle, self.astLambda[dagEle.name])
                paralGen.to_soure(memSize, pageSize, self.destPath)
            for c in dagEle.children.values():
                stack.append(c)
        print("Building Complete!!!")
