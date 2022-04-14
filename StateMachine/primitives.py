from __future__ import annotations
from copy import deepcopy
from functools import wraps
from typing import Iterable, List, Dict, Set, Tuple
from filprofiler.api import profile
import os
import math
import copy
from pympler import asizeof
from pyinstrument import Profiler
from sys import exit

p = Profiler()

def log_state(forward):
    @wraps(forward)
    def logged_forward(self, input: dict, *args, **kargs):
        p.reset()
        p.start()
        rel = forward(self, input, *args, **kargs)
        s = p.stop()
        self.callTime = s.duration
        return rel
    return logged_forward
        
class Block:
    def __init__(self, name):
        self.name = name
        self.next = None
        self.inPool = 'default_Pool'
        self.outPool = 'default_Pool'
        # self.inputMap = Dict[str, type]
        self.inputMap: Set[str] = set()
        # self.outputMap = Dict[str, type]
        self.outputMap: Set[str] = set()
        self.outputSize = dict()
        self.inputSize = dict()
        self.callTime = 0

    def configPoolSource(self, inPool=None, outPool=None):
        if inPool:
            self.inPool = inPool
        if outPool:
            self.outPool = outPool

    def checkInput(self, input: dict):
        if self.inputMap:
            # for reqVar, t in self.inputMap.items():
            #     if reqVar not in input:
            #         return Fail(msg = f"Missing input {reqVar}")
            #     if not isinstance(input[reqVar], t):
            #         return Fail(msg=f"Incorrect type {reqVar}, expected {t}")
            for reqVar in self.inputMap:
                if reqVar not in input:
                    print(f"Missing input {reqVar}")
                    exit(2)

    def checkOutput(self, output: dict):
        if self.outputMap:
            # for commitVar, t in self.outputMap.items():
            #     if commitVar not in output:
            #         return Fail(msg = f"Missing output {commitVar}")
            #     if not isinstance(output[commitVar], t):
            #         return Fail(msg=f"Incorrect type {commitVar}, expected {t}")
            for commitVar in self.outputMap:
                if commitVar not in output:
                    print(f"Missing output {commitVar}")
                    exit(2)
                    
    def filterInput(self, input: dict):
        return {k: input[k] for k in self.inputMap}
    
    def filterOutput(self, output: dict):
        return {k: output[k] for k in self.outputMap}
    
    def forward(self, input: dict):
        self.checkInput(input)
        finput = self.filterInput(input)
        self.recordInSize(finput)
        self.recordOutSize(finput)
        self.checkOutput(finput)
        return input 

    def setInputMap(self, m: Set[str]):
        self.inputMap = m

    def setOutputMap(self, m: Set[str]):
        self.outputMap = m

    def setNext(self, nextStep):
        self.next = nextStep
    
    def recordOutSize(self, rel: dict):
        for k, v in rel.items():
            self.outputSize[k] = asizeof.asizeof(v)

    def recordInSize(self, input: dict):
        for k, v in input.items():
            self.inputSize[k] = asizeof.asizeof(v)

class State:
    def __init__(self, name = None, msg = ""):
        self.name = name
        self.msg = msg
        self.next = None
    @log_state
    def forward(self, input):
        return input
    
class Fail(State):
    def __init__(self, name="Fail", msg = ""):
        super().__init__(name, msg)

class Success(State):
    def __init__(self, name="Success", msg = ""):
        super().__init__(name, msg)

class Application(Block):
    def __init__(self, name):
        super().__init__(name)
        self.outputSizeRecord = {}
        self.inputSizeRecord = {}
        self.inputMaps = {}
        self.outputMaps = {}
        self.callTimes = {}
        self.inPool = 'outside_Pool'
        self.inPools = {}
        self.outPools = {}
        self.totalTime = 0

    def setNext(self, nextStep):
        self.root = self.next = nextStep

    def setInputMap(self, m: Set[str]):
        super().setInputMap(m)
        super().setOutputMap(m)

    def run(self, input: dict, profMode = 0, outDir = None):
        cur = self.next
        self.checkInput(input)
        outpath = None
        if profMode != 0:
            outpath = outDir or 'tmp'
            self.outpath = outpath
        while cur:
            self.inputMaps[cur.name] = cur.inputMap
            if profMode and outpath:
                outpath = os.path.join(outpath, cur.name)
                input = profile(lambda: cur.forward(input), outpath)
            else:
                input = cur.forward(input)
            self.outputMaps[cur.name] = cur.outputMap
            self.inputSizeRecord[cur.name] = cur.inputSize
            self.outputSizeRecord[cur.name] = cur.outputSize
            self.inPools[cur.name] = cur.inPool
            self.outPools[cur.name] = cur.outPool
            self.totalTime += cur.callTime
            self.callTimes[cur.name] = cur.callTime
            cur= cur.next
        return input

class Lambda(Block):
    def __init__(self, name='default-lambda'):
        super().__init__(name)
        self.unstacker = None
        self.stacker = None

    def materialize_worker(self, func: function, inputMap: Set[str], outputMap: Set[str]):
        self.func = func
        self.funcName = func.__name__
        self.setInputMap(inputMap)
        self.setOutputMap(outputMap)
        self.stacker: Stacker = None
        self.unstacker: Unstacker = None

    @log_state
    def forward(self, input: dict):
        if self.stacker:
            input = self.stacker.forward(input)
        self.checkInput(input)
        finput = self.filterInput(input)
        self.recordInSize(finput)
        rel = self.func(**finput)
        # rel = input | rel
        self.recordOutSize(rel)
        self.checkOutput(rel)
        if self.stacker:
            self.stacker.restoreTargets(input)
        rel = {**input, **rel}
        if self.unstacker:
            rel = self.unstacker.forward(rel)
        return rel
    
    def registStacker(self, stk):
        self.stacker = stk
    
    def registUnstacker(self, ustk):
        self.unstacker = ustk
    
class Unstacker(Block):
    def __init__(self, name='default-unstacker', targets: dict = {}):
        super().__init__(name)
        # targets: {obj: offset}
        self.targets = targets
        self.setInputMap(set(self.targets.keys()))

    def forward(self, input: dict):
        self.checkInput(input)
        # Check unstackable
        for name in self.inputMap:
            if not isinstance(input[name], List):
                print('Cannot unstack non-List object: '+ name)

        # Unstack each object
        output = {}
        finput = self.filterInput(input)
        for name, offset in self.targets.items():
            l = finput[name]
            for i in range(len(l)):
                output[f'{name}-idx{i+offset}'] = l[i]
            input[name] = ('List', len(l))
        return {**input, **output}

class Stacker(Block):
    def __init__(self, name='default-stacker', targets: dict={}):
        super().__init__(name)
        self.targets = targets
        self.setInputMap(set(self.targets.keys()))
        self.targetBackUp = {}
    
    # targets : {objname: range list | tuple }
    def forward(self, input: dict):
        self.checkInput(input)
        self.targetBackUp = {k: input[k] for k in self.targets.keys()}
        output: Dict[str, List] = {}
        for name, (start, end, step) in self.targets.items():
            cur = start or 0
            step = step or 1
            if not end:
                output[name] = []
                while True:
                    curUstk = f'{name}-idx{cur}'
                    if curUstk in input:
                        output[name].append(input[curUstk])
                    else:
                        break
                    cur += step
            else:
                output[name] = [input[f'{name}-idx{i}'] for i in range(start, end, step)]

        return {**input, **output}
    
    def restoreTargets(self, input: dict):
        input.update(self.targetBackUp)
        return input
            
class SwitchFlow(Block):
    def __init__(self, name='default-Switch-Flow'):
        super().__init__(name)

    # cmp takes all keyword and return the index of next function
    def addRule(self, inputMap, outputMap, cmp: function, nexts: list, default=0):
        self.setInputMap(inputMap)
        self.setOutputMap(outputMap)
        self.cmp = cmp
        self.funcName = cmp.__name__
        self.nexts = nexts
        self.default = default

    @log_state
    def forward(self, input: dict):
        self.checkInput(input)
        finput = self.filterInput(input)
        self.recordInSize(finput)
        nid, cmpOut = self.cmp(**finput)
        if nid is None:
            self.next = self.default
        else:
            self.next = self.nexts[nid]
        self.recordOutSize(cmpOut)
        self.checkOutput(cmpOut)
        out = {**input, **cmpOut}
        # out = input | cmpOut
        return out

class Paral(Block):
    def __init__(self, name='default-Parallelism'):
        super().__init__(name)
    
    def registWorker(self, inputMap, outputMap, worker, targets, parallelism, ustkMap: Dict):
        self.setInputMap(inputMap)
        self.setOutputMap(outputMap)
        self.parallelism = parallelism
        self.worker = worker
        self.funcName = worker.__name__
        self.targets = targets
        self.inputSize = {k: 0 for k in inputMap}
        self.outputSize = {k: 0 for k in outputMap}
        self.ustkMap = ustkMap
    
    @log_state
    def forward(self, input: dict):
        iterLen = self.checkIter(input)
        workloads = math.ceil(iterLen / self.parallelism)
        i = 0
        p = 0
        curOffsets = {objName: 0 for objName, needUstk in self.ustkMap.items() if needUstk}
        while i < iterLen:
            if i + workloads < iterLen:
                begin, end = i, i + workloads
            else:
                begin, end = i, iterLen
            targetsWithRange = {tname: (begin,end,1) for tname in self.targets}
            stk = Stacker(name=f'{self.name}-ustk-{p}', targets=targetsWithRange)
            sinput = stk.forward(input)
            # print(sinput)
            self.checkInput(sinput)
            finput = self.filterInput(sinput)
            self.paralInsize(finput)
            rel = self.worker(**finput)
            self.paralOutSize(rel)
            self.checkOutput(rel)
            
            stk.restoreTargets(input)

            targetsWithOffset = {}
            for outName, needUStk in self.ustkMap.items():
                if not needUStk:
                    rel[f'{outName}-idx{p}'] = rel[outName]
                else:
                    targetsWithOffset[outName] = curOffsets[outName]
                    curOffsets[outName] += len(rel[outName])
            ustk = Unstacker(name=f'{self.name}-stk-{p}', targets=targetsWithOffset)
            rel = ustk.forward(rel)
            input = {**input, **rel}
            i += workloads
            p += 1
        for outName, needUStk in self.ustkMap.items():
            if not needUStk:
                input[outName] = ('List', self.parallelism)
            else:
                input[outName] = ('List', curOffsets[outName])
        return input
    
    def paralInsize(self, input: dict):
        for k, v in input.items():
            self.inputSize[k] = max(asizeof.asizeof(v), self.inputSize[k])

    def paralOutSize(self, output: dict):
        for k, v in output.items():
            self.outputSize[k] = max(asizeof.asizeof(v), self.outputSize[k])

    def checkIter(self, input: dict):
        iterLen = 0
        for k in self.targets:
            if k not in input:
                print(f'Missing target {k}')
                exit(1)
            if input[k][0] != 'List':
                print(f'{k} should have been unstacked before')
                exit(1)
            if not iterLen:
                iterLen = input[k][1]
            elif iterLen != input[k][1]:
                print(f'Targets should have identical length: {k}-{input[k][1]}, expected: {iterLen}')
                exit(1)
        return iterLen


