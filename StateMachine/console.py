from __future__ import annotations

from . import primitives, profUtil 
from typing import List, Dict, Set, Tuple
import math
import copy, os

B = 1
KB = 1024
MB = 1024 * KB
GB = 1024 * MB

OVERHEAD_CTX_SETUP= 0.5
OVERHEAD_ELE_TALK_PMB = 8e-5
NO_PROF = 0
MAX_PROF = 1
MEAN_PROF = 2

class Inputs:
    inputCase: List[str]
    inputMaps: Dict[str, dict]
    
    def __init__(self):
        self.inputCase = []
        self.inputMaps = {}
    
    def addInput(self, input, comment=None):
        comment = comment or f'testCase-{len(self.inputCase)}'
        self.inputCase.append(comment)
        self.inputMaps[comment] = input

    def getInput(self, case):
        return self.inputMaps[case]

class AggProf:
    def __init__(self, rt, iroh, ele):
        self.reducedTime = rt
        self.increasedROH = iroh
        self.ele = ele
        self.r = self.sigmoidr(rt, iroh)
    
    def sigmoidr(self, rt, iroh):
        sigiroh = 1 / (1+math.exp(-iroh))
        return rt / sigiroh

    def __lt__(self, other):
        return self.r < other.r

    def __repr__(self) -> str:
        return f'rt: {self.reducedTime}, iroh: {self.increasedROH}, ratio: {self.r}'

class VirtualObj:
    def __init__(self, name, size=0, fromP=True, fromR=False, lastReq: List[ElementInDAG] = None):
        self.name = name
        self.size = size
        self.fromPrev = fromP
        self.fromRemote = fromR
        self.lastReq = lastReq
    
    def update(self, another: VirtualObj):
        if self.name != another.name:
            print("Merge two distinct obj")
            exit(1)
        self.size = max(self.size, another.size)
        self.fromPrev = self.fromPrev or another.fromPrev
        self.fromRemote = self.fromRemote or another.fromRemote
        self.lastReq += another.lastReq

    def __str__(self) -> str:
        return f'({self.name}, FromPrev: {self.fromPrev}, FromRemote: {self.fromRemote})'
    
    def __repr__(self) -> str:
        return f'("{self.name}", {self.fromPrev}, {self.fromRemote})'

class VirtualObjPool:
    def __init__(self):
        self.pool: Dict[str, VirtualObj] = {}

    def giveWhatIHave(self, outSize: Dict):
        uploads = []
        for name in outSize.keys():
            if name in self.pool:
                obj = self.pool.pop(name)
                # Tell last remote req to clear remote pool
                if obj.fromRemote:
                    for node in obj.lastReq:
                        node.clearRemote.add(obj.name)
                uploads.append(obj)
        for obj in self.pool.values():
            if not obj.fromRemote:
                obj.fromRemote = True
                obj.fromPrev = False
        return uploads
    
    @DeprecationWarning
    def regWhatINeed(self, inSize: Dict, node: ElementInDAG):
        for name, size in inSize.items():
            if name in self.pool:
                self.pool[name].fromPrev = True
            else:
                self.pool[name] = VirtualObj(name, size=size, fromP=True, fromR=False, lastReq=[node])
    
    # For remote only registration
    def regWhatINeedR(self, inSize: Dict, node: ElementInDAG):
        for name, size in inSize.items():
            if name not in self.pool:
                self.pool[name] = VirtualObj(name, size=size, fromP=False, fromR=True, lastReq=[node])
                
                
    def mergeFromPools(self, pools: List[VirtualObjPool]):
        for p in pools:
            for objName, vobj in p.pool.items():
                if objName not in self.pool:
                    self.pool[objName] = vobj
                else:
                    self.pool[objName].update(vobj)

    # Return the size of current remote pool
    def getRemoteSize(self):
        s = 0
        for obj in self.pool.values():
            if obj.fromRemote:
                s += obj.size
        return s

class InputGuide:
    def __init__(self, name):
        self.name = name
        self.pullFromRemote = set()
        self.beCareful = set()
    

class ElementInDAG:
    def __init__(self, node: profUtil.Node, parent: ElementInDAG):
        self.name = node.name
        self.newAlloc = node.newAlloc
        self.parent = parent
        self.children: Dict[str, ElementInDAG] = {}
        self.nType = node.nType
        self.inMap = node.inMap
        self.inSize = node.inSize
        self.outMap = node.outMap
        self.outSize = node.outSize
        self.callTime = node.callTime
        self.paral = node.paral
        self.nexts = node.nexts
        self.inRefCount = {k: 1 for k in self.inMap}
        self.composition: List[profUtil.Node] = [node]
        self.clearRemote = set()
        self.stackerSupport = False
        self.unstackerSupport = False
        if node.stackerTargets:
            self.stackerSupport = True
        if node.unstackerTargets:
            self.unstackerSupport = True
        self.profSelf()

    def profSelf(self):
        self.totalInSize = sum(self.inSize.values()) / MB
        self.totalOutSize = sum(self.outSize.values()) / MB
        self.prepareInput = self.totalInSize * OVERHEAD_ELE_TALK_PMB 
        self.prepareOutput = self.totalOutSize * OVERHEAD_ELE_TALK_PMB
        self.memTime = (self.totalInSize + self.newAlloc / MB) * (self.prepareInput + self.prepareOutput + self.callTime + OVERHEAD_CTX_SETUP)
        self.duration = self.prepareInput + self.prepareOutput + self.callTime + OVERHEAD_CTX_SETUP

    def profAggNext(self, nextEle: ElementInDAG):
        # Communication reduction
        if self.nType == 'Paral' or self.nType == 'SwitchFlow':
            print(f"{self.nType} should not be aggregated with the next one")
            exit(1)
        if nextEle.nType == 'Paral':
            print(f"Should not aggregate with Paral: {self.name}")
            exit(1)
        commonIO = 0
        for output, outSize in self.outSize.items():
            if output in nextEle.inMap:
                commonIO += outSize / MB
            if output in nextEle.outMap:
                commonIO += outSize / MB
        for input, inputSize in nextEle.inSize.items():
            # Need copy-on-write support
            if input in self.outMap or input in self.inMap:
                commonIO += inputSize / MB
        reducedTime = commonIO * OVERHEAD_ELE_TALK_PMB + OVERHEAD_CTX_SETUP

        # Resource Overhead:
        notAggResource = self.memTime + nextEle.memTime
        aggPeakMem = max((self.totalInSize+self.newAlloc/MB), (nextEle.totalInSize+nextEle.newAlloc/ MB))
        aggTime = self.duration + nextEle.duration - reducedTime
        aggResource = aggPeakMem * aggTime
        increasedROH = aggResource - notAggResource
        return AggProf(reducedTime, increasedROH, self)

    def aggWithNext(self):
        if self.nType == 'SwitchFlow' or self.nType == 'Paral':
            print(f"Fail agg current {self.nType}")
            exit(1)
        next = list(self.children.values())[0]
        if next.nType == 'Paral':
            print(f'Cannot agg with Paral: {self.name}')
            exit(1)

        if next.nType == 'SwitchFlow':
            self.nType = 'SwitchFlow'
        self.nexts = next.nexts
        # Update fields
        self.stackerSupport = self.stackerSupport or next.stackerSupport
        self.unstackerSupport = self.unstackerSupport or next.unstackerSupport
        oldName = self.name
        self.name += '-' + next.name
        if self.parent:
            del self.parent.children[oldName]
            self.parent.children[self.name] = self
            for i, n in enumerate(self.parent.nexts):
                if n == oldName:
                    self.parent.nexts[i] = self.name 
        self.newAlloc = max(self.newAlloc, next.newAlloc)
        self.children = next.children
        for c in self.children.values():
            c.parent = self
        self.inMap.update(next.inMap - self.outMap)
        inSize = {}
        for input in self.inMap:
            if input in self.inSize:
                inSize[input] = self.inSize[input]
            elif input in next.inSize:
                inSize[input] = next.inSize[input]
        self.inSize = inSize
        self.outMap.update(next.outMap)
        outSize = {}
        for output in self.outMap:
            if output in next.outSize:
                outSize[output] = next.outSize[output]
            elif output in self.outSize:
                outSize[output] = self.outSize[output]
        self.outSize = outSize
        self.callTime += next.callTime
        self.composition += next.composition
        self.profSelf()
    
    # Used for hybrid communication
    @DeprecationWarning
    def updateObjPool(self, reqPools: Dict[str, VirtualObjPool]) -> VirtualObjPool:
        self.uploads: Dict[str, List[VirtualObj]] = {}
        combineNeeds = VirtualObjPool()
        if reqPools:
            combineNeeds.mergeFromPools(list(reqPools.values()))
        else:
            combineNeeds.regWhatINeed(self.outSize, self)
        muploads = combineNeeds.giveWhatIHave(self.outSize)
        combineNeeds.regWhatINeed(self.inSize, self)
        if len(self.children) > 0 and reqPools:
            for cname in self.children.keys():
                pool = reqPools[cname]
                self.uploads[cname] = pool.giveWhatIHave(self.outSize)
        elif len(self.children) == 0 and not reqPools:
            self.uploads['FinalOutput'] = muploads
        return combineNeeds
    
    # Used for remote only
    def updateObjPoolR(self, reqPools: Dict[str, VirtualObjPool]) -> VirtualObjPool:
        self.uploads: Dict[str, List[VirtualObj]] = {}
        combineNeeds = VirtualObjPool()
        if reqPools:
            # print(reqPools.keys())
            combineNeeds.mergeFromPools(list(reqPools.values()))
        else:
            combineNeeds.regWhatINeedR(self.outSize, self)
        # print('Needs: ' + combineNeeds.pool.keys())
        muploads = combineNeeds.giveWhatIHave(self.outSize)
        combineNeeds.regWhatINeedR(self.inSize, self)
        # print('Uploads: ' + muploads)
        if len(self.children) > 0 and reqPools:
            for cname in self.children.keys():
                pool = reqPools[cname]
                self.uploads[cname] = pool.giveWhatIHave(self.outSize)
        elif len(self.children) == 0 and not reqPools:
            self.uploads['FinalOutput'] = muploads
        return combineNeeds
    
    def downloadOnNeeds(self):
        self.inGuide: Dict[str, InputGuide] = {}
        # regist guids 
        for node in self.composition:
            self.inGuide[node.name] = InputGuide(node.name)

        # Top-down, get pull request
        inPool = set()
        for node in self.composition:
            pullR = set()
            for inName in node.inMap:
                if inName not in inPool:
                    pullR.add(inName)
            self.inGuide[node.name].pullFromRemote = pullR
            inPool.update(node.outMap)
        
        # Bottom-up, get be careful
        inCount = {}
        for node in self.composition[::-1]:
            beC = set()
            for outName in node.outMap:
                if outName in inCount:
                    inCount[outName] = 0
            for inName in node.inMap:
                inCount[inName] = inCount.get(inName, 0) + 1
                if inCount[inName] > 1:
                    beC.add(inName)
            self.inGuide[node.name].beCareful = beC
                    
    def __repr__(self, level=0) -> str:
        ret = '\t' * level + f'{self.name} \n' \
            + '\t' * level + f'inputSize: {self.inSize} \n' \
            + '\t' * level + f'New allocation: {self.newAlloc} \n' \
            + '\t' * level + f'Call time: {self.callTime} \n' \
            + '\t' * level + f'outSize: {self.outSize}\n'\
            + '\t' * level + f'totalInSize: {self.totalInSize}\n' \
            + '\t' * level + f'totalOutSize: {self.totalOutSize}\n' \
            + '\t' * level + f'memTime (MB*s): {self.memTime}\n'
        return ret
    
    def __str__(self, level=0) -> str:
        ret = self.__repr__(level)
        for child in self.children.values():
            ret += child.__str__(level+1)
        return ret

class DagOpt:
    def __init__(self, profTree: profUtil.ProfTree):
        self.profTree = profTree
        self.root = self.createDAG(profTree.root, None)
        
    def createDAG(self, rootNode: profUtil.Node, parent: ElementInDAG= None) -> ElementInDAG:
        element = ElementInDAG(rootNode, parent)
        for cname, cnode in rootNode.children.items():
            element.children[cname] = self.createDAG(cnode, element)
        return element
    
    def profAgg(self):
        self.aggProfPool: Dict[str, AggProf] = {}
        stack = [self.root]
        while stack:
            ele = stack.pop()
            # If control state or paral, no aggregation
            if ele.nType == 'Paral' or ele.nType == 'SwitchFlow':
                for c in ele.children.values():
                    stack.append(c)
            # If next is paral, no aggregation
            elif ele.children:
                c = list(ele.children.values())[0]
                if c.nType != 'Paral':
                    aggProf = ele.profAggNext(c)
                    self.aggProfPool[ele.name] = aggProf
                stack.append(c)
    
    def createOptions(self):
        options = []
        options.append((0, 0, copy.deepcopy(self.root)))
        while self.aggProfPool:
            eleName = max(self.aggProfPool, key=lambda k: self.aggProfPool[k].r)
            aggProf = self.aggProfPool[eleName]
            curEle: ElementInDAG = aggProf.ele
            parent = curEle.parent
            child = list(curEle.children.values())[0]
            
            # Delete stale prof in pool, cur, parent, child
            del self.aggProfPool[curEle.name]
            if parent and parent.name in self.aggProfPool:
                del self.aggProfPool[parent.name]
            if child.name in self.aggProfPool:
                del self.aggProfPool[child.name]

            # Aggregation current with next 
            curEle.aggWithNext()
            lastRT, lastIROH, _ = options[-1]
            options.append((lastRT+aggProf.reducedTime, lastIROH+aggProf.increasedROH, copy.deepcopy(self.root)))

            # Generate agg prof for neighbours
            if curEle.nType != 'SwitchFlow' and curEle.nType != 'Paral' and curEle.children and list(curEle.children.values())[0].nType != 'Paral':
                self.aggProfPool[curEle.name] = curEle.profAggNext(list(curEle.children.values())[0])
            if curEle.parent and parent.nType != 'SwitchFlow' and parent.nType != 'Paral':
                self.aggProfPool[curEle.parent.name] = curEle.parent.profAggNext(curEle)
        return options

                
class StateMachine:
    inputs: Inputs
    def __init__(self, 
        app: primitives.Application, 
        inputs: Inputs = None):
        self.app = app
        self.inputs = inputs

    def run(self):
        rels = {}
        app = copy.deepcopy(self.app)
        for case in self.inputs.inputCase:
            r = app.run(self.inputs.getInput(case), NO_PROF, None)
            rels[case] = r
        return rels

    def prof(self, profMode, outpath):
        rels, prfTrees = {}, {}
        rootDir = outpath or 'tmp'
        rootDir = os.path.join(rootDir, self.app.name)
        for case in self.inputs.inputCase:
            app = copy.deepcopy(self.app)
            outDir = os.path.join(rootDir, case)
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            rel = app.run(self.inputs.getInput(case), profMode, outDir)
            rels[case] = rel
            profTree = profUtil.ProfTree(app)
            profTree.createTree(outDir)
            profTree.genGraphviz(outDir)
            prfTrees[case] = profTree
        mergedTree= self.mergeProfTree(name='MergedProf', prfTrees=prfTrees, profMode=profMode)
        mergedTree.genGraphviz(rootDir)
        return rels, prfTrees, mergedTree 
    
    def copyBase(self, target: profUtil.Node, source: profUtil.Node):
        target.inMap = source.inMap
        target.outMap = source.outMap
        target.nexts = source.nexts
        target.paral = source.paral
        target.nType = source.nType
        target.funcName = source.funcName
        target.stackerTargets = source.stackerTargets
        target.unstackerTargets = source.unstackerTargets
        target.unstkMap = source.unstkMap

    def maxMerge(self, nodes: List[profUtil.Node]) -> Tuple[profUtil.Node, Dict[str, List[profUtil.Node]]]:
        newNode = profUtil.Node(nodes[0].name)
        self.copyBase(newNode, nodes[0])
        maxInSize = {}
        maxOutSize = {}
        maxNewAlloc = 0 
        maxCallTime = 0
        childrenNodes: Dict[str, List[profUtil.Node]] = {}
        for n in nodes:
            if n.name != newNode.name:
                print("Combining differnt node types, error")
                exit(2)
            for k, v in n.inSize.items():
                maxInSize[k] = max(maxInSize.get(k, 0), v)
            for k, v in n.outSize.items():
                maxOutSize[k] = max(maxOutSize.get(k, 0), v)
            maxNewAlloc = max(maxNewAlloc, n.newAlloc)
            maxCallTime = max(maxCallTime, n.callTime)
            for k, v in n.children.items():
                if k not in childrenNodes:
                    childrenNodes[k] = [v]
                else:
                    childrenNodes[k].append(v)
        newNode.newAlloc = maxNewAlloc
        newNode.inSize = maxInSize
        newNode.outSize = maxOutSize
        newNode.callTime = maxCallTime
        return newNode, childrenNodes

    def meanMerge(self, nodes: List[profUtil.Node]) -> Tuple[profUtil.Node, Dict[str, List[profUtil.Node]]]:
        newNode = profUtil.Node(nodes[0].name)
        self.copyBase(newNode, nodes[0])
        meanInSize = {}
        meanOutSize = {}
        meanNewAlloc = 0 
        meanCallTime = 0
        childrenNodes: Dict[str, List[profUtil.Node]] = {}
        for n in nodes:
            if n.name != newNode.name:
                print("Combining differnt nodes, error")
                exit(2)
            for k, v in n.inSize.items():
                meanInSize[k] = meanInSize.get(k, 0) + v
            for k, v in n.outSize.items():
                meanOutSize[k] = meanOutSize.get(k, 0) + v
            meanNewAlloc += n.newAlloc
            meanCallTime += n.callTime
            for k, v in n.children.items():
                if k not in childrenNodes:
                    childrenNodes[k] = [v]
                else:
                    childrenNodes[k].append(v)
        newNode.newAlloc = meanNewAlloc
        newNode.inSize = meanInSize
        newNode.outSize = meanOutSize
        newNode.callTime = meanCallTime
        return newNode, childrenNodes

    def mergeProfTree(self, name, prfTrees: Dict[str, profUtil.ProfTree], profMode) -> profUtil.ProfTree:
        relTree = profUtil.ProfTree(app=None, name=name)
        for pt in prfTrees.values():
            relTree.inPoolSet.update(pt.inPoolSet)
            relTree.outPoolSet.update(pt.outPoolSet)
            relTree.totalMem = max(relTree.totalMem, pt.totalMem)
            relTree.totalTime = max(relTree.totalTime, pt.totalTime)
        m = self.maxMerge
        if profMode is MEAN_PROF:
            m = self.meanMerge
        if profMode is MAX_PROF:
            m = self.maxMerge

        def recurHelper(prfNodes: List[profUtil.Node], mergeMethod) -> profUtil.Node:
            newNode, childDict = mergeMethod(prfNodes)
            for cname, cnodes in childDict.items():
                newNode.children[cname] = recurHelper(cnodes, mergeMethod)
            return newNode
        
        roots = []
        for t in prfTrees.values():
            roots.append(t.root)
        mergedRoot = recurHelper(roots, m)
        relTree.root = mergedRoot

        return relTree
    
    def createDagOptions(self, profTree: profUtil.ProfTree):
        dag = DagOpt(profTree)
        dag.profAgg()
        options = dag.createOptions()
        self.options = options
        return options

    @DeprecationWarning
    def configObjPoolHybrid(self, optionId):
        _, _, n = self.options[optionId]
        peak_pool_mem = 0
        # Post order traversal
        def recHelper(n: ElementInDAG) -> VirtualObjPool:
            nonlocal peak_pool_mem
            if len(n.children) == 0:
                return n.updateObjPool({})
            pools = {cname: recHelper(c) for cname, c in n.children.items()}
            updatedPool = n.updateObjPool(pools)
            # Update remote peak size
            curMem = 0
            for obj in updatedPool.pool.values():
                if obj.fromRemote:
                    curMem += obj.size
            peak_pool_mem = max(peak_pool_mem, curMem)
            return updatedPool
        
        poolAtBegin = recHelper(n)
        inPool = set()
        for cname, vobj in poolAtBegin.pool.items():
            if vobj.fromRemote:
                print("Should have no remote req at beggining")
                exit(1)
            inPool.add(cname)
        if inPool != n.inMap:
            print("Should have same input map at app entry")
        return n, peak_pool_mem
    
    def configObjPoolRemoteOnly(self, optionId):
        reducedTime, increasedOverhead, n = self.options[optionId]
        peak_pool_mem = 0
        def recHelper(n: ElementInDAG) -> VirtualObjPool:
            nonlocal peak_pool_mem
            if len(n.children) == 0:
                return n.updateObjPoolR({})
            pools = {cname: recHelper(c) for cname, c in n.children.items()}
            updatedPool = n.updateObjPoolR(pools)
            peak_pool_mem = max(peak_pool_mem, updatedPool.getRemoteSize())
            return updatedPool

        poolAtBegin = recHelper(n)
        inPool = set(poolAtBegin.pool.keys())
        if inPool != n.inMap:
            print("Should have same input map at app entry")

        # Detailed download, upload, clearRemote rules
        
        return n, peak_pool_mem