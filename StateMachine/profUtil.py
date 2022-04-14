from typing import Dict, Set, List

from .primitives import Lambda, Paral, SwitchFlow, Application
from .parser_util import util
import graphviz
import os
from sys import exit
os.environ["PATH"] += '/usr/local/Cellar/graphviz/2.50.0/bin'


class Node:
    def __init__(self, name, 
                 newAlloc=0, 
                 nType='default-type',
                 inPool='default_Pool',
                 outPool='default_Pool',
                 funcName=None,
                 children=None,
                 inMap=None,
                 inSize=None,
                 outMap=None,
                 outSize=None,
                 callTime=0,
                 nexts=None,
                 paral=1,
                 stackerTargets=None,
                 unstackerTargets=None,
                 unstkMap=None
                 ):
        self.name = name
        self.newAlloc = newAlloc
        self.nType = nType
        self.inPool = inPool
        self.outPool = outPool
        self.funcName = funcName
        self.children: Dict = children or {}
        self.inMap = inMap or set()
        self.inSize = inSize or {}
        self.outMap = outMap or set()
        self.outSize = outSize or {}
        self.callTime = callTime
        self.nexts = nexts or []
        self.paral = paral
        self.stackerTargets=stackerTargets
        self.unstackerTargets = unstackerTargets
        self.unstkMap = unstkMap

    def __repr__(self, level=0) -> str:
        if self.nType == 'Lambda':
            ret = '\t' * level + f'{self.name} \n' \
                + '\t' * level + f'InputSize: {self.inSize} \n' \
                + '\t' * level + f'New allocation: {self.newAlloc} \n' \
                + '\t' * level + f'Call time: {self.callTime} \n' \
                + '\t' * level + f'OutSize: {self.outSize}\n'
        if self.nType == 'SwitchFlow':
            ret = '\t' * level + f'{self.name} \n' \
                + '\t' * level + f'InputSize: {self.inSize} \n' \
                + '\t' * level + f'New allocation: {self.newAlloc} \n' \
                + '\t' * level + f'Call time: {self.callTime} \n' \
                + '\t' * level + f'OutSize: {self.outSize}\n'
        if self.nType == 'Paral':
            ret = '\t' * level + f'{self.name} \n' \
                + '\t' * level + f'level of Parallelism: {self.paral} \n' \
                + '\t' * level + f'max InputSize: {self.inSize} \n' \
                + '\t' * level + f'max New allocation: {self.newAlloc} \n' \
                + '\t' * level + f'total Call time: {self.callTime} \n' \
                + '\t' * level + f'max OutSize: {self.outSize}\n'
        return ret

    def __str__(self, level=0) -> str:
        ret = self.__repr__(level)
        for child in self.children.values():
            ret += child.__str__(level+1)
        return ret

    def graphvizMenifest(self) -> str:
        name = self.name
        insize = '\\n'.join([f'{k}: {v}' for k, v in self.inSize.items()])
        newAlloc = str(self.newAlloc)
        callTime = str(self.callTime)
        outsize = '\\n'.join([f'{k}: {v}' for k, v in self.outSize.items()])
        baseInfo = [f'<inSize> {insize}', f'<newAlloc> {newAlloc} B',
                    f'<callTime> {callTime} s', f'<outSize> {outsize}']
        if self.nType == 'Paral':
            baseInfo.append('<Paral> ' + 'Parallelism: ' + str(self.paral))
        return f'<name> {name} | {{' + ' | '.join(baseInfo) + '}}'


class ProfTree:
    def __init__(self, app: Application = None, name=None):
        if app:
            self.name = app.name
            self.appRoot = app.root
            self.root = self.getNode(self.appRoot)
            self.totalTime = app.totalTime
            self.totalMem = 0
            self.inPoolSet: Set = set(app.inPools.values())
            self.outPoolSet: Set = set(app.outPools.values())
        else:
            self.name = name
            self.appRoot = None
            self.root = None
            self.totalTime = 0
            self.totalMem = 0
            self.inPoolSet = set()
            self.outPoolSet = set()

    def getType(self, appNode):
        if isinstance(appNode, Lambda):
            return 'Lambda'
        if isinstance(appNode, SwitchFlow):
            return 'SwitchFlow'
        if isinstance(appNode, Paral):
            return 'Paral'
        else:
            return 'Unknown'

    def getNode(self, appNode) -> Node:
        nType = self.getType(appNode)
        profNode = Node(
            name=appNode.name,
            newAlloc=0,
            nType=nType,
            inPool=appNode.inPool,
            outPool=appNode.outPool,
            funcName=appNode.funcName,
            children={},
            inMap=appNode.inputMap,
            inSize=appNode.inputSize,
            outMap=appNode.outputMap,
            outSize=appNode.outputSize,
            callTime=appNode.callTime,
            nexts=[],
            paral=1
        )
        if nType == 'Lambda':
            if appNode.stacker:
                profNode.stackerTargets = appNode.stacker.targets
            if appNode.unstacker:
                profNode.unstackerTargets = appNode.unstacker.targets
        if nType == 'SwitchFlow':
            profNode.nexts = [n.name for n in appNode.nexts]
        if nType == 'Paral':
            profNode.paral = appNode.parallelism
            profNode.stackerTargets = appNode.targets
            profNode.unstkMap = appNode.ustkMap
        return profNode

    def createTree(self, rootDir):
        stack = [(self.root, self.appRoot)]
        for path, dirs, fs in os.walk(os.path.join(rootDir, self.root.name)):
            profNode, appNode = stack.pop()
            # if has profile
            for fname in fs:
                if fname == 'peak-memory.prof':
                    profNode.newAlloc = util.getTotalMem(
                        os.path.join(path, fname))
                    self.totalMem = max(self.totalMem, sum(appNode.inputSize.values())+profNode.newAlloc)
                    break
            # for d in dirs[::-1]:
            if len(dirs) > 1:
                print('Current does not support n-nary flow')
                exit(1)
            if appNode.next:
                if dirs[0] != appNode.next.name:
                    print(f'Contain invalid profile {dirs[0]}')
                    exit(1)
                newNode = self.getNode(appNode.next)
                profNode.children[dirs[0]] = newNode
                stack.append((newNode, appNode.next))

    def __str__(self) -> str:
        return self.root.__str__(0)

    def genGraphviz(self, outpath):
        g = graphviz.Digraph(self.name, format='pdf')
        with g.subgraph(name='cluster_pools') as p:
            p.attr('node', shape='doublecircle', color='lightgrey')
            for pool in self.inPoolSet.union(self.outPoolSet):
                p.node(pool)
        with g.subgraph(name='cluster_states') as s:
            stack = [self.root]
            s.attr('node', shape='circle')
            while stack:
                node = stack.pop()
                s.node(node.name)
                for c in node.children.values():
                    stack.append(c)
                    s.edge(node.name, c.name)
        stack = [self.root]
        while stack:
            node = stack.pop()
            g.edge(node.inPool, node.name)
            g.edge(node.name, node.outPool)
            for c in node.children.values():
                stack.append(c)
        gpath = os.path.join(outpath, f'{self.name}-data-gv')
        graphout = g.render(gpath, view=False)
        print(f'View state machine data graph at {graphout}')

        appProf = f'<name> {self.name} | {{<TotalSize> {self.totalMem} B | <TotalTime> {self.totalTime} s}}'
        cg = graphviz.Digraph(self.name+'prof', format='pdf')
        cg.attr('node', shape='record')
        cg.node(self.name, appProf)
        stack = [self.root]
        while stack:
            node = stack.pop()
            cg.node(node.name, node.graphvizMenifest())
            for c in node.children.values():
                stack.append(c)
                cg.edge(node.name+':name', c.name+':name')
        gpath = os.path.join(outpath, f'{self.name}-prof-gv')
        graphout = cg.render(gpath, view=False)
        print(f'View state machine prof graph at {graphout}')
