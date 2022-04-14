import ast
from collections import deque
import pydot
import os

def top_level_functions(body):
    return [f for f in body if isinstance(f, ast.FunctionDef)]

def parse_ast(filepath):
    with open(filepath, "rt") as file:
        return ast.parse(file.read()) 

def parse_funcs(filepath):
    print(filepath)
    tree = parse_ast(filepath)
    funcs = top_level_functions(tree.body)
    for func in funcs:
        print("  %s" % func.name)
    return funcs

def findMainCaller(grammar: ast.Module):
    for node in ast.walk(grammar):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == "main":
                    return node.lineno

def findMainFuncDef(grammar: ast.Module):
    for node in ast.walk(grammar):
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            return node

def getTotalMem(profPath):
    mem = 0
    with open(profPath, "rt") as f:
        profLines = f.read().splitlines()
    for line in profLines:
        _, size = line.rsplit(' ', 1)
        mem += int(size)
    return mem

class TrieNode:
    def __init__(self, name="", parent="", size=0):
        self.children = dict()
        self.size = size
        self.name = name
        self.parent = parent

    def __repr__(self) -> str:
        name_size = 'name: {}, size: {}\n'.format(self.name, self.size)
        children = '[ ' + '\n '.join([c.name + str(c.size) for c in self.children.values()]) + ' ]\n'
        return name_size + children

class TraceTrie:
     
    # Trie data structure class
    def __init__(self, filepath: str):
        self.root = self.getNode("root")
        self.filepath = filepath 
 
    # Returns new trie node (initialized to NULLs)
    def getNode(self, name, parent="", size=0):
        return TrieNode(name=name, parent=parent, size=size)
    
    # descard deeper frame by filename
    def insert(self, profLine: str, filefilter = ""):
        trs, size = profLine.rsplit(' ', 1)
        traces = trs.split(';', -1)
        stk = self.root
        size = int(size)
        stk.size += size
        for i in range(len(traces)):
            trace = traces[i]
            if filefilter:
                filepath = trace.split(':', 1)[0]
                if filepath != filefilter:
                    break
            if trace not in stk.children:
                newNode = self.getNode(trace, stk.name)
                stk.children[trace] = newNode
            stk = stk.children[trace]
            stk.size += size

    # def insertTrace(self, profLIne: str):
    #     trs, size = profLIne.rsplit(' ', )

    def getMainProf(self, mainCaller: str):
    #   BFS
        stk = deque([self.root])
        while stk:
            trace = stk.popleft()
            if mainCaller in trace.children:
                return trace.children[mainCaller]
            else:
                for node in trace.children.values():
                    stk.append(node)
        return None 
    
    def plot(self, output='tmp'):
        filename = os.path.splitext(os.path.basename(self.filepath))[0]
        graph = pydot.Dot("Trie of {}".format(filename), graph_type="digraph", strict=True)
        nodes = deque([self.root])
        while nodes:
            node = nodes.popleft()
            for c in node.children.values():
                graph.add_edge(pydot.Edge(node.name+str(node.size), c.name + str(c.size)))
                nodes.append(c)
        opath = f'{output}/TracePlot'
        if not os.path.exists(opath):
            os.makedirs(opath)
        graph.write_png(f'{opath}/{filename}.png')
        return f'{opath}/{filename}.png'

    # def getCallerTotoal(self, lineno: int, caller: str):
    #     # BFS
    #     s = "{}:{} ({})".format(self.filename, lineno, caller)
    #     stk = deque([self.root])
    #     while stk:
    #         trace = stk.popleft()
    #         if s in trace.children:
    #             return trace.children[s]
    #         else:
    #             for node in trace.children.values():
    #                 stk.append(node)
    #     return None

    # def getCalleeTotoal(self, lineno: int, callee: str):
    #     # BFS
    #     s = "{}:{} ({})".format(self.filename, lineno, callee)
    #     stk = deque([self.Rroot])
    #     while stk:
    #         trace = stk.popleft()
    #         if s in trace.children:
    #             return trace.children[s]
    #         else:
    #             for node in trace.children.values():
    #                 stk.append(node)
    #     return None





    # def search(self, filename: str, funcname: str, lineno: int):
    #     key = '{}:{} ({})'.format(filename, lineno, funcname)
    #     forwardTrace = 
    
