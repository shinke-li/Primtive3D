import operator
import time
import os
import random 
import numpy as np
from scipy.spatial.distance import cdist

import constructive_tree.primitive_pymesh as pp

PRINT = False
def myprint(*args, **kwargs):
    if PRINT:
        print(*args, **kwargs)

def operation_func():
    return [operator.add,operator.mul,operator.sub]

def primitive_func():
    l = ["Sphere",
         "Cylinder", 
         "Box",
         "Torus",
         "Cone",]
    return [getattr(pp, a) for a in l]

def random_relation(m1, m2):
    pt1= m1.sample_points(1)
    pt2 = m2.sample_points(1)
    m2.rigid_transform(offset=np.array(pt1)-np.array(pt2))
    
def random_relation_truncate(m1, m2, candidates=100, low_percentile=10, high_percentile=90):
    pt1= m1.sample_points(candidates)[0]
    pt2 = m2.sample_points(candidates)[0]
    d = cdist(pt1, pt2, 'euclidean')
    p_low = np.percentile(d, low_percentile)
    p_high = np.percentile(d, high_percentile)
    ind1, ind2 = np.where((d>=p_low)&(d<=p_high))
    i = random.randint(0, len(ind1) - 1)
    ind = (ind1[i], ind2[i])
    m2.rigid_transform(offset=np.array(pt1[ind[0]])-np.array(pt2[ind[1]]))
    
    
def random_init_primitive(m1, low=0.5, high=2.0):
    axis = np.array([pp.gaussian_dist(0, 1) for _ in range(3)])
    axis = axis / np.linalg.norm(axis)
    angle = random.random() *  np.pi 
    theta_l, theta_h = np.arctan(low), np.arctan(high)
    scale = np.tan(random.uniform(theta_l, theta_h))
    #m1.normalize()
    m1.scale(scale)
    m1.rigid_transform(axis=axis,angle=angle)


class CSGNodeValue(object):
    OP_DICT = {0:"union", 1:"intersection", 2:"substract"}
    PRM_DICT = {0:"Sphere", 
                1:"Cylinder", 
                2:"Box",
                3:"Torus",
                4:"Cone",}
    PRM_WEIGHTS = [4, 7, 9, 6, 7]
    def __init__(self,node_type="op", value=None): #"op" or "ps"
        self.value = value
        self.node_type = node_type
        self.mesh = None 
        
    @staticmethod
    def random(node_type="op", level_loc=1.0):
        if node_type == "op":
            union_w,inter_w,sub_w = 1.0, 0., 0.
            value = random.choices(list(CSGNodeValue.OP_DICT.keys()), weights=[union_w,inter_w,sub_w])[0]
        elif  node_type == "ps":
            value = random.choices(list(CSGNodeValue.PRM_DICT.keys()), weights=CSGNodeValue.PRM_WEIGHTS)[0]
        else:
            raise AttributeError
        return CSGNodeValue(node_type=node_type,
                            value=value)
            
    def eval_(self, *args):
        if self.node_type == "op":
            m1 = args[0].mesh
            m2 = args[1].mesh
            random_relation(m1, m2)
            #random_relation_truncate(m1, m2)
            op_dict = operation_func()
            self.mesh = op_dict[self.value](m1, m2)

        else:
            prm_dict = primitive_func()
            self.mesh = prm_dict[self.value].random()
            random_init_primitive(self.mesh)
        
        
    def __repr__(self):
        return ":".join([str(self.node_type), str(self.value)])
            
class CSGNode(object):
    def __init__(self, index ,value=None):
        self.index = index
        self.left = None 
        self.right = None
        if value is None:
            self.value = None
        else:
            self.value = CSGNodeValue.random(node_type=value)
        
    def isExternal(self):
        return self.index%2 == 0
            
        
    def getMesh(self):
        return self.value.mesh
    
    def assign_value(self):
        root = self 
        if not root: return []
        queue = [root]
        res = []
        while queue:
            res.append(queue)
            ll = []
            for node in queue:
                if node.left:
                    ll.append(node.left)
                if node.right:
                    ll.append(node.right)
            queue = ll
            
        level_total = len(res) - 2 # remove last level
        for i, level in enumerate(res[::-1]):
            if level_total == 0:
                level_loc = 1.0
            else:
                level_loc = 1 - (float(i) - 1 ) / level_total
            for p in level:
                if p.isExternal():
                    p.value = CSGNodeValue.random(node_type="ps")
                else:
                    p.value = CSGNodeValue.random(node_type="op",level_loc=level_loc)
    
    def csg_eval(self):
        root = self 
        if not root: return []
        queue = [root]
        res = []
        while queue:
            res.append(queue)
            ll = []
            for node in queue:
                if node.left:
                    ll.append(node.left)
                if node.right:
                    ll.append(node.right)
            queue = ll
        for level in res[::-1]:
            for p in level:
                if p.isExternal():
                    p.value.eval_()
                else:
                    p.value.eval_(p.left.value, p.right.value)
        
    
    def print_tree(self, print_term="index"):
        print_term = print_term
        def printTree(node, level=0):
            if node != None:
                printTree(node.left, level + 1)
                print(' ' * 4 * level + '--->', str(getattr(node, print_term)))
                printTree(node.right, level + 1)
        printTree(self)


def random_link(internal=3):
    n = 0
    L = [0 for _ in range(2 * internal + 1)]
    L[1] = 0
    while True:
        x = random.randint(0, 4 * n + 1)
        n += 1
        b = x%2 
        k = int(x / 2.0)
        L[2 * n - b] = 2*n
        L[2 * n - 1 + b] = L[k]
        L[k] = 2* n - 1 
        if n==internal: break 
    return L 

def parse_link(l, c=0):
    ind = l[c] 
    node = CSGNode(ind)
    if ind%2 == 0: #exteranl or not
        return node
    k = int((ind + 1) / 2)
    n = (len(l) -1) / 2
    if 1<=k<=n:
        node_left = parse_link(l, c=2*k - 1)
        node_right = parse_link(l, c=2*k )
        node.left = node_left
        node.right = node_right
    return node

def cal_dof(csg_tree):
    def front_stack(root):
        if root == None:
            return
        myStack = []
        all_stack = []
        node = root
        while node or myStack:
            while node: 
                myStack.append(node)
                all_stack.append(node)
                node = node.left
            node = myStack.pop()            
            node = node.right
        return all_stack
    stack = front_stack(csg_tree)
    dof = [node.value.mesh.cal_dof() for node in stack if node.isExternal()]
    return sum(dof) + (len(dof) - 1) * 3


class Timer(object):
    def __init__(self):
        self.init_tic = time.time()
        self.tic = self.init_tic
    def print_time(self,s=""):
        toc = time.time()
        myprint(s, toc - self.tic, toc - self.init_tic)
        self.tic = toc

def random_csg_tree(node_num=3, print_tree=False):
    if node_num==1:
        prm_dict = primitive_func()
        node = CSGNode(0)
        ind = random.randint(0, len(prm_dict) - 1)
        node.value = CSGNodeValue(node_type='ps', value=ind)
        mesh = prm_dict[ind].random()
        random_init_primitive(mesh)
        node.value.mesh = mesh
        return node, node.value.mesh.cal_dof()
    node = parse_link(random_link(node_num-1))
    node.assign_value()
    if print_tree:
        node.print_tree("value")
    node.csg_eval()
    return node, cal_dof(node)


def random_csg_tree_gen_and_save(save_name, node_num=3, name_with_dof=True): 
    node, dof = random_csg_tree(node_num=node_num)
    if name_with_dof:
        file_name_split = os.path.splitext(save_name)
        save_name = file_name_split[0] + "_" + str(dof) + file_name_split[1]
        print(save_name)
    node.getMesh().save(save_name)
    return save_name

