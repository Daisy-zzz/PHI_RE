#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import typing


# In[4]:


"""
'dependency' cols example for each text:

child-node	child-type	child-text	father-node 	father-type     	 relation
        2	E340ca71c	麻木   	         1  	 E320ca3f6 	左上肢 	R742a31d5
        4	E340ca71c	言语不清	       3    	E1ceb2bd7	发作性 	R742a31d5
        4	E340ca71c	言语不清	       5    	E1deb2d6a	5天  	R742a31d5
        9	E310ca263	改善循环	       8    	E1deb2d6a	入院后 	R742a31d5
        ......
"""


# In[5]:


class EntityNode(object):
    def __init__(self, id: int, type: str, text: str) -> None:
        self.id = id
        self.type = type
        self.text = text
        self.father = list()
        self.child = list()
        
    
    def get_id(self) -> int:
        return self.id
    
    
    def get_type(self) -> str:
        return self.type
    
    
    def get_text(self) -> str:
        return self.text
    
    
    def add_child(self, child: int) -> None:
        self.child.append(child)
        
        
    def add_father(self, father: int) -> None:
        self.father.append(father)
        
        
    def get_child(self) -> typing.List[int]:
        return self.child
    
    
    def get_father(self) -> typing.List[int]:
        return self.father


# In[6]:


class RelationArc(object):
    def __init__(self, father: int, child: int, relation: str) -> None:
        self.head = father
        self.tail = child
        self.relation = relation
    
    
    def get_head(self) -> int:
        return self.head
    
    
    def get_tail(self) -> int:
        return self.tail
    
    
    def get_relation(self) -> str:
        return self.relation


# In[7]:


class DependencyGraph(object):
    def __init__(self) -> None:
        self.nodedict = dict()
        self.arclist = list()
    
    
    def add_node(self, node: EntityNode) -> None:
        self.nodedict[node.get_id()] = node
        
    
    def get_node_list(self) -> typing.List[int]:
        node_list = []
        for node_id in self.nodedict.keys():
            node_list.append(node_id)
        node_list.sort()
        return node_list
    
    
    def get_node(self, id: int) -> EntityNode:
        if id in self.nodedict.keys():
            return self.nodedict.get(id)
        return None
        # 若不存在，返回空值
        
        
    def add_arc(self, arc: RelationArc) -> None:
        if arc not in self.arclist:
            self.arclist.append(arc)
        
    
    def get_arc(self, head: int, tail: int) -> RelationArc:
        for arc in self.arclist:
            if head == arc.get_head() and tail == arc.get_tail():
                return arc
        return None
        # 若不存在，返回空值


# In[8]:


try:  
  get_ipython().system('jupyter nbconvert --to python semgraph.ipynb')
  # python即转化为.py，script即转化为.html
  # file_name.ipynb即当前module的文件名
except:
  pass


# In[ ]:




