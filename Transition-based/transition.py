#!/usr/bin/env python
# coding: utf-8

# In[136]:


from semgraph import DependencyGraph, EntityNode, RelationArc
import pandas as pd
import typing


# In[137]:


data = pd.read_csv('data/data.csv')


# In[138]:


line = data['dependency'][0]
graph = DependencyGraph()
line.split('\n')
for arc_info in list(filter(None, line.split('\n'))):
    arc_info = arc_info.split()
    child_id = int(arc_info[0])
    father_id = int(arc_info[3])
    # 添加child节点
    if graph.get_node(child_id) is not None:
        child = graph.get_node(child_id)
        child.add_father(father_id)
    else:
        child = EntityNode(child_id, arc_info[1], arc_info[2])
        child.add_father(father_id)
        graph.add_node(child)
    # 添加father节点   
    if graph.get_node(father_id) is not None:
        father = graph.get_node(father_id)
        father.add_child(child_id)
    else:
        father = EntityNode(father_id, arc_info[4], arc_info[5])
        father.add_child(child_id)
        graph.add_node(father)
    # 添加关系弧
    relation = arc_info[6]
    arc = RelationArc(father_id, child_id, relation)
    graph.add_arc(arc)

# test: 打印节点，孩子节点和父亲节点
graph.add_node(EntityNode(0, '_', 'ROOT'))
node_list = graph.get_node_list()
id_list = []
for node in node_list:
    print(node, graph.get_node(node).get_child(), graph.get_node(node).get_father())


# In[139]:


test_graph = DependencyGraph()
node_0 = EntityNode(0, '_', 'ROOT')
node_0.add_child(3)
test_graph.add_node(node_0)
test_graph.add_arc(RelationArc(0, 3, 'ROOT'))

node_1 = EntityNode(1, '_', '他')
node_1.add_father(3)
node_1.add_father(5)
test_graph.add_node(node_1)
test_graph.add_arc(RelationArc(3, 1, 'Agt'))
test_graph.add_arc(RelationArc(5, 1, 'Agt'))

node_2 = EntityNode(2, '_', '将')
node_2.add_father(3)
test_graph.add_node(node_2)
test_graph.add_arc(RelationArc(3, 2, 'mTime'))

node_3 = EntityNode(3, '_', '离开')
node_3.add_father(0)
node_3.add_child(1)
node_3.add_child(2)
node_3.add_child(4)
node_3.add_child(5)
test_graph.add_node(node_3)
test_graph.add_arc(RelationArc(3, 4, 'Lini'))
test_graph.add_arc(RelationArc(3, 5, 'eSucc'))

node_4 = EntityNode(4, '_', '北京')
node_4.add_father(3)
test_graph.add_node(node_4)

node_5 = EntityNode(5, '_', '去')
node_5.add_father(3)
node_5.add_child(1)
node_5.add_child(6)
test_graph.add_node(node_5)
test_graph.add_arc(RelationArc(5, 6, 'Lfin'))

node_6 = EntityNode(6, '_', '上海')
node_6.add_father(5)
test_graph.add_node(node_6)

test_node_list = test_graph.get_node_list()
for node in test_node_list:
    print(node, test_graph.get_node(node).get_child(), test_graph.get_node(node).get_father())


# In[140]:


def left_reduce(g, σ: typing.List[int], δ: typing.List[int], β: typing.List[int], A: typing.List[RelationArc]):
    wi = σ.pop()
    wj = β[-1]
    arc = g.get_arc(wj, wi)
    A.append(arc)
    print('LEFT-REDUCE', σ, δ, β, g.get_node(arc.get_head()).get_text(), '->', g.get_node(arc.get_tail()).get_text())
    return σ, δ, β, A


# In[141]:


def right_shift(g, σ: typing.List[int], δ: typing.List[int], β: typing.List[int], A: typing.List[RelationArc]):
    wi = σ[-1]
    wj = β.pop()
    while δ:
        σ.append(δ.pop(0))
    σ.append(wj)
    arc = g.get_arc(wi, wj)
    A.append(arc)
    print('RIGHT-SHIFT', σ, δ, β, g.get_node(arc.get_head()).get_text(), '->', g.get_node(arc.get_tail()).get_text())
    return σ, δ, β, A


# In[142]:


def no_shift(σ: typing.List[int], δ: typing.List[int], β: typing.List[int], A: typing.List[RelationArc]):
    wj = β.pop()
    while δ:
        σ.append(δ.pop(0))
    σ.append(wj)
    print('NO-SHIFT', σ, δ, β)
    return σ, δ, β, A


# In[143]:


def no_reduce(σ: typing.List[int], δ: typing.List[int], β: typing.List[int], A: typing.List[RelationArc]):
    σ.pop()
    print('NO-REDUCE', σ, δ, β)
    return σ, δ, β, A


# In[144]:


def left_pass(g, σ: typing.List[int], δ: typing.List[int], β: typing.List[int], A: typing.List[RelationArc]):
    wi = σ.pop()
    wj = β[-1]
    δ.append(wi)
    arc = g.get_arc(wj, wi)
    A.append(arc)
    print('LEFT-PASS', σ, δ, β, g.get_node(arc.get_head()).get_text(), '->', g.get_node(arc.get_tail()).get_text())
    return σ, δ, β, A


# In[145]:


def right_pass(g, σ: typing.List[int], δ: typing.List[int], β: typing.List[int], A: typing.List[RelationArc]):
    wi = σ.pop()
    wj = β[-1]
    δ.append(wi)
    arc = g.get_arc(wi, wj)
    A.append(arc)
    print('RIGHT-PASS', σ, δ, β, g.get_node(arc.get_head()).get_text(), '->', g.get_node(arc.get_tail()).get_text())
    return σ, δ, β, A


# In[146]:


def no_pass(σ: typing.List[int], δ: typing.List[int], β: typing.List[int], A: typing.List[RelationArc]):
    wi = σ.pop()
    δ.append(wi)
    print('NO-PASS', σ, δ, β)
    return σ, δ, β, A


# In[147]:


def transition(g: DependencyGraph):
    σ, δ, β, A = [], [], [], []
    node_list = g.get_node_list()
    for node in node_list:
        if node == 0:
            σ.append(node)
        else:
            β.append(node)
    β.reverse()
    print('Initialization', σ, δ, β)
#     root = 0
#     σ.append(root)

    while β:
        wj = β[-1]
        wi = σ[-1]
        shift = True
        reduce = True
        #shift
        for i in range(len(σ) - 1):
            if g.get_arc(σ[i], wj) or g.get_arc(wj, σ[i]):
                shift = False
                break
        if shift:
            if g.get_arc(wi, wj):
                σ, δ, β, A = right_shift(g, σ, δ, β, A)
                continue
            elif g.get_arc(wj, wi):
                pass
            else:
                σ, δ, β, A = no_shift(σ, δ, β, A)
                continue
        # reduce
        if not g.get_node(wi).get_father():
            reduce = False
        else:
            for i in range(len(β) - 1):
                if g.get_arc(wi, β[i]) or g.get_arc(β[i], wi):
                    reduce = False
                    break
        if reduce:
            if g.get_arc(wi, wj):
                pass
            elif g.get_arc(wj, wi):
                σ, δ, β, A = left_reduce(g, σ, δ, β, A)
                continue
            else:
                σ, δ, β, A = no_reduce(σ, δ, β, A)
                continue
        # pass
        if g.get_arc(wi, wj):
            σ, δ, β, A = right_pass(g, σ, δ, β, A)
        elif g.get_arc(wj, wi):
            σ, δ, β, A = left_pass(g, σ, δ, β, A)
        else:
            σ, δ, β, A = no_pass(σ, δ, β, A)
    print(σ, δ, β)
transition(test_graph)


# In[ ]:


transition(graph)


# In[124]:


try:  
  get_ipython().system('jupyter nbconvert --to python transition.ipynb')
  # python即转化为.py，script即转化为.html
  # file_name.ipynb即当前module的文件名
except:
  pass


# In[ ]:




