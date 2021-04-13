from grammar.vertex import *
from grammar.tree import *
from copy import deepcopy
class OccurList(object):

    def __init__(self):
        self.occur_list = []
        self.tid2vertexList = dict()

    def add(self, template_id, original_vertex):
        self.occur_list.append((template_id, original_vertex))
        if template_id in self.tid2vertexList:
            self.tid2vertexList[template_id].append(original_vertex)
        else:
            self.tid2vertexList[template_id] = [original_vertex]

    def size(self):
        return len(self.occur_list)

    def items(self):
        return self.tid2vertexList.items()

    def array(self):
        return self.occur_list

    def tid_set(self):
        return self.tid2vertexList.keys()

    def __repr__(self):
        return ' '.join(["({}, {})".format(t, o) for t,o in self.tid2vertexList.items()])

class TemplateDB(object):

    def __init__(self):
        self.tree2occurList = dict()
        self.extended_tree_set = set()
        self.vertex2actionList = dict()
        self.action2id = dict()
        self.id2action = dict()
        self.action2tidSet = dict()
        self.processed_trees = set()
        self.var_action2id = dict()
        self.id2var_action = dict()
        self.entity_list = []
        self.variable_list = []

    def add_tree_occurList(self, tree_key, tid, new_ref_tree):

        if not tree_key in self.tree2occurList:
            occurList = OccurList()
            self.tree2occurList[tree_key] = occurList
        else:
            occurList = self.tree2occurList[tree_key]
        occurList.add(tid, new_ref_tree)

    def add_occurList(self, tree, occurList):
        if tree in self.tree2occurList:
            old_occurList = self.tree2occurList[tree]
            tid_set = old_occurList.tid_set()
            for tid, tree in occurList.array():
                if not tid in tid_set:
                    old_occurList.add(tid, tree)
        else:
            self.tree2occurList[tree] = occurList
        return occurList

    def is_tree_extended(self, tree):
        return tree in self.extended_tree_set

    def size(self):
        return len(self.tree2occurList)

    def index_action(self, tid, action):
        if action in self.action2id:
            action_id = self.action2id[action]
            self.action2tidSet[action].add(tid)
        else:
            action_id = len(self.action2id)
            tid_set = set()
            self.action2id[action] = action_id
            self.action2tidSet[action] = tid_set
            tid_set.add(tid)
        return action_id

    def index_var_action(self, tid, action):
        if action in self.action2id:
            var_action_id = self.var_action2id[action]
            self.var_action2tidSet[action].add(tid)
        else:
            var_action_id = len(self.var_action2id)
            tid_set = set()
            self.var_action2id[action] = var_action_id
            self.var_action2tidSet[action] = tid_set
            tid_set.add(tid)
        return var_action_id

def index_leaves(root, template_id, template_db):
    leaf2occurList = template_db.tree2occurList
    visited, queue = set(), [root]
    while queue:
        vertex = queue.pop(0)
        v_id = id(vertex)
        visited.add(v_id)
        if not vertex.has_children():
            if vertex.parent and not vertex.parent.is_struct():
                new_var = BFSTree(vertex.copy_no_link())
                if not new_var in leaf2occurList:
                    occurList = OccurList()
                    leaf2occurList[new_var] = occurList
                else:
                    occurList = leaf2occurList[new_var]
                occurList.add(template_id, BFSTree(vertex))

        elif vertex.is_struct():
            new_struct = build_tree_with_direct_children(vertex)
            if not new_struct in leaf2occurList:
                occurList = OccurList()
                leaf2occurList[new_struct] = occurList
            else:
                occurList = leaf2occurList[new_struct]
            ref_tree = BFSTree(vertex)
            ref_tree_level = TreeLevel()
            ref_tree_level.add(vertex.children)
            ref_tree.add(ref_tree_level)
            occurList.add(template_id, ref_tree)
        else:
            for child in vertex.children:
                if id(child) not in visited:
                    queue.append(child)

def build_tree_with_direct_children(vertex):
    new_vertex = vertex.copy_no_link()
    tree = BFSTree(new_vertex)
    level = TreeLevel()
    tree.add(level)
    level.add(new_vertex.children)
    for child in vertex.children:
        new_child = child.copy_no_link()
        new_child.parent = new_vertex
        new_vertex.children.append(new_child)
    return tree

def tree_expand(tree_key, occurList, existing_treeSet, processed_trees, use_all_subtrees = False):
    new_tree2occurList = dict()
    if not tree_key in processed_trees and tree_key.depth() > 0:
        i = -1
        occur_list = occurList.array()
        for tid, ref_bfs_tree in occur_list:
            i += 1
            if ref_bfs_tree.depth() > 0:
                ref_children = ref_bfs_tree.root.children
                key_children = tree_key[0][0]
                if len(ref_children) > len(key_children):
                    all_indexed = True
                    bfs_tree_list = [c.bfs_tree() for c in ref_children]
                    if use_all_subtrees:
                        for subtree in bfs_tree_list:
                            if not subtree in existing_treeSet:
                                all_indexed = False
                    else:
                        key_children_copy = key_children.copy()
                        for child in ref_children:
                            if child in key_children_copy:
                                key_children_copy.remove(child)
                            elif not child.is_variable():
                                all_indexed = False
                    if all_indexed:
                        full_ref_tree = ref_bfs_tree.root.bfs_tree()
                        new_tree = full_ref_tree.clone()

                        if not new_tree in existing_treeSet:
                            new_occurList = OccurList()
                            new_tree2occurList[new_tree] = new_occurList
                            new_occurList.add(tid, full_ref_tree)
                            j = i + 1
                            ref_repr = new_tree.__repr__()
                            assert(full_ref_tree == new_tree), 'REF: {} NEW: {}'.format(full_ref_tree, new_tree)

                            while j < occurList.size():
                                tid_next, tree_next = occur_list[j]
                                ref_tree_next = tree_next.root.bfs_tree()
                                ref_tree_next_repr = ref_tree_next.__repr__()
                                if ref_tree_next_repr ==  ref_repr:
                                     new_occurList.add(tid_next, ref_tree_next)
                                j += 1
                            processed_trees.add(new_tree)
                            if new_occurList.size() > 0:
                                break
        processed_trees.add(tree_key)
    return new_tree2occurList


def tree_extend(template_db):
    tree2occurList = template_db.tree2occurList
    tree_list = list(tree2occurList.keys())
    for bfs_tree in tree_list:
        if not template_db.is_tree_extended(bfs_tree):
            new_tree2occurList = dict()
            pre_occurList = tree2occurList[bfs_tree]
            for tid, ref_tree in pre_occurList.array():
                parent = ref_tree.root.parent
                if parent is not None:
                    new_root = parent.copy_no_link()
                    new_tree = deepcopy(bfs_tree)
                    new_tree.insert_new_root(new_root)

                    new_ref_tree = ref_tree.copy()
                    new_ref_tree.extend_parent_as_new_root()

                    if new_tree in new_tree2occurList:
                        occurList = new_tree2occurList[new_tree]
                    else:
                        occurList = OccurList()
                        new_tree2occurList[new_tree] = occurList
                    occurList.add(tid, new_ref_tree)

            template_db.extended_tree_set.add(bfs_tree)
            final_tree2occurList = dict()
            for tree_key, occurList in new_tree2occurList.items():
                if occurList.size() >= pre_occurList.size():
                    valid_tree2occurList=tree_expand(tree_key, occurList, tree2occurList.keys(), template_db.processed_trees)
                    for t in valid_tree2occurList.keys():
                        assert not t in final_tree2occurList, '{} already exist'.format(t)
                    final_tree2occurList.update(valid_tree2occurList)
            for new_tree, occurList in final_tree2occurList.items():
                template_db.add_occurList(new_tree, occurList)
            if len(final_tree2occurList) > 0:
                del template_db.tree2occurList[bfs_tree]


def normalize_trees(trees):
    temp_db = TemplateDB()
    for tid in range(len(trees)):
        index_leaves(trees[tid], tid, temp_db)
    num_tree = 0
    iter = 0
    tree_set = set()
    tree_updated = True
    while tree_updated:
        print('Ite : {}'.format(iter))
        tree_set = set(temp_db.tree2occurList.keys()).copy()
        tree_extend(temp_db)
        has_new = False
        for t in temp_db.tree2occurList.keys():
            if not t in tree_set:
                has_new = True
        tree_updated = has_new
        iter+=1

    return temp_db

def create_template_to_leaves(template_trees, template_db,freq=0, topk=-1):
    leave2occurList = template_db.tree2occurList
    leaves = [[] for i in range(len(template_trees))]

    freq_list = [occurList.size() for occurList in leave2occurList.values()]
    freq_list.sort(reverse=True)
    topk_freq = freq_list[topk]
    for occurList in leave2occurList.values():
        if occurList.size() >= freq and occurList.size() >= topk_freq:
            for tid,ref_tree in occurList.array():
                leaves[tid].append(ref_tree)
    return leaves


def convert_tree_to_composite_vertex(leaves):
    for tid, leaf in enumerate(leaves):
        if leaf.root.parent is not None and leaf.root.has_children():
            com_vertex = CompositeTreeVertex(leaf.root)
            #print('Link {} to {}'.format(com_vertex, leaf.root.parent))
            children = leaf.root.parent.children
            new_children = []
            for child in children:
                if id(child) == id(leaf.root):
                    new_children.append(com_vertex)
                else:
                    new_children.append(child)
            leaf.root.parent.children = new_children

