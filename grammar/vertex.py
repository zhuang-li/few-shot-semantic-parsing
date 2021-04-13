from grammar.consts import *
from grammar.tree import *
class TreeVertex(object):

    def __init__(self, parent = None, children = [], depth = 0):
        self.parent = parent
        self.children = children
        self.depth = depth
        self.var_name = VAR_NAME
        self.position = 0
        self.is_auto_nt = False
        self.prototype_tokens = []
        self.finished = True
        self.created_time = -1

    def __getitem__(self, key):
        return self.children[key]

    def is_terminal(self):
        return len(self.children) == 0

    def root(self):
        if self.parent is not None:
            parent = self.parent
            while parent:
                parent = self.parent
            return parent
        else:
            return self

    def is_root(self):
        return self.parent is None

    def add(self, rule_vertex):
        self.children.append(rule_vertex)

    def shallow_eq(self, other):
        pass

    def has_children(self):
        return len(self.children) > 0

    def bfs_tree(self):
        visited, queue = set(), [self]
        tree = BFSTree(self)
#        if self.depth > 0:
#            raise Exception("Vertex ({}) has depth {}, but it should be 0.".format(self.head, self.depth))
        c_depth = 0
        while queue:
            vertex = queue.pop(0)
            depth = vertex.depth
            if vertex.has_children():
                c_depth = depth + 1

                if c_depth <= tree.depth():
                    tree_level = tree.levels[c_depth - 1]
                else:
                    tree_level = TreeLevel()
                    tree.add(tree_level)

            v_id = id(vertex)
            if v_id not in visited:
                visited.add(v_id)
                if vertex.has_children():
                    children = []
                    for child in vertex.children:
                        if id(child) not in visited:
                            queue.append(child)
                            children.append(child)
                            child.depth = c_depth
                    tree_level.add(children)

        return tree

    def is_variable(self):
        pass

    def is_struct(self):
        pass

    def rep(self):
        raise Exception('Not implemented.')

class RuleVertex(TreeVertex):

    def __init__(self, head):
        TreeVertex.__init__(self,None, [], 0)
        self.head = head
        self.original_var = None
        self.original_entity = None
        # Point to the grammar sharing the same var
        self.share_var_with = set()

    def copy_no_link(self):
        new_v = RuleVertex(self.head)
        new_v.depth = self.depth
        new_v.is_auto_nt = self.is_auto_nt
        new_v.original_var = self.original_var
        new_v.original_entity = self.original_entity
        return new_v

    def copy(self):
        new_tree = self.copy_no_link()
        for old_child in self.children:
            new_tree.add(old_child.copy())
        return new_tree

    def add(self, rule_vertex):
        self.children.append(rule_vertex)

    def is_answer_root(self):
        return self.is_root() and self.head == ROOT

    def shallow_eq(self, other):
        return self.head == other.head and self.is_auto_nt == other.is_auto_nt

    def rep(self):
        v = self.copy_no_link()
        return v

    def dfs_code(self, display_auto_nt = True, show_position = False, type = "prolog"):
        sym_list = []
        if type == 'prolog':
            dfs_recursive_search(self, sym_list, set(), display_auto_nt, show_position)
        elif type == 'lambda':
            dfs_recursive_search_lambda(self, sym_list, set(), display_auto_nt, show_position)
        else:
            raise ValueError
        return sym_list

    @property
    def to_prolog_expr(self):
        return " ".join(self.dfs_code(False, False))

    @property
    def to_lambda_expr(self):
        rep_list = self.dfs_code(False, False, type='lambda')
        return " ".join(rep_list)

    def __str__(self):
        return " ".join(self.dfs_code(True, False))

    def __repr__(self):
        return " ".join(self.dfs_code(True, False))

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def is_variable(self):
        return (not self.has_children())

    def is_struct(self):
        body_set = set([c.is_variable() for c in self.children])
        return True in body_set and len(body_set) == 1


class CompositeTreeVertex(TreeVertex):

    def __init__(self, vertex):
        TreeVertex.__init__(self,vertex.parent, [], vertex.depth)
        self.vertex = vertex
        self.position = vertex.position
        self.is_auto_nt = vertex.is_auto_nt
        self.var_vertex_list = []

    def copy_no_link(self):
        new_v = CompositeTreeVertex(self.vertex.copy())
        return new_v

    def copy(self):
        new_tree = self.copy_no_link()
        for old_child in self.children:
            new_tree.add(old_child.copy())
        return new_tree

    def rep(self):
        return self.vertex

    def shallow_eq(self, other):
        return self.__repr__() == other.__repr__()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.vertex.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def is_variable(self):
        if isinstance(self.vertex, RuleVertex):
            return (not self.vertex.has_children())
        else:
            return False

    def is_struct(self):
        if isinstance(self.vertex, RuleVertex):
            body_set = set([c.is_variable() for c in self.vertex.children])

            return True in body_set and len(body_set) == 1
        else:
            return False


def get_children_num(vertex):
    num = 1
    list_vertex = []
    list_vertex.append(vertex)
    while len(list_vertex) > 0:
        vertex = list_vertex.pop()
        for child in vertex.children:
            list_vertex.append(child)
            num +=1
    return num

def dfs_recursive_search(vertex, sym_list, visited, display_auto_nt=False, show_position = False):
        if not vertex.is_auto_nt or display_auto_nt:
            if isinstance(vertex, RuleVertex):
                sym_list.append(vertex.head)
            else:
                sym_list.append(vertex.__repr__())
        if vertex.has_children():
            if isinstance(vertex, RuleVertex) and (not (vertex.head == '\+')):
                sym_list.append('(')
        visited.add(id(vertex))
        i = 0
        last_child = None
        for child in vertex.children:
            if i > 0:
                if (isinstance(child, RuleVertex) and (child.head == ';')) or ((last_child is not None) and (isinstance(last_child, RuleVertex)) and ((last_child.head == ';'))):
                    sym_list = sym_list
                else:
                    sym_list.append(',')
            i += 1
            visited = dfs_recursive_search(child, sym_list, visited, display_auto_nt, show_position)
            last_child = child
        if vertex.has_children():
            if isinstance(vertex, RuleVertex) and (not (vertex.head == '\+')):
                sym_list.append(')')
        return visited


def dfs_recursive_search_lambda(vertex, sym_list, visited, display_auto_nt=False, show_position=False):
    if vertex.has_children():
        if (isinstance(vertex, RuleVertex) and (not str(vertex.head) == ROOT)):
            sym_list.append('(')
    if not vertex.is_auto_nt or display_auto_nt:
        if isinstance(vertex, RuleVertex):
            if (not (str(vertex.head) == PAD_PREDICATE)):
                sym_list.append(vertex.head)
        else:
            sym_list.append(vertex.vertex.to_lambda_expr)
    visited.add(id(vertex))
    for child in vertex.children:
        visited = dfs_recursive_search_lambda(child, sym_list, visited, display_auto_nt, show_position)
    if vertex.has_children():
        if (isinstance(vertex, RuleVertex) and (not str(vertex.head) == ROOT)):
            sym_list.append(')')
    return visited

def convert_tree_to_composite_vertex(leaves):
    for tid, leaf in enumerate(leaves):
        if leaf.root.parent is not None and leaf.root.has_children():
            com_vertex = CompositeTreeVertex(leaf.root)

            children = leaf.root.parent.children
            new_children = []
            for child in children:
                if id(child) == id(leaf.root):
                    new_children.append(com_vertex)
                else:
                    new_children.append(child)
            leaf.root.parent.children = new_children
