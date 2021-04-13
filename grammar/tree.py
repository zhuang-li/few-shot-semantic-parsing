class TreeLevel(object):

    def __init__(self):
        self.children_list = []

    def add(self, nodes):
        self.children_list.append(nodes)

    def size(self):
        return len(self.children_list)

    def __getitem__(self, key):
        return self.children_list[key]

class BFSTree(object):

    def __init__(self, root):
        self.root = root
        self.levels = []

    def add(self, level):
        self.levels.append(level)

    def depth(self):
        return len(self.levels)

    def __getitem__(self, key):
        return self.levels[key]

    def clone(self):
        new_tree = BFSTree(self.root.copy_no_link())
        level_index = 0
        for level in self.levels:
            new_level = TreeLevel()
            new_tree.add(new_level)
            for children in level.children_list:
                parent_id = id(children[0].parent)
                parent = None
                if level_index > 0:
                    p_index = 0
                    for parents in self.levels[level_index - 1].children_list:
                        pp_index = 0
                        for p in parents:
                            if parent is None and id(p) == parent_id:
                                parent = new_tree.levels[level_index - 1][p_index][pp_index]
                            pp_index += 1
                        p_index += 1
                else:
                    parent = new_tree.root
                new_children = []
                for child in children:
                    new_child = child.copy_no_link()
                    new_child.parent = parent
                    new_children.append(new_child)
                if parent is not None:
                    parent.children = new_children
                new_level.add(new_children)
            level_index += 1
        return new_tree

    def copy(self):
        new_tree = BFSTree(self.root)
        for level in self.levels:
            new_level = TreeLevel()
            new_tree.levels.append(new_level)
            for children in level.children_list:
                new_children = []
                for child in children:
                    new_children.append(child)
                new_level.add(new_children)
        return new_tree

    def insert_new_root(self, new_root):
        first_level = TreeLevel()
        first_level.add([self.root])
        self.levels.insert(0,first_level)

        new_root.children.append(self.root)
        self.root.parent = new_root

        self.root = new_root

    def extend_parent_as_new_root(self):
        first_level = TreeLevel()
        first_level.add([self.root])
        self.levels.insert(0,first_level)
        self.root = self.root.parent

    def bfs_code(self, num_levels):
        assert num_levels <= len(self.levels) and num_levels >= 0, 'The number of levels {} should be >= 0 and <= {}'.format(num_levels, len(self.levels))
        sym_list = []
        sym_list.append(self.root.head)
        for i in range(0, num_levels):
            level = self.levels[i]
            sym_list.append('[')
            for nodes in level.children_list:
                sym_list.append('(')
                i = 0
                for node in nodes:
                    if i > 0:
                        sym_list.append(',')
                    i += 1
                    sym_list.append(node.head)
                sym_list.append(')')
            sym_list.append(']')
        return sym_list

    def prefix(self):
        if len(self.levels) > 0:
            if self.root.is_struct():
                return ''
            else:
                return self.root.head#' '.join(self.bfs_code(len(self.levels) - 1))
        else:
            return ''

    def rep(self):
        return self.root

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return " ".join(self.bfs_code(len(self.levels)))

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()
