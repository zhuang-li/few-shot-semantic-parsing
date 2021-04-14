from grammar.vertex import *
from grammar.action import *
from grammar.consts import *
from common.registerable import Registrable

class Rule(object):
    def __init__(self, head):
        assert isinstance(head, TreeVertex), '{} is of type {}'.format(head, type(head))
        # The rule head is a RuleVertex object
        self.head = head
        self.body_length = 0

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{}".format(self.head)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()

@Registrable.register('ProductionRuleBL')
class ProductionRuleBL(Rule):
    def __init__(self, head, body_length):
        assert isinstance(head, TreeVertex), '{} is of type {}'.format(head, type(head))
        # The rule head is a RuleVertex object
        self.head = head
        # The rule body is a list of RuleVertex objects
        self.body_length = body_length

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{} :- [{}]".format(self.head, self.body_length)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()

@Registrable.register('ProductionRuleBLB')
class ProductionRuleBLB(Rule):
    def __init__(self, head, body_length, body):
        assert isinstance(head, TreeVertex), '{} is of type {}'.format(head, type(head))
        # The rule head is a RuleVertex object
        self.head = head
        # The rule body is a list of RuleVertex objects
        self.body_length = body_length
        self.body = body

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{} :- {} {}".format(self.head, self.body_length, self.body)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()

def recursive_reduce(dfs_tree_root, config_seq, production_rule, body = None):
    if body:
        new_body = list(body)
        assert len(body) == len(dfs_tree_root.children), "body length must be equal to the children length"
    for id, child in enumerate(dfs_tree_root.children):
        if production_rule == ProductionRuleBLB:
            if not new_body[id] == NT:
                continue
        if not child.has_children():
            action = GenAction(child)
            config_seq.append(action)
            action.entities = extract_action_lit([[action]], 'entity')[0]
            action.variables = extract_action_lit([[action]], 'variable')[0]
        else:
            head = RuleVertex(child.head)
            if str(head).startswith(IMPLICIT_HEAD):
                head.is_auto_nt = True
            if production_rule == ProductionRuleBLB:
                body = []
                body_len = 0
                for c in child.children:
                    if len(c.children) == 0 and (not isinstance(c, CompositeTreeVertex)):
                        body.append(c)
                    else:
                        body.append(NT)
                        body_len += 1
                rule = production_rule(head, body_len, body)
                recursive_reduce(child, config_seq, production_rule, body)
            else:
                rule = production_rule(head, len(child.children))
                recursive_reduce(child, config_seq, production_rule)

            reduce = ReduceAction(rule)
            if production_rule == ProductionRuleBLB:
                reduce.entities = extract_action_lit([[reduce]], 'entity')[0]
                reduce.variables = extract_action_lit([[reduce]], 'variable')[0]
            config_seq.append(reduce)


def turn_var_back(dfs_tree_root, turn_v_back = True, turn_e_back = True):
    list_nodes = [dfs_tree_root]
    while len(list_nodes) > 0:
        node = list_nodes.pop()
        if isinstance(node, RuleVertex):
            if node.original_var and turn_v_back:
                node.head = node.original_var
            if node.original_entity and turn_e_back:
                node.head = node.original_entity
        elif isinstance(node, CompositeTreeVertex):
            turn_var_back(node.vertex)
        for child in node.children:
            list_nodes.append(child)

def product_rules_to_actions_bottomup(template_trees, leaves_list, template_db, use_normalized_trees=True,
                             rule_type="ProductionRuleBL",turn_v_back=False):
    tid2config_seq = []
    if use_normalized_trees:
        for tid, leaves in enumerate(leaves_list):
            convert_tree_to_composite_vertex(leaves)

    production_rule = Registrable.by_name(rule_type)
    composite_ast = []
    for tid, dfs_tree_root in enumerate(template_trees):
        config_seq = []
        if turn_v_back:
            turn_var_back(dfs_tree_root)
        head = RuleVertex(dfs_tree_root.head)
        head.is_auto_nt = True
        if production_rule == ProductionRuleBLB:
            body = []
            for c in dfs_tree_root.children:
                if len(c.children) == 0 and (not isinstance(c, CompositeTreeVertex)):
                    body.append(c)
                    c.parent = None
                else:
                    body.append(NT)
            rule = production_rule(head, FULL_STACK_LENGTH, [])
            recursive_reduce(dfs_tree_root, config_seq, production_rule, body)
        else:
            rule = production_rule(head, FULL_STACK_LENGTH)
            recursive_reduce(dfs_tree_root, config_seq, production_rule)

        reduce = ReduceAction(rule)
        config_seq.append(reduce)
        for c in config_seq:
            template_db.index_action(tid, c)
        tid2config_seq.append(config_seq)
        composite_ast.append(dfs_tree_root)
    return tid2config_seq



def get_vertex_variables(vertex, type, seq_list):
    if vertex.has_children():
        for child in vertex.children:
            get_vertex_variables(child, type, seq_list)
    else:
        if type == 'variable':
            if vertex.original_var:
                vertex.finished = False
                seq_list.append(vertex.original_var)
        elif type == 'entity':
            if vertex.original_entity:
                vertex.finished = False
                seq_list.append(vertex.original_entity)

def extract_action_lit(action_seqs, type = 'variable'):
    seq = []
    for action_seq in action_seqs:
        seq.append([])
        for action in action_seq:
            if isinstance(action, GenAction):
                if isinstance(action.vertex, RuleVertex):
                    vertex = action.vertex
                else:
                    vertex =  action.vertex.vertex
                get_vertex_variables(vertex, type, seq[-1])
            elif isinstance(action, ReduceAction):
                for vertex in action.rule.body:
                    if isinstance(vertex, RuleVertex):
                        if type == 'variable':
                            if vertex.original_var:
                                seq[-1].append(vertex.original_var)
                        elif type == 'entity':
                            if vertex.original_entity:
                                seq[-1].append(vertex.original_entity)
                    elif isinstance(vertex, str):
                        continue
                    else:
                        raise ValueError
            else:
                raise ValueError
    return seq