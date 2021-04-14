from grammar.utils import rm_subbag

class Action(object):
    def __init__(self):
        self.neo_type = False
        self.entities = []
        self.variables = []
        self.prototype_tokens = []
        self.string_sim_score = []
        self.cond_score = []
        self.entity_align = []


class ReduceAction(Action):

    def __init__(self, rule):
        Action.__init__(self)
        self.rule = rule

    def __repr__(self):
        return 'REDUCE {}'.format(self.rule).replace(' ', '_')

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()


class GenAction(Action):

    def __init__(self, vertex):
        Action.__init__(self)
        self.vertex = vertex
        self.parent_t = -1

    def __repr__(self):
        return 'GEN {}'.format(self.vertex.rep()).replace(' ', '_')

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()


class ParserConfig(object):

    def __init__(self, action = None):
        # A queue stores the current level of vertices to process after applying the action.
        # Each vertex is not a reference to any vertex in the full parse tree.
        self.queue = []
        # Parse action
        self.action = action

    def transit(self, action):
        new_config = ParserConfig(action)
        new_config.queue.extend(self.queue)
        if isinstance(action, ReduceAction):
            rm_subbag(action.rule.body, new_config.queue)
            new_config.queue.append(action.rule.head.rep())
        else:
            new_config.queue.append(action.vertex.rep())
        return new_config


    def __repr__(self):
        return '({} {})'.format(self.action, self.queue)

    def __str__(self):
        return self.__repr__()
