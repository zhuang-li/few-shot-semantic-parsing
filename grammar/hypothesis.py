# coding=utf-8
from grammar.rule import ReduceAction
from grammar.vertex import CompositeTreeVertex, RuleVertex, TreeVertex
from grammar.rule import ProductionRuleBLB
from model.attention_util import dot_prod_attention
from grammar.consts import IMPLICIT_HEAD, ROOT, FULL_STACK_LENGTH, NT


class Hypothesis(object):
    def __init__(self):
        self.tree = None
        # action
        self.actions = []
        self.action_id = []
        self.general_action_id = []
        # tgt code
        self.tgt_code_tokens = []
        self.tgt_code_tokens_id = []

        self.var_id = []
        self.ent_id = []

        self.heads_stack = []
        self.tgt_ids_stack = []
        self.heads_embedding_stack = []
        self.embedding_stack = []
        self.hidden_embedding_stack = []
        self.v_hidden_embedding = None

        self.current_gen_emb = None
        self.current_re_emb = None
        self.current_att = None
        self.current_att_cov = None

        self.score = 0.
        self.is_correct = False
        self.is_parsable = True
        # record the current time step
        self.reduce_action_count = 0
        self.t = 0

        self.frontier_node = None
        self.frontier_field = None


    @property
    def to_prolog_template(self):
        if self.tree:
            return self.tree.to_prolog_expr
        else:
            return ""

    @property
    def to_lambda_template(self):
        if self.tree:
            return self.tree.to_lambda_expr
        else:
            return ""

    @property
    def to_logic_form(self):
        return " ".join([str(code) for code in self.tgt_code_tokens])

    def is_reduceable(self, rule, vertex_stack):
        head = rule.head.copy_no_link()
        if head.head.startswith(ROOT) and rule.body_length < len(vertex_stack):
            return False
        temp_stack = [node.copy_no_link() for node in vertex_stack[-rule.body_length:]]
        if head.head.startswith(IMPLICIT_HEAD) or head.head.startswith(ROOT):
            head.is_auto_nt = True

        if isinstance(rule, ProductionRuleBLB):
            if head.head.startswith(ROOT):
                for i in range(len(temp_stack)):
                    head.children.append(vertex_stack.pop().copy())
            else:
                for child in rule.body[::-1]:
                    if not (str(child) == NT):
                        head.children.append(child.copy())
                    else:
                        head.children.append(vertex_stack.pop().copy())

        else:
            for i in range(len(temp_stack)):
                head.children.append(vertex_stack.pop().copy())

        head.children.reverse()
        vertex_stack.append(head)
        return True


    def reduce_actions(self, reduce_action):
        stack = self.heads_stack
        if self.is_reduceable(reduce_action.rule, stack):
            self.is_parsable = True
            if len(stack) == 1 and stack[0].is_answer_root():
                self.tree = stack[0]
        else:
            self.is_parsable = False
        return stack

    def reduce_embedding(self, reduce_embedding, length_of_rule, mem_net):
        assert length_of_rule <= len(self.embedding_stack), "embedding stack length must be longer than or equal to the rule body length"
        if mem_net:
            partial_stack_embedding = torch.stack(self.embedding_stack[-length_of_rule:])
            reduce_embedding = reduce_embedding.unsqueeze(0)
            partial_stack_embedding = partial_stack_embedding.unsqueeze(0)
            cxt_vec, cxt_weight = dot_prod_attention(reduce_embedding, partial_stack_embedding, partial_stack_embedding)
            reduce_embedding = reduce_embedding.squeeze() + cxt_vec.squeeze()
        for i in range(length_of_rule):
            self.embedding_stack.pop()
        self.embedding_stack.append(reduce_embedding)
        return self.embedding_stack

    def reduce_action_ids(self, reduce_id, length_of_rule):
        if length_of_rule == FULL_STACK_LENGTH:
            length_of_rule = len(self.tgt_ids_stack)
        assert length_of_rule <= len(self.tgt_ids_stack), "action id length must be longer than or equal to the rule body length"
        for i in range(length_of_rule):
            self.tgt_ids_stack.pop()
        self.tgt_ids_stack.append(reduce_id)
        return self.tgt_ids_stack


    def copy(self):
        new_hyp = Hypothesis()
        if self.tree:
            new_hyp.tree = self.tree.copy()
        new_hyp.action_id = list(self.action_id)
        new_hyp.actions = list(self.actions)
        new_hyp.general_action_id = list(self.general_action_id)
        new_hyp.var_id = list(self.var_id)
        new_hyp.ent_id = list(self.ent_id)
        new_hyp.heads_stack = list(self.heads_stack)
        new_hyp.tgt_ids_stack = list(self.tgt_ids_stack)
        new_hyp.embedding_stack = [embedding.clone() for embedding in self.embedding_stack]
        new_hyp.heads_embedding_stack = [embedding.clone() for embedding in self.heads_embedding_stack]
        new_hyp.hidden_embedding_stack = [(state.clone(), cell.clone()) for state, cell in self.hidden_embedding_stack]
        new_hyp.tgt_code_tokens_id = list(self.tgt_code_tokens_id)
        new_hyp.tgt_code_tokens = list(self.tgt_code_tokens)

        new_hyp.score = self.score
        new_hyp.t = self.t
        new_hyp.is_correct = self.is_correct
        new_hyp.is_parsable = self.is_parsable
        new_hyp.reduce_action_count = self.reduce_action_count

        if self.current_gen_emb is not None:
            new_hyp.current_gen_emb = self.current_gen_emb.clone()

        if self.current_re_emb is not None:
            new_hyp.current_re_emb = self.current_re_emb.clone()

        if self.current_att is not None:
            new_hyp.current_att = self.current_att.clone()

        if self.current_att_cov is not None:
            new_hyp.current_att_cov = self.current_att_cov.clone()

        if self.v_hidden_embedding is not None:
            new_hyp.v_hidden_embedding = (self.v_hidden_embedding[0].clone(), self.v_hidden_embedding[1].clone())

        return new_hyp

    def completed(self):
        return self.tree
