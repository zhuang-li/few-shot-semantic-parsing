from components.grammar_validation.asdl import LambdaCalculusTransitionSystem
from components.grammar_validation.asdl.hypothesis import Hypothesis, ApplyRuleAction, ASDLGrammar
from components.grammar_validation.asdl.lang.lambda_dcs.logical_form import parse_lambda_expr, logical_form_to_ast, \
    ast_to_logical_form

def read_grammar(lang = 'geo_lambda'):
    if lang.endswith('lambda'):
        grammar = ASDLGrammar.from_text(open('components/grammar_validation/asdl/lang/lambda_dcs/lambda_asdl.txt').read())
        transition_system = LambdaCalculusTransitionSystem(grammar)
        return grammar, transition_system

def validate_logic_form(tgt_code, grammar, transition_system):
    print('Logic Form: %s' % tgt_code)
    lf = parse_lambda_expr(tgt_code)

    return lf.to_string() == tgt_code

    tgt_ast = logical_form_to_ast(grammar, lf)
    reconstructed_lf = ast_to_logical_form(tgt_ast)
    return lf == reconstructed_lf

    tgt_actions = transition_system.get_actions(tgt_ast)

    # print('===== Actions =====')
    # sanity check
    hyp = Hypothesis()
    for action in tgt_actions:
        if not action.__class__ in transition_system.get_valid_continuation_types(hyp):
            return False
        if isinstance(action, ApplyRuleAction):
            if not action.production in transition_system.get_valid_continuating_productions(hyp):
                return False
        hyp = hyp.clone_and_apply_action(action)
    return True
