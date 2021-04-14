VAR_NAME='A'
SLOT_PREFIX='var'
TYPE_SIGN='$'
ROOT = 'answer'
IMPLICIT_HEAD = 'hidden'
FULL_STACK_LENGTH = 100000
NT = 'nonterminal'
PAD_PREDICATE = 'pad_predicate'

# supervised attention
PREDICATE_LEXICON = {
    'loc' : ['location', 'where', 'in', 'loc'],
    'argmax' : ['highest', 'largest', 'most', 'greatest', 'longest', 'biggest', "high", "maximum"],
    'argmin' : ['shortest', 'smallest', 'least', 'lowest', 'minimum'],
    '>' : ['greater', 'larger', '-er', 'than'],
    'count' : ['many','count'],
    '=': ['equal'],
    '\+' : ['not'],
    'req' : ['require'],
    'deg' : ['degree'],
    'exp' : ['experience'],
    'des' : ['desire'],
    'len' : ['length'],
    'next' : ['next', 'border', 'border', 'neighbor', 'surround'],
    'density': ['average', 'population','density'],
    'sum': ['total', 'sum'],
    'size': ['size', 'big', 'biggest', 'largest'],
    'population': ['population', 'people', 'citizen']
}
