import os
import pickle
import sys
from itertools import chain
import sys

import torch

sys.path.append('./')
import numpy as np

from model.utils import GloveHelper
from preprocess_data.utils import produce_data, get_predicate_tokens, generate_dir, \
    pre_few_shot_vocab, generate_examples
from components.vocab import Vocab
from components.vocab import TokenVocabEntry
import torch.nn as nn
from components.vocab import ActionVocabEntry, GeneralActionVocabEntry, GenVocabEntry, ReVocabEntry, VertexVocabEntry, \
    LitEntry

if __name__ == '__main__':

    dataset = 'jobs'
    data_type = 'job_prolog'
    lang = 'prolog'
    place_holder = 3
    frequency_list = [0,50]
    generate_examples(dataset=dataset, data_type=data_type, lang=lang, place_holder=place_holder, frequency_list = frequency_list)