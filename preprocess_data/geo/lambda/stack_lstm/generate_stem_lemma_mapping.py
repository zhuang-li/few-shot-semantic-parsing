import os
import pickle

from preprocess_data.geo.geo_utils import read_geo_prolog_data_lines
from preprocess_data.nlp_utils import token_stem_lemma_mapping

path_prefix = "../../../../datasets/geo/prolog/"
dump_path_prefix = "../../../../datasets/geo/"

def read_file_lines(file_path):
    src_list, tgt_list = read_geo_prolog_data_lines(file_path, train=['train', 'dev', 'test'], is_query_split=True)
    map_dict = {}
    for seq in src_list:
        token_stem_lemma_mapping(seq, map_dict)
    pickle.dump(map_dict, open(os.path.join(dump_path_prefix, 'stem_lemma_map_dict.bin'), 'wb'))
if __name__ == '__main__':
    file_path = os.path.join(path_prefix, 'geo_query_split.txt')
    read_file_lines(file_path)
    pass
