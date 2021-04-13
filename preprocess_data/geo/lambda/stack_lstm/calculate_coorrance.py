import operator
import os

from grammar.utils import is_predicate

path_prefix = "../../../../datasets/geo/lambda/query_split/few_shot_split/"
dump_path_prefix = "../../../../datasets/geo/lambda/query_split/few_shot_split/"

def calculate_coourace(train_file):
    coor_table = dict()
    with open(train_file) as json_file:
        for line in json_file:
            src, tgt = line.split('\t')
            # make the jobs data format same with the geoquery
            tgt = tgt.strip()
            src = src.strip()
            # src = re.sub(r"((?:c|s|m|r|co|n)\d)", VAR_NAME, src)
            src_list = src.split(' ')

            tgt_list = tgt.split(' ')
            for token in tgt_list:
                if is_predicate(token,dataset='geo_lambda'):
                    if token in coor_table:
                        for src_token in src_list:
                            if src_token in coor_table[token]:
                                coor_table[token][src_token] = coor_table[token][src_token] + 1
                            else:
                                coor_table[token][src_token] = 1
                    else:
                        coor_table[token] = {}
                        for src_token in src_list:
                            if src_token in coor_table[token]:
                                coor_table[token][src_token] = coor_table[token][src_token] + 1
                            else:
                                coor_table[token][src_token] = 1

    for pred, orr in coor_table.items():
        print (pred)
        oo  = list(orr)
        oo.sort(reverse=True)
        print (sorted(orr.items(), key=operator.itemgetter(1), reverse=True))

if __name__ == '__main__':
    train_file = os.path.join(path_prefix, 'train.txt')
    test_file = os.path.join(path_prefix, 'test.txt')
    calculate_coourace(train_file)
    pass