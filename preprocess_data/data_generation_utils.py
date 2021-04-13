import operator
import os

import numpy as np

from components.dataset import Example
from grammar.consts import VAR_NAME, SLOT_PREFIX, ROOT

from grammar.utils import is_predicate, is_var, is_lit
from preprocess_data.utils import generate_dir, produce_data


def split_by_new_token(query_dict, num_non_overlap = 5, random_shuffle = True, data_type = 'geo_lambda', predicate_list = []):
    train_dict = {}
    test_dict = {}
    vocab_dict = {}

    for k, v in query_dict.items():
        for word in k.split(' '):
            if is_predicate(word,dataset=data_type):
                if word in vocab_dict:
                    vocab_dict[word] += 1
                else:
                    vocab_dict[word] = 1
    vocab_list_ = sorted(vocab_dict.items(), key=operator.itemgetter(1))

    vocab_list = []
    for k, v in vocab_list_:
        if v >= 2:
            vocab_list.append((k,v))

    if random_shuffle:
        np.random.shuffle(vocab_list)
    vocab_list = vocab_list[:num_non_overlap]
    print("vocab list ", vocab_list)
    vocab_dict_freq = {}
    # dict(vocab_list)
    # vocab_dict_freq = {"departure_time" : 1, "has_meal": 1, "class_type" : 1 , "meal": 1, "ground_fare" : 1}
    for idx, predicate in enumerate(predicate_list):
        vocab_dict_freq[predicate] = 1
    for k, v in query_dict.items():
        flag = True
        for word in k.split(' '):
            if word in vocab_dict_freq:
                test_dict[k] = v
                flag = False
                break
        if flag:
            train_dict[k] = v
    train_vocab_dict = {}
    for k, v in train_dict.items():
        for word in k.split(' '):
            if word in train_vocab_dict:
                train_vocab_dict[word] += 1
            else:
                train_vocab_dict[word] = 1

    return train_dict, test_dict, vocab_dict_freq


def select_support_set(query_dict, vocab_dict_freq, k_shot = 1):
    support_dict = {}
    pred_freq = {}
    query_list = list(query_dict.items())
    np.random.shuffle(query_list)
    new_query_dict = {}
    for template, example_list in query_list:
        new_query_dict[template] = [e for e in example_list]

    for template, example_list in new_query_dict.items():
        if len(example_list) == 1:
            print(example_list)
            continue
        flag = True
        for word in template.split(' '):
            if word in vocab_dict_freq:
                if word in pred_freq:
                    if pred_freq[word] < k_shot:
                        pred_freq[word] += 1
                    else:
                        flag = False
                        break
                else:
                    pred_freq[word] = 1
                break
        if not flag:
            continue
        np.random.shuffle(example_list)
        support_dict[template] = []
        support_dict[template].append(example_list.pop())


    return support_dict, new_query_dict

def split_by_ratio(query_dict, ratio=0.5):
    return dict(list(query_dict.items())[int(len(query_dict) * ratio):]), dict(
        list(query_dict.items())[:int(len(query_dict) * ratio)])

def query_split(data_set, num_non_overlap=5, filter_one = True, data_type='job_prolog', predicate_list = []):
    query_dict = {}
    ori_c = 0
    for e in data_set:
        template = " ".join([str(token) for token in e.to_lambda_template.split(' ')])
        if template in query_dict:
            query_dict[template].append(e)
            ori_c += 1
        else:
            query_dict[template] = []
            query_dict[template].append(e)
            ori_c += 1
    print ("original dataset has {}".format(ori_c))
    filter_c = 0
    res_query_dict = {}
    if filter_one:
        for k, v in query_dict.items():
            if len(v) > 1:
                res_query_dict[k] = v
                filter_c += len(v)
    else:
        res_query_dict = query_dict
    print ("filtered dataset has {}".format(filter_c))

    train, test, vocab_dict_freq = split_by_new_token(res_query_dict, num_non_overlap = num_non_overlap, data_type=data_type, predicate_list = predicate_list)

    return train, test, vocab_dict_freq

def write_align_file(data_set, file_path, dump_path_prefix):
    f = open(os.path.join(dump_path_prefix, file_path), 'w')
    for template, examples in data_set.items():
        for example in examples:
            f.write(' '.join(example.src_sent) + ' ||| ' + ' '.join([str(token) for token in example.tgt_actions]) + '\n')
    f.close()

def write_train_test_file(data_set, file_path, dump_path_prefix):
    f = open(os.path.join(dump_path_prefix, file_path), 'w')
    length = 0
    for template, examples in data_set.items():
        length += len(examples)
        for example in examples:
            f.write(' '.join(example.src_sent) + '\t' + ' '.join(example.tgt_code)+ '\n')
    f.close()
    print ("length is", length)

def query_split_data(train_file, dev_file, test_file, dump_path_prefix, lang = 'lambda', data_type = 'geo_lambda', num_non_overlap = 5, predicate_list = []):
    train_set, train_action2id = produce_data(train_file,data_type = data_type, lang = lang)
    test_set, test_action2id = produce_data(test_file,data_type = data_type, lang = lang)
    whole_dataset = train_set + test_set
    if dev_file:
        dev_set, dev_action2id = produce_data(dev_file, data_type=data_type, lang=lang)
        whole_dataset += dev_set
    train_set, test, vocab_dict_freq = query_split(whole_dataset, num_non_overlap=num_non_overlap,data_type=data_type, predicate_list = predicate_list)
    for i in range(5):
        for j in range(2):
            print ("=====================================================================")
            print ("Shuffle ", i)
            print ("Shot ", j+1)
            support_set, query_set = select_support_set(test, vocab_dict_freq, k_shot=j + 1)
            # train
            #write_align_file(train_set, 'align_train.txt')
            file_path = dump_path_prefix + "shuffle_{}_shot_{}".format(i, j + 1)
            generate_dir(file_path)
            print ("train set template length", len(train_set))
            write_train_test_file(train_set, 'train.txt', file_path)

            print ("support set template length", len(support_set))
            write_train_test_file(support_set, 'support.txt', file_path)
            print ("query set template length", len(query_set))
            write_train_test_file(query_set, 'query.txt', file_path)