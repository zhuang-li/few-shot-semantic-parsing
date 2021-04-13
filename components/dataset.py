# coding=utf-8
from collections import namedtuple

import numpy as np
import pickle

import torch

from grammar.vertex import RuleVertex, CompositeTreeVertex
from model import nn_utils
from grammar.rule import extract_action_lit
from common.utils import cached_property
from grammar.action import *
import random
from math import gcd
import operator
from collections import Counter
from grammar.utils import is_var
from grammar.consts import VAR_NAME, IMPLICIT_HEAD, ROOT
import re

class Dataset(object):
    def __init__(self, examples):
        self.examples = examples

        self.domain = None
        self.known_domains = []
        self.known_test_data_length = []

        (self.class_examples, self.class_idx) = self.generate_class_examples()

        # for l, examples in self.class_examples.items():
            #print (l)
            #print (len(examples))

        (self.template_map, self.template_instances) = self._read_by_templates(self.examples)

    def _read_by_templates(self, examples):
        template_map = dict()
        template_instances = []
        for example in examples:
            template = example.tgt_ast.to_lambda_expr if example.tgt_ast.to_lambda_expr else example.to_logic_form
            # example.tgt_ast_t_seq.to_lambda_expr
            template_instances.append(template)
            if template in template_map:
                template_map[template].append(example)
            else:
                template_map[template] = []
                template_map[template].append(example)
        return template_map, template_instances

    def generate_class_examples(self):
        class_examples = {}
        class_idx = {}
        # print (len(examples))
        for idx, e in enumerate(self.examples):
            # print (set(e.tgt_actions))
            for action in set(e.tgt_actions):
                if action in class_examples:
                    class_examples[action].append(e)
                    class_idx[action].append(idx)
                else:
                    class_examples[action] = []
                    class_examples[action].append(e)
                    class_idx[action] = []
                    class_idx[action].append(idx)

        for lab, lab_examples in class_examples.items():
            # print (len(lab_examples))
            # print (lab)
            lab_examples.sort(key=lambda e: -len(e.src_sent))
        return class_examples, class_idx

    @property
    def all_source(self):
        return [e.src_sent for e in self.examples]

    @property
    def all_targets(self):
        return [e.tgt_code for e in self.examples]

    @property
    def all_actions(self):
        return [e.tgt_actions for e in self.examples]

    @staticmethod
    def from_bin_file(file_path):
        examples = pickle.load(open(file_path, 'rb'))
        return Dataset(examples)

    @staticmethod
    def from_bin_file_list(file_path_list):
        examples = []
        for file_path in file_path_list:
            examples.extend(pickle.load(open(file_path, 'rb')))
        return Dataset(examples)

    def add(self, new_dataset):
        self.examples.extend(new_dataset.examples)
        self.class_examples = self.generate_class_examples()
        self.template_map, self.template_instances = self._read_by_templates(self.examples)

    def add_examples(self, new_examples):
        self.examples.extend(new_examples)
        self.class_examples = self.generate_class_examples()
        self.template_map, self.template_instances = self._read_by_templates(self.examples)

    def batch_iter(self, batch_size, shuffle=False, sort=True):
        index_arr = np.arange(len(self.examples))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.examples) / float(batch_size)))
        for batch_id in range(batch_num):
            #print (batch_id)
            batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
            batch_examples = [self.examples[i] for i in batch_ids]
            if sort:
                batch_examples.sort(key=lambda e: -len(e.src_sent))

            yield batch_examples

    def random_sample_batch_iter(self, sample_size):
        index_arr = np.arange(len(self.examples))
        # print (index_arr)
        np.random.shuffle(index_arr)

        batch_ids = index_arr[:sample_size]
        # print(batch_ids)
        batch_examples = [self.examples[i] for i in batch_ids]
        # batch_examples.sort(key=lambda e: -len(e.src_sent))
        """
        template_count = Counter(
            [e.tgt_ast_t_seq.to_lambda_expr if e.tgt_ast_t_seq.to_lambda_expr else e.to_logic_form for e in
             self.examples])

        instance_count = [template_count[e.tgt_ast_t_seq.to_lambda_expr] for e in batch_examples]
        print(template_count.values())
        print (instance_count)
        """
        return batch_examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

class ContinuumDataset(object):
    def __init__(self, args, datasets, vocabs):
        self.datasets = datasets
        self.vocabs = vocabs
        self.complete_examples = []
        known_domain_length = int(args.num_known_domains)

        assert known_domain_length < len(datasets), "known domain number should be smaller than the number of all domains"

        # for v in self.vocabs:
            # print (v.t_action.token2id)
            # print(len(v.t_action))

        dataset_number = len(datasets)

        self.task_permutation = []

        if args.shuffle_tasks:
            torch.manual_seed(args.seed)
            self.dataset_permutation_indx = torch.randperm(dataset_number).tolist()

        # print (self.dataset_permutation_indx)

        self.known_domains = []
        self.known_test_data_length = []
        known_datasets = datasets[self.dataset_permutation_indx[0]]

        self.known_domains.append(known_datasets.domain)
        self.known_test_data_length.append(len(known_datasets.test.examples))
        self.vocab = self.vocabs[self.dataset_permutation_indx[0]]

        self.t_nc_per_task = {}

        self.nt_nc_per_task = {}



        #print (known_domain_length)

        for permute_indx in self.dataset_permutation_indx[1:known_domain_length]:
            known_datasets.train.add(datasets[permute_indx].train)
            known_datasets.test.add(datasets[permute_indx].test)
            self.known_test_data_length.append(len(known_datasets.test.examples))
            self.known_domains.append(datasets[permute_indx].domain)

            if known_datasets.dev:
                known_datasets.dev.add(datasets[permute_indx].dev)

            self.vocab = merge_vocab_entry(self.vocab, self.vocabs[permute_indx])

        known_datasets.test.known_domains = self.known_domains
        known_datasets.test.known_test_data_length = self.known_test_data_length

        self.t_nc_per_task[0] = len(self.vocab.t_action) - 1

        self.nt_nc_per_task[0] = len(self.vocab.nt_action)

        self.task_permutation.append(known_datasets)
        self.complete_examples.append(known_datasets.test.examples)
        self.following_domains = []

        for idx, permutate_indx in enumerate(self.dataset_permutation_indx[known_domain_length:]):

            self.task_permutation.append(datasets[permutate_indx])
            self.complete_examples.append(datasets[permutate_indx].test.examples)
            self.following_domains.append(datasets[permutate_indx].domain)
            datasets[permutate_indx].test.domain = datasets[permutate_indx].domain
            datasets[permutate_indx].test.known_test_data_length.append(len(datasets[permutate_indx].test.examples))

            prev_vocab_length = len(self.vocab.t_action)
            # print (prev_vocab_length)

            permute_vocab_length = len(self.vocabs[permutate_indx].t_action) - 1

            # print(self.vocabs[permutate_indx].t_action.token2id)

            self.vocab = merge_vocab_entry(self.vocab, self.vocabs[permutate_indx])

            # print (len(self.vocab.t_action))

            self.t_nc_per_task[idx + 1] = len(self.vocab.t_action) - prev_vocab_length

            self.nt_nc_per_task[idx + 1] = len(self.vocab.nt_action)

            assert permute_vocab_length == self.t_nc_per_task[idx + 1], "permute_vocab_length: {0}, t_nc_per_task: {1}".format(permute_vocab_length, self.t_nc_per_task[idx + 1])

        print (self.known_domains)
        print (self.following_domains)
        print (self.t_nc_per_task)
        print (self.nt_nc_per_task)

        last_task_data = self.task_permutation[0].train
        print ("Task template number is : ", len(last_task_data.template_map.keys()))
        print ("Dataset number is : ", len(last_task_data))
        # print ("Template instance freq is : ")
        # print (Counter(last_task_data.template_instances))
        # print ("Logical form freq is : ")
        # print (Counter([e.to_logic_form for e in last_task_data.examples]))
        # print ("Task specific action freq is : ")
        # print (Counter([" ".join(e.tgt_ast_t_seq) for e in last_task_data.examples]))

        # print (len(Counter([" ".join(e.tgt_ast_t_seq) for e in last_task_data.examples]).items()))

        for idx, task_data in enumerate(self.task_permutation[1:]):

            last_templates = last_task_data.template_map.keys()

            current_templates = task_data.train.template_map.keys()
            print("Task template number is : ", len(current_templates))
            print("Dataset number is : ", len(task_data.train))
            print ("Overlap instance number : ", len(list(set(last_templates) & set(current_templates))))
            last_task_data = task_data.train

        for idx, out_task_data in enumerate(self.datasets):
            print ("=======================================")
            last_templates = out_task_data.train.template_map.keys()
            print("Out Task template number is : ", len(last_templates))
            for idx, in_task_data in enumerate(self.datasets):
                current_templates = in_task_data.train.template_map.keys()
                print("In Task template number is : ", len(current_templates))
                print("Overlap instance number : ", len(list(set(last_templates) & set(current_templates))))

            #print (list(set(last_templates) & set(current_templates)))

            # print("Template instance freq is : ")
            # print(Counter(task_data.train.template_instances))
            # print("Logical form freq is : ")
            # print(Counter([e.to_logic_form for e in task_data.train.examples]))

            # print("Task specific action freq is : ")
            # print(Counter([" ".join(e.tgt_ast_t_seq) for e in task_data.train.examples]))
            # print(len(Counter([" ".join(e.tgt_ast_t_seq) for e in task_data.train.examples]).items()))


        # print (self.nc_per_task)
        # print (self.vocab.t_action.token2id)

    def __len__(self):
        return len(self.datasets)

    def __iter__(self):
        return iter(self.task_permutation)


    @staticmethod
    def read_continuum_data(args, train_file_path_list, test_file_path_list, dev_file_path_list, vocab_path_list, domains):
        datasets = []
        vocabs = []
        for idx, train_file_path in enumerate(train_file_path_list):
            DataCollection = namedtuple('DataCollection', 'train test dev domain')
            test_file_path = test_file_path_list[idx]
            domain = domains[idx]
            if dev_file_path_list:
                dev_file_path = dev_file_path[idx]
                data_collection = DataCollection(Dataset.from_bin_file(train_file_path),
                                                 Dataset.from_bin_file(test_file_path),
                                                 Dataset.from_bin_file(dev_file_path),
                                                 domain)
            else:
                data_collection = DataCollection(Dataset.from_bin_file(train_file_path),
                                                 Dataset.from_bin_file(test_file_path),
                                                 None,
                                                 domain)

            vocabs.append(pickle.load(open(vocab_path_list[idx], 'rb')))
            datasets.append(data_collection)

        #datasets.sort(key=lambda item: (-len(item.train), item))

        index = [i[0] for i in sorted(enumerate(datasets), key=lambda x: -len(x[1].train))]
        # print (index)
        datasets = [datasets[idx] for idx in index]
        vocabs = [vocabs[idx] for idx in index]

        print([len(dataset.train) for dataset in datasets])
        if args.mode == 'train':
            datasets = datasets#[:-2]
            vocabs = vocabs#[:-2]
        elif args.mode == 'valid':
            # main_domain = datasets[:args.num_known_domains]
            # main_vocab = vocabs[:args.num_known_domains]
            continual_domain = datasets[4:]
            continual_vocab = vocabs[4:]
            assert len(continual_domain) == 4
            # assert len(main_domain) == args.num_known_domains
            datasets = continual_domain
            vocabs = continual_vocab
        else:
            raise ValueError
        # print ([len(dataset.train) for dataset in datasets])

        return ContinuumDataset(args, datasets, vocabs)

def get_template_values(replay_examples, template_type = 'general'):
    if template_type == 'general':
        template_count = Counter([e.tgt_ast.to_lambda_expr if e.tgt_ast.to_lambda_expr else e.to_logic_form for e in replay_examples])
    elif template_type == 'specific':
        template_count = Counter(
            [e.tgt_ast_t_seq.to_lambda_expr if e.tgt_ast_t_seq.to_lambda_expr else e.to_logic_form for e in replay_examples])

    template_example_dict = {}

    for example in replay_examples:
        if template_type == 'general':
            t_template = example.tgt_ast.to_lambda_expr if example.tgt_ast.to_lambda_expr else example.to_logic_form
        else:
            t_template = example.tgt_ast_t_seq.to_lambda_expr if example.tgt_ast_t_seq.to_lambda_expr else example.to_logic_form

        if t_template in template_example_dict:
            template_example_dict[t_template].append(example)
        else:
            template_example_dict[t_template] = []
            template_example_dict[t_template].append(example)

    return template_count, template_example_dict

def sample_balance_data(replay_examples, n_mem, template_type='general'):
    template_count, template_example_dict = get_template_values(replay_examples, template_type=template_type)

    # print (lcm)
    # print (int_count_list)
    # print (template_count)


    rebalance_examples = []

    while len(rebalance_examples) < n_mem:
        #print (count)
        template_index = random.randint(0, len(template_count.most_common()) - 1)
        template, count = template_count.most_common()[template_index]
        # print(template)
        template_examples = template_example_dict[template]
        if len(template_examples) > 0:
            #if template_type=='specific':
            index = random.randint(0, len(template_examples) - 1)
            rebalance_examples.append(template_examples.pop(index))
            #else:
            #    rebalance_examples.extend(sample_balance_data(replay_examples, 1, template_type='specific'))


    return rebalance_examples

def reweight_t_data(replay_examples):
    template_count = Counter([e.tgt_ast.to_lambda_expr if e.tgt_ast.to_lambda_expr else e.to_logic_form for e in replay_examples])

    # print (template_count)
    # print (len(template_count.items()))

    # template_count = Counter([e.to_lambda_template for e in replay_examples])
    # print (template_count)
    # print (len(template_count.items()))

    count_list = list(template_count.values())
    int_count_list = [int(i) for i in count_list]
    lcm = int_count_list[0]
    for i in int_count_list[1:]:
        lcm = lcm * i // gcd(lcm, i)

    template_example_dict = {}

    for example in replay_examples:

        t_template = example.tgt_ast.to_lambda_expr if example.tgt_ast.to_lambda_expr else example.to_logic_form

        if t_template in template_example_dict:
            template_example_dict[t_template].append(example)
        else:
            template_example_dict[t_template] = []
            template_example_dict[t_template].append(example)

    # print (lcm)
    # print (int_count_list)
    # print (template_count)


    rebalance_examples = []

    for template, count in template_count.items():
        template_examples = template_example_dict[template]
        multiple = int(lcm/count)
        # print (multiple)
        for i in range(multiple):
            for multi_e in template_examples:
                rebalance_examples.append(multi_e.copy())
    # print (len(rebalance_examples))
    return rebalance_examples

def reweight_data(replay_examples):
    template_count = Counter([e.to_lambda_template for e in replay_examples])

    count_list = list(template_count.values())
    int_count_list = [int(i) for i in count_list]
    lcm = int_count_list[0]
    for i in int_count_list[1:]:
        lcm = lcm * i // gcd(lcm, i)

    template_example_dict = {}

    for example in replay_examples:
        if example.to_lambda_template in template_example_dict:
            template_example_dict[example.to_lambda_template].append(example)
        else:
            template_example_dict[example.to_lambda_template] = []
            template_example_dict[example.to_lambda_template].append(example)

    # print (lcm)
    # print (int_count_list)
    # print (template_count)


    rebalance_examples = []

    for template, count in template_count.items():
        template_examples = template_example_dict[template]
        multiple = int(lcm/count)
        # print (multiple)
        for i in range(multiple):
            for multi_e in template_examples:
                rebalance_examples.append(multi_e.copy())
    # print (len(rebalance_examples))
    return rebalance_examples



def generate_concat_samples(replay_examples, current_examples):
    augment_examples = []

    for e in replay_examples:
        # if str(e) == "what is the gender of an employee who doe not have mckinsey a an employer":
            # print ("debug")
        rand_int = random.randint(0, len(current_examples) - 1)

        abe = current_examples[rand_int]

        concat_e = Example(abe.src_sent + e.src_sent, abe.tgt_actions + e.tgt_actions, [], [])

        augment_examples.append(concat_e)


    return augment_examples

def generate_augment_samples(examples, augment_memory_dict):

    augment_examples = []

    for e in examples:
        # if str(e) == "what is the gender of an employee who doe not have mckinsey a an employer":
            # print ("debug")
        if e in augment_memory_dict:

            rand_int = random.randint(0, len(augment_memory_dict[e]) - 1)

            abe = augment_memory_dict[e][rand_int]

            concat_e = Example(abe.src_sent + e.src_sent, abe.tgt_actions + e.tgt_actions, [], [])

            augment_examples.append(concat_e)
        else:
            augment_examples.append(e)

    return augment_examples

def data_augmentation(examples, t_vocab):
    augmentated_dict = dict()

    for e in examples:
        augmentated_instances = []

        # if str(e) == "what is the gender of an employee who doe not have mckinsey a an employer":
            # print ("debug")

        init_example = Example(e.src_sent, [], [], [])

        augmentated_instances.append(init_example)

        instance_sub_index = [[False]]
        segment_complete_sent = [[" ".join(e.src_sent)]]

        replaced_actions = set()
        for action in e.tgt_actions:
            is_dup = False
            if action in replaced_actions:
                is_dup = True
            else:
                replaced_actions.add(action)

            replace_examples = []

            replace_sub_index = []

            replace_complete_sent = []

            for idx, aug_e in enumerate(augmentated_instances):
                if isinstance(action, GenTAction) and action.type in t_vocab.entype2action:
                    for replace_action in t_vocab.entype2action[action.type]:
                        if is_dup and replace_action not in aug_e.tgt_actions:
                            continue

                        if action == replace_action:
                            temp_e = Example(aug_e.src_sent.copy(), [action.copy() for action in aug_e.tgt_actions], [],
                                             [])
                            temp_e.tgt_actions.append(action.copy())
                            replace_examples.append(temp_e)

                            replace_sub_index.append([replace_flag for replace_flag in instance_sub_index[idx]])

                            replace_complete_sent.append([segment for segment in segment_complete_sent[idx]])

                        else:

                            current_nl_seg_list = t_vocab.action2nl[action]

                            if is_dup:
                                current_nl_seg_list = t_vocab.action2nl[replace_action]

                            replace_nl_seg_list = t_vocab.action2nl[replace_action]

                            # complete_sent = " ".join(aug_e.src_sent)

                            partial_complete_sent = ""
                            for sub_indx, index_flag in enumerate(instance_sub_index[idx]):
                                if not index_flag:
                                    partial_complete_sent += segment_complete_sent[idx][sub_indx]

                            is_in_sent = False
                            current_nl_seg = ""
                            for nl_seg in current_nl_seg_list:
                                if nl_seg in partial_complete_sent:
                                    if len(current_nl_seg) < len(nl_seg):
                                        current_nl_seg = nl_seg
                                    is_in_sent = True


                            if is_in_sent:

                                for nl_seg in replace_nl_seg_list:
                                    if len(nl_seg.split(' ')) >= len(current_nl_seg.split(' ')):
                                        temp_e = Example(aug_e.src_sent.copy(),
                                                         [action.copy() for action in aug_e.tgt_actions], [],
                                                         [])
                                        replace_nl_seg = nl_seg

                                        replaced_sent_list = []

                                        replace_flag = []

                                        for sub_indx, index_flag in enumerate(instance_sub_index[idx]):
                                            if index_flag:
                                                replaced_sent_list.append(segment_complete_sent[idx][sub_indx])
                                                replace_flag.append(True)
                                            else:
                                                import re
                                                last_index = 0
                                                current_seg = segment_complete_sent[idx][sub_indx]
                                                for m in re.finditer(current_nl_seg, current_seg):

                                                    start_index = m.start()
                                                    end_index = m.end()

                                                    if not start_index == 0:
                                                        replaced_sent_list.append(current_seg[last_index:start_index])
                                                        replace_flag.append(False)
                                                    replaced_sent_list.append(replace_nl_seg)
                                                    replace_flag.append(True)
                                                    last_index = end_index
                                                if not last_index == len(current_seg):
                                                    replaced_sent_list.append(current_seg[last_index:])
                                                    replace_flag.append(False)

                                        temp_e.src_sent = "".join(replaced_sent_list).split(' ')

                                        copy_action = replace_action.copy()

                                        copy_action.parent_t = replace_action.parent_t

                                        temp_e.tgt_actions.append(copy_action)

                                        replace_examples.append(temp_e)

                                        replace_sub_index.append(replace_flag)

                                        replace_complete_sent.append(replaced_sent_list)


                            else:
                                continue
                else:
                    temp_e = Example(aug_e.src_sent.copy(), [action.copy() for action in aug_e.tgt_actions], [],
                                     [])
                    temp_e.tgt_actions.append(action.copy())
                    replace_examples.append(temp_e)

                    replace_sub_index.append([replace_flag for replace_flag in instance_sub_index[idx]])

                    replace_complete_sent.append([segment for segment in segment_complete_sent[idx]])


            augmentated_instances = replace_examples

            instance_sub_index = replace_sub_index

            segment_complete_sent = replace_complete_sent
        # if len(augmentated_instances) > 1:
        filterd_instances = set()

        for instance in augmentated_instances:
            if str(instance) == str(e):
                continue
            filterd_instances.add(instance)

        if len(filterd_instances) > 0:
            augmentated_dict[e] = list(filterd_instances)
    return augmentated_dict

class ContinualDataset(Dataset):
    def __init__(self, examples):
        Dataset.__init__(self, examples)
        # template map
        self.template_instances = self._read_by_templates(examples)

    # new functions
    def add(self, new_dataset):
        self.examples.extend(new_dataset.examples)
        self.template_instances = self._read_by_templates(self.examples)

    def _read_by_templates(self, examples):
        template_instances = dict()
        for example in examples:
            template = example.to_lambda_template
            if template in template_instances:
                template_instances[template].append(example)
            else:
                template_instances[template] = []
                template_instances[template].append(example)
        return template_instances

    @staticmethod
    def from_bin_file(file_path):
        examples = pickle.load(open(file_path, 'rb'))
        return ContinualDataset(examples)

    @staticmethod
    def from_bin_file_list(file_path_list):
        examples = []
        for file_path in file_path_list:
            examples.extend(pickle.load(open(file_path, 'rb')))
        return ContinualDataset(examples)


    def template_sample_batch_iter(self, sample_size, previous_template_samples):

        index_arr = np.arange(len(previous_template_samples))
        np.random.shuffle(index_arr)

        batch_ids = index_arr[0: sample_size]
        template_examples = [previous_template_samples[i] for i in batch_ids]

        batch_examples = []
        visited_examples = set()
        example_rand_inds = torch.randperm(len(self.examples)).tolist()

        idx = 0
        for example in template_examples:
            previous_template = example.to_lambda_template
            if previous_template in self.template_instances:
                template_examples = self.template_instances[previous_template]
                rand_inds = torch.randperm(len(template_examples)).tolist()
                sample = template_examples[rand_inds[0]]
                visited_examples.add(id(sample.to_logic_form))
                batch_examples.append(sample)
            else:
                sample = self.examples[example_rand_inds[idx]]
                #print (visited_examples)
                while id(sample.to_logic_form) in visited_examples:
                    idx += 1
                    #print (idx)
                    sample = self.examples[example_rand_inds[idx]]
                visited_examples.add(id(sample.to_logic_form))
                batch_examples.append(sample)
                idx += 1

        batch_examples.sort(key=lambda e: -len(e.src_sent))
        #for e in batch_examples:
         #   print (len(e.src_sent))
        return batch_examples



class EposideDataset(object):
    def __init__(self, examples):
        self.examples = examples
        self.template_instances = self._read_by_templates(examples)

    def _read_by_templates(self, examples):
        template_instances = dict()
        for example in examples:
            template = " ".join([str(action) for action in example.tgt_actions])
            if template in template_instances:
                template_instances[template].append(example)
            else:
                template_instances[template] = []
                template_instances[template].append(example)
        return template_instances

    @property
    def all_source(self):
        return [e.src_sent for e in self.examples]

    @property
    def all_targets(self):
        return [e.tgt_code for e in self.examples]

    @property
    def all_actions(self):
        return [e.tgt_actions for e in self.examples]

    @staticmethod
    def from_bin_file(file_path):
        examples = pickle.load(open(file_path, 'rb'))
        return EposideDataset(examples)

    def batch_iter(self, n_way, k_shot, query_num, shuffle=False):
        index_arr = np.arange(len(self.template_instances.keys()))
        if shuffle:
            np.random.shuffle(index_arr)

        batch_num = int(np.ceil(len(self.template_instances.keys()) / float(n_way)))
        #print (len(self.template_instances.keys()))
        for batch_id in range(batch_num):
            batch_ids = index_arr[n_way * batch_id: n_way * (batch_id + 1)]
            template_examples = [list(self.template_instances.keys())[i] for i in batch_ids]
            template_examples.sort(key=lambda template: -len(template.split(' ')))
            support_batch = []
            query_batch = []
            for template in template_examples:
                type_examples = self.template_instances[template]
                #print (len(type_examples))
                #if len(type_examples) == 1:
                    #print (type_examples[0].src_sent)
                rand_inds = torch.randperm(len(type_examples))
                #print (rand_inds)
                support_inds = rand_inds[:k_shot]
                query_inds = rand_inds[k_shot:k_shot+query_num]
                #print (query_inds)
                xs = [type_examples[s_index] for s_index in support_inds.tolist()]
                xq = [type_examples[q_index] for q_index in query_inds.tolist()]

                support_batch.extend(xs)
                query_batch.extend(xq)
                    #batch.append((xs, xq))
            support_batch.sort(key=lambda e: -len(e.src_sent))
            query_batch.sort(key=lambda e: -len(e.src_sent))
            #print (len(query_batch))
            yield support_batch, query_batch

    def task_batch_iter(self, n_way, k_shot, query_num, task_num = 4, shuffle=False):
        for i in range(task_num):
            index_arr = np.arange(len(self.template_instances.keys()))
            if shuffle:
                np.random.shuffle(index_arr)

            batch_ids = index_arr[:n_way]
            template_examples = [list(self.template_instances.keys())[i] for i in batch_ids]
            template_examples.sort(key=lambda template: -len(template.split(' ')))
            support_batch = []
            query_batch = []
            for template in template_examples:
                type_examples = self.template_instances[template]
                #print (len(type_examples))
                #if len(type_examples) == 1:
                    #print (type_examples[0].src_sent)
                rand_inds = torch.randperm(len(type_examples))
                #print (rand_inds)
                support_inds = rand_inds[:k_shot]
                query_inds = rand_inds[k_shot:k_shot+query_num]
                #print (query_inds)
                xs = [type_examples[s_index] for s_index in support_inds.tolist()]
                xq = [type_examples[q_index] for q_index in query_inds.tolist()]

                support_batch.extend(xs)
                query_batch.extend(xq)
                    #batch.append((xs, xq))
            support_batch.sort(key=lambda e: -len(e.src_sent))
            query_batch.sort(key=lambda e: -len(e.src_sent))
            #print (len(query_batch))
            yield support_batch, query_batch

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

class Example(object):
    def __init__(self, src_sent, tgt_actions, tgt_code, tgt_ast, idx=0, meta=None):
        # str list
        self.src_sent = src_sent
        # str list
        self.tgt_code = tgt_code
        # vertext root
        self.tgt_ast = tgt_ast
        # action sequence
        self.tgt_actions = tgt_actions

        self.idx = idx
        self.meta = meta


    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return " ".join(self.src_sent)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def copy(self):
        new_src_sent = [token for token in self.src_sent]
        new_tgt_code = [token for token in self.tgt_code]
        new_tgt_ast = self.tgt_ast.copy()
        new_tgt_actions = [action.copy() for action in self.tgt_actions]
        new_idx = self.idx
        new_meta = self.meta

        return Example(new_src_sent, new_tgt_actions ,new_tgt_code,  new_tgt_ast, idx=new_idx, meta=new_meta)


    @property
    def to_prolog_template(self):
        return self.tgt_ast.to_prolog_expr

    @property
    def to_lambda_template(self):
        return self.tgt_ast.to_lambda_expr


    @property
    def to_logic_form(self):
        tgt_code = list(self.tgt_code)
        if not (self.tgt_code[0] == '(' and self.tgt_code[-1] == ')'):
            tgt_code = ['('] + tgt_code + [')']
        return " ".join([str(code) for code in tgt_code])

    @property
    def tgt_ast_seq(self):
        tgt_ast_seq = []
        for action in self.tgt_actions:
            if isinstance(action, GenAction):

                tgt_ast_seq.append(action.vertex)
            elif isinstance(action, ReduceAction):
                tgt_ast_seq.append(action.rule.head)
            else:
                raise ValueError

        return tgt_ast_seq

    @property
    def tgt_ast_t_seq(self):
        tgt_ast_t_seq = self.tgt_ast.copy()

        visited, queue = set(), [tgt_ast_t_seq]

        while queue:
            vertex = queue.pop(0)
            v_id = id(vertex)
            visited.add(v_id)
            idx = 0
            for child in vertex.children:
                if id(child) not in visited:
                    parent_node = vertex
                    if isinstance(child, RuleVertex):
                        child_vertex = child

                        if child_vertex.original_var is not None:
                            parent_node.children[idx] = child_vertex.original_var
                            child_vertex.original_var.parent = parent_node
                        else:
                            child_vertex.head = ROOT
                            child_vertex.is_auto_nt = True
                            child.is_auto_nt = True
                            queue.append(child_vertex)
                    elif isinstance(child, CompositeTreeVertex):
                        self.reduce_comp_node(child.vertex)

                idx += 1

        return tgt_ast_t_seq

    def reduce_comp_node(self, vertex):
        visited, queue = set(), [vertex]

        while queue:
            vertex = queue.pop(0)
            v_id = id(vertex)
            visited.add(v_id)
            idx = 0
            for child in vertex.children:
                if id(child) not in visited:
                    parent_node = vertex
                    child_vertex = child

                    if child_vertex.original_var is not None:
                        parent_node.children[idx] = child_vertex.original_var
                        child_vertex.original_var.parent = parent_node
                    else:
                        queue.append(child_vertex)
                idx += 1



class Batch(object):
    def __init__(self, examples, vocab, training=True, append_boundary_sym=True, use_cuda=False, data_type='overnight'):
        self.examples = examples

        # source token seq
        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_len = [len(e.src_sent) + 2 if append_boundary_sym else len(e.src_sent) for e in self.examples]

        # target token seq
        self.tgt_code = [e.tgt_code for e in self.examples]
        self.tgt_code_len = [len(e.tgt_code) + 2 if append_boundary_sym else len(e.tgt_code) for e in self.examples]

        # action seq

        self.action_seq = [e.tgt_actions for e in self.examples]
        self.action_seq_len = [len(e.tgt_actions) + 1 if append_boundary_sym else len(e.tgt_actions) for e in
                               self.examples]
        self.max_action_num = max(self.action_seq_len)
            # max(self.action_seq_len)

        # max(self.action_seq_len)

        # action seq
        self.max_ast_seq_num = max(len(e.tgt_ast_seq) for e in self.examples)
        self.ast_seq = [e.tgt_ast_seq for e in self.examples]
        self.ast_seq_len = [len(e.tgt_ast_seq) + 1 if append_boundary_sym else len(e.tgt_ast_seq) for e in
                               self.examples]

        self.entity_seq = []
        for seq in self.action_seq:
            self.entity_seq.append([])
            for action in seq:
                self.entity_seq[-1].extend(action.entities)

        self.variable_seq = []
        for seq in self.action_seq:
            self.variable_seq.append([])
            for action in seq:
                self.variable_seq[-1].extend(action.variables)

        self.vocab = vocab
        self.use_cuda = use_cuda
        self.training = training
        self.append_boundary_sym = append_boundary_sym


        # init overnight index
        if data_type == 'overnight':
            self.init_overnight_index_tensors()

    def __len__(self):
        return len(self.examples)



    # source sentence
    @cached_property
    def src_sents_var(self):
        return nn_utils.to_input_variable(self.src_sents, self.vocab.source,
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode='token')


    def src_sents_span(self, predicate_tokens):
        return nn_utils.to_src_sents_span(self.src_sents, predicate_tokens)


    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len,
                                                    use_cuda=self.use_cuda)

    # action sequence
    @cached_property
    def action_seq_var(self):
        #print (self.action_seq)
        return nn_utils.to_input_variable(self.action_seq, self.vocab.action,
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode='action')

    # action sequence
    @cached_property
    def action_seq_pad(self):
        return nn_utils.to_seq_pad(self.action_seq, append_boundary_sym=True, mode ='action')

    @cached_property
    def action_seq_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.action_seq_len,
                                                    use_cuda=self.use_cuda)

    # general action sequence
    @cached_property
    def general_action_seq_var(self):

        return nn_utils.to_input_variable(self.action_seq, self.vocab.general_action,
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode='action')

    @cached_property
    def general_action_seq_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.action_seq_len,
                                                    use_cuda=self.use_cuda)

    # target sequence
    @cached_property
    def tgt_seq_var(self):
        return nn_utils.to_input_variable(self.tgt_code, self.vocab.code,
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode='token')

    @cached_property
    def tgt_seq_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.tgt_code_len,
                                                    use_cuda=self.use_cuda)

    # hierarchy target sequence
    @cached_property
    def hierarchy_action_seq_var(self):
        return nn_utils.to_hierarchy_input_variable(self.action_seq,
                                                    {GenAction.__name__:  self.vocab.gen_action,
                                                     ReduceAction.__name__: self.vocab.re_action},
                                                    use_cuda=self.use_cuda,
                                                    append_boundary_sym=False)

    # ast sequence
    @cached_property
    def ast_seq_var(self):
        return nn_utils.to_input_variable(self.ast_seq, self.vocab.vertex,
                                          use_cuda=self.use_cuda, append_boundary_sym=self.append_boundary_sym,
                                          mode='ast')

    @cached_property
    def ast_seq_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.ast_seq_len, use_cuda=self.use_cuda)

    @cached_property
    def entity_seq_var(self):
        return nn_utils.to_input_variable(self.entity_seq, self.vocab.entity,
                                          use_cuda=self.use_cuda, append_boundary_sym=True,
                                          mode='lit')

    @cached_property
    def variable_seq_var(self):
        return nn_utils.to_input_variable(self.variable_seq, self.vocab.variable,
                                          use_cuda=self.use_cuda, append_boundary_sym=True,
                                          mode='lit')

    def init_overnight_index_tensors(self):
        self.nt_action_idx_matrix = []
        self.nt_action_mask = []
        self.t_action_idx_matrix = []
        self.t_action_mask = []

        for t in range(self.max_action_num):
            nt_action_idx_row = []
            nt_action_mask_row = []
            t_action_idx_matrix_row = []
            t_action_mask_row = []

            for e_id, e in enumerate(self.examples):
                nt_action_idx = nt_action_mask = t_action_idx = t_action_mask = 0
                if t < len(e.tgt_actions):
                    action = e.tgt_actions[t]

                    if isinstance(action, GenNTAction):
                        nt_action_idx = self.vocab.nt_action.token2id[action]

                        # assert self.grammar.id2prod[app_rule_idx] == action.production
                        nt_action_mask = 1
                    elif isinstance(action, GenTAction):
                        #print (self.vocab.t_action.token2id)
                        #print (action)
                        t_action_idx = self.vocab.t_action.token2id[action]
                        #print (self.vocab.primitive.id2word[0])
                        t_action_mask = 1


                    else:
                        raise ValueError

                nt_action_idx_row.append(nt_action_idx)
                #print (app_rule_idx_row)
                nt_action_mask_row.append(nt_action_mask)

                t_action_idx_matrix_row.append(t_action_idx)
                t_action_mask_row.append(t_action_mask)

            #print ("================")
            #print (app_rule_idx_row)
            #print (token_row)
            self.nt_action_idx_matrix.append(nt_action_idx_row)
            self.nt_action_mask.append(nt_action_mask_row)

            self.t_action_idx_matrix.append(t_action_idx_matrix_row)
            self.t_action_mask.append(t_action_mask_row)



        T = torch.cuda if self.use_cuda else torch
        self.nt_action_idx_matrix = T.LongTensor(self.nt_action_idx_matrix)
        self.nt_action_mask = T.FloatTensor(self.nt_action_mask)
        self.t_action_idx_matrix = T.LongTensor(self.t_action_idx_matrix)
        self.t_action_mask = T.FloatTensor(self.t_action_mask)
