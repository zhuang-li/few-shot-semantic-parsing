import os

from preprocess_data.data_generation_utils import query_split_data

if __name__ == '__main__':
    overlap_predicate = 3
    predicate_list = ['salary_greater_than', 'recruiter', 'req_exp']
    path_prefix = "../../datasets/jobs/"
    dump_path_prefix = "../../datasets/jobs/query_split/few_shot_split_random_{}_predi".format(overlap_predicate)
    train_file = os.path.join(path_prefix, 'train.txt')
    test_file = os.path.join(path_prefix, 'test.txt')
    query_split_data(train_file, None, test_file, dump_path_prefix, lang = 'prolog', data_type = 'job_prolog', num_non_overlap = overlap_predicate, predicate_list = predicate_list)
    pass