import os
from glob import glob

from preprocess_data.data_generation_utils import query_split_data
from preprocess_data.utils import produce_data, generate_dir


def write_train_test_file(data_set, file_path, dump_path_prefix):
    f = open(os.path.join(dump_path_prefix, file_path), 'w')
    print (os.path.join(dump_path_prefix, file_path))
    length = 0
    for example in data_set:
        f.write(' '.join(example.src_sent) + '\t' + "( " + example.tgt_ast.to_lambda_expr + " )" + '\n')
    f.close()
    print ("length is", length)

if __name__ == '__main__':

    PATH = "../../datasets/jobs/jobss"

    result = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.txt'))]

    for file_path in result:
        data_set, train_temp_db = produce_data(file_path, 'job_prolog', 'prolog', turn_v_back=True)

        base_name_file = os.path.basename(file_path)

        out_put_file = base_name_file.split('.')[0]+ '_lambda.txt'

        dump_prefix = os.path.dirname(file_path)

        #print (dump_prefix)

        write_train_test_file(data_set, out_put_file, dump_prefix)

    """
    overlap_predicate = 3

    path_prefix = "../../datasets/jobs/"
    dump_path_prefix = "../../datasets/jobs/query_split_previous_5/few_shot_split_random_{}_predi".format(overlap_predicate)
    
    for j in range(2):
        instance_list = []
        for i in range(5):
            print ("=====================================================================")
            print ("Shuffle ", i)
            print ("Shot ", j+1)
            # train
            #write_align_file(train_set, 'align_train.txt')
            file_path = dump_path_prefix + "shuffle_{}_shot_{}".format(i, j + 1)
            dump_file_path = os.path.join(dump_path_prefix , "shuffle_{}_shot_{}".format(i, j + 1))
            generate_dir(dump_file_path)
            train_file = os.path.join(file_path, 'train.txt')
            support_file = os.path.join(file_path, 'support.txt')
            query_file = os.path.join(file_path, 'query.txt')

            train_set, train_temp_db = produce_data(train_file, 'job_prolog', 'prolog', turn_v_back=True)
            write_train_test_file(train_set, "train_lambda.txt", dump_file_path)
            support_set, support_temp_db = produce_data(support_file, 'job_prolog', 'prolog', turn_v_back=True)
            write_train_test_file(support_set, "support_lambda.txt", dump_file_path)
            test_set, test_temp_db = produce_data(query_file, 'job_prolog', 'prolog', turn_v_back=True)
            write_train_test_file(test_set, "query_lambda.txt", dump_file_path)
            if i == 0:
                for e in support_set:
                    instance_list.append(" ".join(e.src_sent))
                for e in test_set:
                    instance_list.append(" ".join(e.src_sent))
            else:
                target_list = instance_list
                instance_list = instance_list.copy()
                for e in support_set:
                    target_list.remove(" ".join(e.src_sent))
                for e in test_set:
                    target_list.remove(" ".join(e.src_sent))
                if not len(target_list) == 0:
                    print ("key")
    """