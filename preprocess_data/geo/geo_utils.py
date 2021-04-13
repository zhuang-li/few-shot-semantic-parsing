import json

from preprocess_data.geo.process_geoquery import q_process


def read_geo_prolog_data_lines(template_filepath, train = ['train', 'dev'], is_query_split = True):
    tgt_code_list = []
    tgt_code_list_splited = []
    src_list = []
    with open(template_filepath) as json_file:
        json_data = json.load(json_file)
        for program in json_data:
            if not is_query_split:
                for q_s in program['sentences']:
                    if q_s['question-split'] in train:
                        template = program['logic'][0]
                        tgt_code_list.append(template)
                        tgt_code_list_splited.append(template.split(' '))
                        for var,name in q_s['variables'].items():
                            q_s['text'] = q_s['text'].replace(var,name)
                        src_list.append(q_s['text'].split(' '))
            else:
                if program['query-split'] in train:
                    for q_s in program['sentences']:
                        template = program['logic'][0]
                        tgt_code_list.append(template)
                        tgt_code_list_splited.append(template.split(' '))
                        for var,name in q_s['variables'].items():
                            q_s['text'] = q_s['text'].replace(var,name)
                        src_list.append(q_s['text'].split(' '))
    return src_list, tgt_code_list_splited