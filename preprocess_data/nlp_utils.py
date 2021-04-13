import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from preprocess_data.geo.process_geoquery import norm_word

stemmer = SnowballStemmer("english")
lemmer = WordNetLemmatizer()

def token_stem_lemma_mapping(token_list, map_dict):
    for token in token_list:
        stem_token = norm_word(token)
        lemma_token = lemmer.lemmatize(token)
        if stem_token in map_dict:
            map_dict[stem_token]['lemma'].add(lemma_token)
            map_dict[stem_token]['ori'].add(token)
        else:
            map_dict[stem_token] = {'lemma': set(), 'ori': set()}
            map_dict[stem_token]['lemma'].add(lemma_token)
            map_dict[stem_token]['ori'].add(token)

def turn_stem_into_lemma(token_list, map_dict):
    lemma_token_list = []
    for token in token_list:
        if token in map_dict:
            #print (map_dict[token])
            #print (list(map_dict[token]['lemma'])[0])
            lemma_list = list(map_dict[token]['lemma'])
            edit_dis = [nltk.edit_distance(token, lemma_token) for lemma_token in lemma_list]
            index = edit_dis.index(min(edit_dis))
            lemma_token_list.append(lemma_list[index])
        else:
            lemma_token_list.append(token)
    return lemma_token_list