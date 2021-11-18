import json
import re
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from db_control import download_characteristics, download_stopwords

REMOVE_ITEMS = ["菜系", "餐點", "價位", "特色"]
REMOVE_STRING = ["用餐", "料理", "菜系", "飲食", "風味", "提供", "有", "供應", "可供", "選擇", "特別", "特殊", "服務", "設施", "廳", "館", "接受",
                                    "使用Discover", "使用Mastercard", "使用Visa", "使用美國運通", "通行", "付款"]
MAX_DF = 1.0
MAX_VOCAB = 512

def load_stopwords(stopwords_content):
    stopwords = []
    f= stopwords_content.split("\n")
    for line in f:
        if len(line)>0:
            stopwords.append(line.strip())
    return stopwords
    
def has_remove_item(characteristic):
    return characteristic in REMOVE_ITEMS

def has_numbers(characteristic):
    return any(map(str.isdigit, characteristic))

def clean_parsed_characteristics(characteristics):
    cleaned_characteristics = []
    for characteristic in characteristics:
        # remove extra space/tab
        characteristic = re.sub("[\s\t\n]", "", characteristic)
        for string in REMOVE_STRING:
            characteristic = re.sub(string, "", characteristic)
        cleaned_characteristics.append(characteristic)
    return cleaned_characteristics

def parse_string_list(string):
    characteristics = re.split(",|/", string)
    characteristics = clean_parsed_characteristics(characteristics)
    return characteristics

def clean_characteristics(characteristics):
        
    cleaned_characteristics = []
    
    characteristics = characteristics.split(" ")
    
    index = 0
    while index < len(characteristics):
        characteristic = characteristics[index]
        # skip remove words
        if has_remove_item(characteristic):
            index += 1
            continue
        elif has_numbers(characteristic):
            index += 1
            continue
        
        # append cleaned characteristic:
        cleaned_characteristics += parse_string_list(characteristic)
        index += 1
    
    return cleaned_characteristics

def get_chinese(name):
    result = re.findall(u"[\u4e00-\u9fa5]", name)
    return "".join(result)

def do_clustering(characteristics_path, stopwords_path, output_path, min_samples=5, download_enable=True):

    if download_enable:
        download_stopwords(stopwords_path)
        download_characteristics(characteristics_path)
    
    with open(characteristics_path, "r") as read_file:
        characteristics = json.load(read_file)
    
    with open(stopwords_path, "r") as read_file:
        stopwords_content = json.load(read_file)
    stopwords = load_stopwords(stopwords_content)
    
    # Exam all characteristics
    tag_set = set()
    tags = {}
    
    for restaurant_id, characteristic in characteristics.items():
        tag = clean_characteristics(characteristic)
        tag_set |= set(tag)
        
        tags[restaurant_id] = tag
        
    print("n_tags: {}".format(len(tag_set)))
    corpus = []
    ids = []
    for restaurant_id, characteristic in characteristics.items():
        tag = tags[restaurant_id]
        tag_string = "/".join(tag)
        
        cut = jieba.cut(tag_string)        
        words = [word for word in cut if word not in stopwords]
        corpus.append("/".join(words))
        ids.append(restaurant_id)
        
    print("Train vectorizer")
    vectorizer = TfidfVectorizer(max_df=MAX_DF, max_features=MAX_VOCAB)
    X = vectorizer.fit_transform(corpus)
    
    print("Feature names: ")
    print(vectorizer.get_feature_names())
    print("Feature shape: ")
    print(X.shape)

    cluster = DBSCAN(min_samples=min_samples)
    labels = cluster.fit_predict(X)
    
    print("n_clusters: {}".format(len(np.unique(labels))))
    print("n_noisy_smaples: {}".format(np.sum(labels == -1)))
    
    clusters = []
    
    for label in np.unique(labels):
        
        if label != -1:
            sel = labels == label
        
            print("*** LABEL {}: {} elements ***".format(label, np.sum(sel)))
            
            ids_ = np.array(ids)[sel]
            
            in_cluster = ids_.tolist()
            clusters.append(in_cluster)

    with open(output_path, "w") as write_file:
        json.dump(clusters, write_file)
    print("n_classifiers: {}".format(len(clusters)))

if __name__ == "__main__":
    download_enable = True
    characteristics_path = "./characteristics.json"
    stopwords_path = "./stopwords.json"
    output_path = "./clusters.json"
    
    do_clustering(characteristics_path, stopwords_path, output_path, download_enable=download_enable)
    