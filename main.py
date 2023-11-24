import json
import os
import joblib
from sklearn.neighbors import NearestNeighbors
import sys
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np

from flask import Flask, jsonify, request

model_file_path = 'C:/Users/nhona/OneDrive/Máy tính/Nam/AI/trained_model.joblib'
excel_file_path = 'C:/Users/nhona/OneDrive/Máy tính/Nam/AI/Data.xlsx'

def chunk_feature(str, arrPolarityTerm):
    # Maximum_Matching
    vTerm = []
    strRemain = ""
    start = 0
    isTerm = False
    isStop = False

    str = str.lower()
    str = str.lstrip(" ").rstrip(" ")
    WordList = str.split(" ")
    stop = len(WordList)

    while (isStop == False and stop >= 0):
        for num in range(start, stop):
            strRemain = strRemain + WordList[num] + " "

        strRemain = strRemain.lstrip(" ").rstrip(" ").lower()
        isTerm = False
        for cha in range(0, len(arrPolarityTerm)):
            arr = arrPolarityTerm[cha]
            if (arr == strRemain):
                vTerm.append(strRemain)
                isTerm = True
                if (start == 0):
                    isStop = True
                else:
                    stop = start
                    start = 0

        if (isTerm == False):
            if (start == stop):
                stop = stop - 1
                start = 0
            else:
                start += 1

        strRemain = ""
    strRemain = ""
    for stt in range(0, len(vTerm)):
        strRemain = strRemain + " " + vTerm[stt]

    return vTerm

arrTerm = ['hôm nay', 'hôm', 'nay', 'thi tốt', 'thi tốt nghiệp', 'tốt nghiệp']
strtemp = 'hôm nay thi tốt nghiệp'
arr = chunk_feature(strtemp, arrTerm)

def read_file_data(excel_file_path):
    df = pd.read_excel(excel_file_path)

    keyword_dictionary = {}
    all_keywords = []

    for index, row in df.iterrows():
        product_type = row['Key']
        attributes = row['Attributes'].split(', ')
        
        for attr in attributes:
            if attr not in all_keywords:
                all_keywords.append(attr)
        
        keyword_dictionary[product_type] = attributes

    with open('C:/Users/nhona/OneDrive/Máy tính/Nam/AI/all_keywords.json', 'w', encoding='utf-8') as file:
        json.dump(all_keywords, file, ensure_ascii=False)

    with open('C:/Users/nhona/OneDrive/Máy tính/Nam/AI/labels.json', 'w', encoding='utf-8') as file:
        json.dump(list(keyword_dictionary.keys()), file, ensure_ascii=False)

    return keyword_dictionary, all_keywords



def train_knn_model(vectors):
    knn_model = NearestNeighbors(n_neighbors=3, metric='euclidean')
    knn_model.fit(vectors)

    # Save the trained NearestNeighbors model
    joblib.dump(knn_model, model_file_path)

    return knn_model


def find_similar_product(keySearch, knn_model, all_keywords, labels):
    keySearch = word_tokenize(keySearch)
    user_vector = np.zeros(len(all_keywords))

    for i, keyword in enumerate(all_keywords):
        keyword_tokens = word_tokenize(keyword)
        user_vector[i] = sum(1 for token in keySearch if token in keyword_tokens)

    # Kiểm tra và đảm bảo rằng số lượng đặc trưng của user_vector phù hợp với mô hình
    if len(user_vector) != knn_model.n_features_in_:
        print("Số lượng đặc trưng của user_vector không phù hợp với mô hình.")
        return None

    _, indices = knn_model.kneighbors([user_vector])
    similar_plants = [labels[i] for i in indices[0]]
    return similar_plants

# Trong hàm train_and_save_model:
def train_and_save_model():
    keyword_dictionary, all_keywords = read_file_data(excel_file_path)

    vectors = []
    labels = []

    for label, keywords in keyword_dictionary.items():
        vector = np.zeros(len(all_keywords))
        for i, keyword in enumerate(all_keywords):
            vector[i] = 1 if keyword in keywords else 0
        vectors.append(vector)
        labels.append(label)

    vectors = np.array(vectors)
    labels = np.array(labels)

    # Train the NearestNeighbors model
    knn_model = train_knn_model(vectors)

    # Save the trained NearestNeighbors model
    joblib.dump(knn_model, model_file_path)



def load_all_keywords_labels():
    try:
        file_path_all_keywords = 'C:/Users/nhona/OneDrive/Máy tính/Nam/AI/all_keywords.json'
        file_path_labels = 'C:/Users/nhona/OneDrive/Máy tính/Nam/AI/labels.json'

        with open(file_path_all_keywords, 'r', encoding='utf-8') as json_file:
            all_keywords = json.load(json_file)

        with open(file_path_labels, 'r', encoding='utf-8') as json_file:
            labels = json.load(json_file)
        
        # Load the trained NearestNeighbors model
        knn_model = joblib.load(model_file_path)

        return all_keywords, labels, knn_model

    except FileNotFoundError:
        print("FileNotFoundError")
        return None, None, None

app = Flask(__name__)
@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message='Hello, World!')

@app.route('/api/chat', methods=['POST'])
def add_numbers():
        data = request.get_json()
        question = data['question']
        return jsonify(result=question)
if __name__ == "__main__":
    train_and_save_model()
    app.run()

    

    if os.path.exists(model_file_path):
        all_keywords, labels, knn_model = load_all_keywords_labels()
        keySearch = ""
        
        similar_plants = find_similar_product(
            keySearch, knn_model, all_keywords, labels)
        
        for plant in similar_plants:
            print(plant)
            # sys.stderr.write(plant + ',')
