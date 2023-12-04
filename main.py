import json
import os
import joblib
from sklearn.neighbors import NearestNeighbors
import sys
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from itertools import product
from flask import Flask, jsonify, request
from itertools import combinations
model_file_path = 'D:\\DoAnTotNghiep\\AI_Main\\Project_Graduate_AI_Chat\\trained_model.joblib'
excel_file_path = 'D:\\DoAnTotNghiep\\AI_Main\\Project_Graduate_AI_Chat\\Data.xlsx'
file_path_vectors_keywords = 'D:\\DoAnTotNghiep\\AI_Main\\Project_Graduate_AI_Chat\\vector_train_export.xlsx'
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
            if (arr.lower() == strRemain):
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



def read_file_data():
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

    with open('D:\\DoAnTotNghiep\\AI_Main\\Project_Graduate_AI_Chat\\all_keywords.json', 'w', encoding='utf-8') as file:
        json.dump(all_keywords, file, ensure_ascii=False)

    with open('D:\\DoAnTotNghiep\\AI_Main\\Project_Graduate_AI_Chat\\labels.json', 'w', encoding='utf-8') as file:
        json.dump(list(keyword_dictionary.keys()), file, ensure_ascii=False)

    return keyword_dictionary, all_keywords, df



def train_knn_model(vectors):
    knn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn_model.fit(vectors)

    # Save the trained NearestNeighbors model
    joblib.dump(knn_model, model_file_path)

    return knn_model


def find_similar_product(keySearch, knn_model, all_keywords, labels, list_result):
    user_vector = np.zeros(len(all_keywords))
    user_vector = user_vector.astype(int).tolist()

    for i, keyword in enumerate(all_keywords):
        keyword_tokens = word_tokenize(keyword)
        # user_vector[i] = 1 for token in keySearch if token in keyword_tokens
        # user_vector[i] = 1 if keyword in keyword_tokens else 0
        for i, token in enumerate(keySearch):
            if token in keyword_tokens:
                user_vector[i] = 1

    # Kiểm tra và đảm bảo rằng số lượng đặc trưng của user_vector phù hợp với mô hình
    print("user_vector", user_vector)
    if len(user_vector) != knn_model.n_features_in_:
        print("Số lượng đặc trưng của user_vector không phù hợp với mô hình.")
        return None

    _, indices = knn_model.kneighbors([user_vector])
    # for i in indices[0]:
    #     print(labels[i])
    # print(labels)

 


    similar_products = [list_result[i-1] for i in indices[0]]
    print("similar_products",similar_products)
    # print(indices)
    return similar_products

# Trong hàm train_and_save_model:
def train_and_save_model():
   
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

    with open('D:\\DoAnTotNghiep\\AI_Main\\Project_Graduate_AI_Chat\\all_keywords.json', 'w', encoding='utf-8') as file:
        json.dump(all_keywords, file, ensure_ascii=False)

    with open('D:\\DoAnTotNghiep\\AI_Main\\Project_Graduate_AI_Chat\\labels.json', 'w', encoding='utf-8') as file:
        json.dump(list(keyword_dictionary.keys()), file, ensure_ascii=False)
  

  

# Create a DataFrame
  

# Save the DataFrame to an Excel file
    # knn_model = train_knn_model(vector_list)
    #         # Save the trained NearestNeighbors model
    # joblib.dump(knn_model, model_file_path)
    
    return "list_result"


def loadData():
    keyword_dictionary, all_keywords, df = read_file_data(excel_file_path)
    labels = list( keyword_dictionary.keys())
    vectors = []
    list_of_lists=[]
    list_result = []
    abc = 0
    vector_list =[]
  

    for index, row in df.iterrows():
        for r in range(1, len(all_keywords)+1):
            arr = list(combinations(all_keywords, r))
            for  ih,keywords in enumerate(arr):
                vector = np.zeros(len(all_keywords))
                for i, keyword in enumerate(all_keywords):
                    vector[i] = 1 if keyword in keywords else 0
                    
                    float_list = np.array(vector).tolist()
                    # print(row['Key'])
                    int_list = [int(element) for element in float_list]
                    print("int_list",i, int_list)
                    # list_result[i] ={[row['Key']]: int_list}

                    while len(list_result) <= abc:
                        list_result.append({})

                    list_result[abc][row['Key']] = int_list
                    vector_list.append(int_list)
                    abc +=1

    # knn_model = train_knn_model(vector_list)
    #         # Save the trained NearestNeighbors model
    # joblib.dump(knn_model, model_file_path)
    print(list_result)
    return list_result

def load_all_keywords_labels_1():
    try:
        file_path_all_keywords = 'D:\\DoAnTotNghiep\\AI_Main\\Project_Graduate_AI_Chat\\all_keywords.json'
        file_path_labels = 'D:\\DoAnTotNghiep\\AI_Main\\Project_Graduate_AI_Chat\\labels_1.json'

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


def load_all_keywords_labels():
    try:
        file_path_all_keywords = 'D:\\DoAnTotNghiep\\AI_Main\\Project_Graduate_AI_Chat\\all_keywords.json'
        file_path_labels = 'D:\\DoAnTotNghiep\\AI_Main\\Project_Graduate_AI_Chat\\labels.json'

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
        all_keywords, labels, knn_model = load_all_keywords_labels_1()
        print(labels)
        
        return jsonify(result=chunk_feature(question, labels))
if __name__ == "__main__":
    df = train_and_save_model()
    # app.run()

    # df = loadData()

    if os.path.exists(model_file_path):
        all_keywords, labels, knn_model = load_all_keywords_labels()
        # keySearch = chunk_feature("tôi muốn mua thảm",labels)
        keySearch ="vợt cầu lông"
        # chunk_feature
        
        similar_product = find_similar_product(
            keySearch, knn_model, all_keywords, labels,df)
        
        print("similar_product",similar_product)
        
    
