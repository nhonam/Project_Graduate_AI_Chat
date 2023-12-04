import json
import os
import joblib
from sklearn.neighbors import NearestNeighbors
import sys
import nltk
import re
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from itertools import product
from flask import Flask, jsonify, request
from itertools import combinations
#máy vnpt
model_file_path = 'C:\\Users\\nhona\\OneDrive\\Máy tính\\Nam\\Project_Graduate_AI_Chat\\trained_model.joblib'
excel_file_path = 'C:\\Users\\nhona\\OneDrive\\Máy tính\\Nam\\Project_Graduate_AI_Chat\\Data.xlsx'
file_path_vectors_vector_train_export = 'C:\\Users\\nhona\\OneDrive\\Máy tính\\Nam\\Project_Graduate_AI_Chat\\vector_train_export.xlsx'
all_keywords_json_path ='C:\\Users\\nhona\\OneDrive\\Máy tính\\Nam\\Project_Graduate_AI_Chat\\all_keywords.json'
labels_json_path ='C:\\Users\\nhona\\OneDrive\\Máy tính\\Nam\\Project_Graduate_AI_Chat\\labels.json'
#---------------
#máy cá nhân
# model_file_path = 'D:\\DoAnTotNghiep\\AI_Main\\Project_Graduate_AI_Chat\\trained_model.joblib'
# excel_file_path = 'D:\\DoAnTotNghiep\\AI_Main\\Project_Graduate_AI_Chat\\Data.xlsx'
# file_path_vectors_keywords = 'D:\\DoAnTotNghiep\\AI_Main\\Project_Graduate_AI_Chat\\vector_train_export.xlsx'



def train_knn_model(vectors):
    knn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn_model.fit(vectors)
    return knn_model

def train_and_save_model():
    # Đọc dữ liệu từ tệp Excel
    vectors = pd.read_excel(file_path_vectors_vector_train_export)
    # Lấy giá trị từ DataFrame (loại bỏ cột 'Label')
    values_without_label = vectors.drop('Label', axis=1).values
    knn_model = train_knn_model(values_without_label)
    # Lưu trữ mô hình vào một tệp
    joblib.dump(
        knn_model, model_file_path)

    return vectors

# Trong hàm train_and_save_model:
def load_and_process_data():
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

    with open(all_keywords_json_path, 'w', encoding='utf-8') as file:
        json.dump(all_keywords, file, ensure_ascii=False)

    with open(labels_json_path, 'w', encoding='utf-8') as file:
        json.dump(list(keyword_dictionary.keys()), file, ensure_ascii=False)

    #tạo vector cho mỗi sản phẩm và nhãn
    all_vectors = []
    for i, row in df.iterrows():
        attributes = row['Attributes'].split(', ')

        #tạo các vector
        vectors = []

        for j in range(1, len(all_keywords) + 1):

            # tố hợp từ khóa
            combinations_attr = list(combinations(attributes, j))
            #tạo vector
            for combination in combinations_attr:
                vector = np.zeros(len(all_keywords))
                for attr in combination:
                    vector[all_keywords.index(attr)] = 1
                    print(vector)
                vectors.append(vector)

          # Gắn nhãn cho vector
        label = f"{row['Key']}"
        all_vectors.extend([(label, *vector) for vector in vectors])

         # Lưu kết quả vào Excel
    result_df = pd.DataFrame(all_vectors, columns=['Label', *all_keywords])
    result_df.to_excel(file_path_vectors_vector_train_export, index=False)

    return keyword_dictionary, all_keywords


  

  

# Create a DataFrame
  

# Save the DataFrame to an Excel file
    # knn_model = train_knn_model(vector_list)
    #         # Save the trained NearestNeighbors model
    # joblib.dump(knn_model, model_file_path)
    
    return "list_result"


def find_similar_keySearch(keySearch, knn_model, all_keywords, labels, df):
    user_vector = np.zeros(len(all_keywords))

    for i, keyword in enumerate(all_keywords):
        for index, token in enumerate(keySearch):
            if token in keyword:
                user_vector[i] = 1
    _, indices = knn_model.kneighbors([user_vector])
    similar_plants = [df.iloc[i - 1] for i in indices[0]]
    # Trích xuất giá trị từ cột "Label"
    label_values = [row["Label"] for row in similar_plants]
    return label_values

def load_file_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            result_json = json.load(json_file)
        return result_json
    except FileNotFoundError:
        print("Không tìm thấy file " + file_path)
        return None, None
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

app = Flask(__name__)
@app.route('/api/chat', methods=['POST'])
def add_numbers():
        data = request.get_json()
        question = data['question']
        if os.path.exists(model_file_path):
            try:
                labels = load_file_json(labels_json_path)
                all_keywords = load_file_json(all_keywords_json_path)
    
                list_keywords = chunk_feature(
                    re.sub(r'[^\w\s]', '',question), all_keywords)
                if list_keywords==[]:
                    return jsonify(result="Bạn vui lòng mô tả chi tiết hơn ?")
                df = pd.read_excel(file_path_vectors_vector_train_export)

              
    
                # Load mô hình từ tệp
                knn_model = joblib.load(model_file_path)
                similar = find_similar_keySearch(
                    list_keywords, knn_model, all_keywords, labels, df)
                print("================")
                return jsonify(result=similar[0])


            except FileNotFoundError:
                return jsonify(result="Bạn vui lòng mô tả rõ hơn ?")


        print(labels)
        
        
if __name__ == "__main__":

    #tranning
    # load_and_process_data()
    # df = train_and_save_model()
    #-----------

    app.run()

    
