# -*- coding: utf-8 -*-
import json
import os
import joblib
from sklearn.neighbors import NearestNeighbors
import sys
import re
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from itertools import product
from itertools import combinations
from collections import defaultdict  # Import defaultdict from collections module

# Tải tài liệu và từ điển tiếng Việt
# nltk.download('punkt')

model_file_path = 'D:/TNDH/souce-code/server/src/assets/trained_model.joblib'
excel_file_path = 'D:/TNDH/souce-code/server/src/assets/Data3456.xlsx'
file_path_all_keywords = 'D:/TNDH/souce-code/server/src/assets/all_keywords.json'
file_path_labels = 'D:/TNDH/souce-code/server/src/assets/labels.json'
file_path_keyword_dict = 'D:/TNDH/souce-code/server/src/assets/keyword_dict.json'
file_path_vectors_keywords = 'D:/TNDH/souce-code/server/src/assets/output_vectors_keywords.xlsx'
# Nhận dữ liệu từ Node.js
# keySearch = sys.argv[1]

# Đọc dữ liêu từ file excel


def prepare_training_data():
    df = pd.read_excel(excel_file_path)
    keyword_dictionary = {}
    all_keywords = []

    # Tạo all_keywords từ dữ liệu
    for index, row in df.iterrows():
        plant_type = row['Key']
        good_attributes = row['LabelGood'].split(', ')
        bad_attributes = row['LabelBad'].split(', ')
        for attr in good_attributes + bad_attributes:
            if attr not in all_keywords:
                all_keywords.append(attr)
                keyword_dictionary[plant_type] = good_attributes + \
                    bad_attributes

    # Lưu dữ liệu dưới dạng JSON
    with open(file_path_all_keywords, 'w', encoding='utf-8') as file:
        json.dump(all_keywords, file, ensure_ascii=False)

    with open(file_path_keyword_dict, 'w', encoding='utf-8') as file:
        json.dump(keyword_dictionary, file, ensure_ascii=False)

    # Tạo vector cho mỗi loại cây và nhãn tương ứng
    all_vectors = []

    # Tạo vector cho mỗi loại cây và nhãn tương ứng
    for index, row in df.iterrows():
        good_attributes = row['LabelGood'].split(', ')
        bad_attributes = row['LabelBad'].split(', ')
        # Tạo các vector
        good_vectors = []
        bad_vectors = []
        for r in range(1, len(all_keywords) + 1):
            # Tổ hợp các từ khóa
            good_combinations = list(combinations(good_attributes, r))
            bad_combinations = list(combinations(bad_attributes, r))

           # Tạo vector
            for combination in good_combinations:
                good_vector = np.zeros(len(all_keywords))
                for attr in combination:
                    good_vector[all_keywords.index(attr)] = 1
                good_vectors.append(good_vector)

            for combination in bad_combinations:
                bad_vector = np.zeros(len(all_keywords))
                for attr in combination:
                    bad_vector[all_keywords.index(attr)] = 1
                bad_vectors.append(bad_vector)

        # Gắn nhãn cho vector
        good_label = f"{row['Key']}_NhanTot"
        bad_label = f"{row['Key']}_NhanKem"

        all_vectors.extend([(good_label, *vector) for vector in good_vectors])
        all_vectors.extend([(bad_label, *vector) for vector in bad_vectors])

    # Lưu kết quả vào Excel
    result_df = pd.DataFrame(all_vectors, columns=['Label', *all_keywords])
    result_df.to_excel(file_path_vectors_keywords, index=False)

    return keyword_dictionary, all_keywords


# Thêm hàm để huấn luyện và lưu trữ mô hình
def train_knn_model(vectors):
    knn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
    knn_model.fit(vectors)
    return knn_model


def train_and_save_model():
    # Đọc dữ liệu từ tệp Excel
    vectors = pd.read_excel(file_path_vectors_keywords)
    # Lấy giá trị từ DataFrame (loại bỏ cột 'Label')
    values_without_label = vectors.drop('Label', axis=1).values
    knn_model = train_knn_model(values_without_label)
    # Lưu trữ mô hình vào một tệp
    joblib.dump(
        knn_model, 'D:/TNDH/souce-code/server/src/assets/trained_model.joblib')

    return vectors


def load_file_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            result_json = json.load(json_file)
        return result_json
    except FileNotFoundError:
        print("Không tìm thấy file " + file_path)
        return None, None


def chunk_feature(input_str, arrPolarityTerm):
    vTerm = []
    strRemain = ""
    start = 0
    isTerm = False
    isStop = False
    input_str = input_str.lower()
    input_str = input_str.lstrip(" ").rstrip(" ")
    WordList = input_str.split(" ")
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
    return vTerm


def find_similar_plant(keySearch, knn_model, all_keywords, labels, df):
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


# Tìm từ khóa người dùng cung cấp so với tập từ điển
if __name__ == "__main__":
    # Kiểm tra sự tồn tại của file mô hình và xem liệu có sự kiện thay đổi trong file Excel hay không
    # if not os.path.exists(model_file_path) or (len(sys.argv) > 1 and sys.argv[1] == 'update'):
    if not os.path.exists(model_file_path):
        try:
            keyword_dictionary, all_keywords = prepare_training_data()
            vectors = train_and_save_model()
        except FileNotFoundError:
            sys.stderr.write(json.dumps({
                "type": "string",
                "data": "Lỗi khi xây dựng tập dữ liệu"}))
    else:
        if os.path.exists(model_file_path):
            try:
                keyword_dictionary = load_file_json(file_path_keyword_dict)
                all_keywords = load_file_json(file_path_all_keywords)
                labels = list(keyword_dictionary.keys())
                # Lá cây sen đá của tôi bị đóm nâu có làm sao không?
                list_keywords = chunk_feature(
                    re.sub(r'[^\w\s]', '', "Sen đá của tôi bị úng"), all_keywords)
                df = pd.read_excel(file_path_vectors_keywords)
                # Lọc các dòng có Label là "Sen đá"
                # rows = df[df['Label'].str.lower() == "Sen đá".lower()]
                rows = df.loc[df['Label'].str.lower() == "sen đá".lower()]

                print("rows", rows)

                # Load mô hình từ tệp
                knn_model = joblib.load(model_file_path)
                similar_plants = find_similar_plant(
                    list_keywords, knn_model, all_keywords, labels, df)
                sys.stderr.write(json.dumps({
                    "success": True,
                    "type": "object",
                    "data": similar_plants
                }))
            except FileNotFoundError:
                sys.stderr.write(json.dumps({
                    "success": False,
                    "type": "string",
                    "data": "Lỗi khi tìm kiếm từ khóa"}))
