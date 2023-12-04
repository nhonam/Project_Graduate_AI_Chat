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

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
import joblib

def generate_vectors(all_keywords, keywords_combinations):
    vectors = []
    for keywords in keywords_combinations:
        vector = np.zeros(len(all_keywords))
        for i, keyword in enumerate(all_keywords):
            vector[i] = 1 if keyword in keywords else 0
        vectors.append(vector)
    return vectors
def train_and_save_model():
    keyword_dictionary, all_keywords, df = read_file_data()
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
                    # print("int_list",i, int_list)
                    # list_result[i] ={[row['Key']]: int_list}

                    while len(list_result) <= abc:
                        list_result.append({})

                    list_result[abc][row['Key']] = int_list
                    vector_list.append(int_list)
                    abc +=1

    print(list_result)
    flat_data = [{'Label': key, 'Vector': value} for d in list_result for key, value in d.items()]

# Create a DataFrame
    result_df = pd.DataFrame(flat_data)

# Save the DataFrame to an Excel file
    result_df.to_excel(file_path_vectors_keywords, index=False)
    # knn_model = train_knn_model(vector_list)
    #         # Save the trained NearestNeighbors model
    # joblib.dump(knn_model, model_file_path)
    
    return list_result
def train_and_save_knn_model(data, model_file_path):
    knn_model = NearestNeighbors(n_neighbors=2)
    knn_model.fit(data)
    joblib.dump(knn_model, model_file_path)

def main():
    df = pd.DataFrame({'Key': [1, 2, 3], 'Data': ['example1', 'example2', 'example3']})
    all_keywords = ['keyword1', 'keyword2', 'keyword3']  # Thay thế bằng danh sách tất cả các keyword của bạn

    for index, row in df.iterrows():
        vectors = []
        for r in range(1, len(all_keywords) + 1):
            arr = list(combinations(all_keywords, r))
            vectors.extend(generate_vectors(all_keywords, arr))

        list_of_lists = [[int(element) for element in arr] for arr in vectors]

        print(row['Key'])
        row_key = row['Key'] - 1  # Chỉ số trong list_of_lists
        df.at[index, 'Vector'] = [list_of_lists[row_key]]

    # Combine vectors from all rows
    all_vectors = [item for sublist in df['Vector'] for item in sublist]

    # Train and save KNN model
    train_and_save_knn_model(all_vectors, 'your_model_path.pkl')

if __name__ == "__main__":
    main()