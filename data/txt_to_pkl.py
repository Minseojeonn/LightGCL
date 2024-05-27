import pickle 
import os
import numpy as np
from scipy.sparse import coo_matrix


def read_txt_fine(file_path):
    test = file_path + "_test.txt"
    train = file_path + "_training.txt"
    val = file_path + "_val.txt"
    fr_list = []
    to_list = []
    store_dict = {'test':[[],[],[]], 'training':[[],[],[]],  'val':[[],[],[]]}
    value = []
    store = [test,train,val]
    for filename in store:
        converted_name = filename.split(".")[1].split("_")[-1]
        f = open(filename)
        lines = f.readlines()
        for edge in lines:
            edge = edge.replace("\n"," ")
            fr, to, sign = edge.split("\t")
            fr_list.append(int(fr))
            to_list.append(int(to))
            value.append(int(sign))
            store_dict[converted_name][0].append(int(fr))
            store_dict[converted_name][1].append(int(to))
            store_dict[converted_name][2].append(int(sign))
    num_node_u = max(fr_list)
    num_node_v = max(to_list)
    len_of_edge = len(fr_list)
    del fr_list
    del to_list
    return num_node_u, num_node_v, store_dict, len_of_edge

def list_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        return files
    except FileNotFoundError:
        return f"Error: The directory {directory_path} does not exist."
    except PermissionError:
        return f"Error: Permission denied for accessing the directory {directory_path}."

# 예제 사용법
directory_path = '/home/minseo/LightGCL/data/copyed'
file_names = list_files_in_directory(directory_path)

for i in file_names:
    name, role = i.split("_")
    dataset_name = name[:-2]
    dataset_number = name[-1:]
    role = role[:-4]
    if os.path.isdir(f"./{dataset_name}") == False:
        os.makedirs(f"./{dataset_name}")
    num_node_u, num_node_v, store_dict, len_of_edge = read_txt_fine(f"./copyed/{dataset_name}-{dataset_number}") # sotre  dict = train,val,test로 이루어진 dict로, 각 요소별로 fr, to, sign이 list로 들어있음.
    train_matrix = np.zeros((num_node_u+1, num_node_v+1))
    val_matrix = np.zeros((num_node_u+1, num_node_v+1))
    test_matrix = np.zeros((num_node_u+1, num_node_v+1))
    original_edge_num = 0
    filtered_edge_num = 0
    for key in store_dict:
        fr, to, sign = store_dict[key]
        for f, t, s in zip(fr,to,sign) :
            #print(f,t,s)
            if key == "training":
                if s == 1:
                    train_matrix[f][t] = 1
                    filtered_edge_num += 1
            elif key == "val":
                val_matrix[f][t] = s
                filtered_edge_num += 1
            else:
                test_matrix[f][t] = s
                filtered_edge_num += 1
            original_edge_num += 1
    if os.path.isdir(f"./{dataset_name}-{dataset_number}" == False):
        os.makedirs(f"./{dataset_name}-{dataset_number}")
    with open(f"./{dataset_name}-{dataset_number}/trnMat.pkl", 'wb') as f:
        pickle.dump(coo_matrix(train_matrix), f, pickle.HIGHEST_PROTOCOL)
    with open(f"./{dataset_name}-{dataset_number}/tstMat.pkl", 'wb') as f:
        pickle.dump(coo_matrix(test_matrix), f, pickle.HIGHEST_PROTOCOL)
    with open(f"./{dataset_name}-{dataset_number}/valMat.pkl", 'wb') as f:
        pickle.dump(coo_matrix(val_matrix), f, pickle.HIGHEST_PROTOCOL)  
    
    
###
exit()
###


