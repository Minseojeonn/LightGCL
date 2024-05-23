import pickle 
import os
import numpy as np


def read_txt_fine(file_path):
    test = file_path + "_test.txt"
    train = file_path + "_training.txt"
    val = file_path + "_val.txt"
    fr_list = []
    to_list = []
    value = []
    store = [test,train,val]
    for filename in store:
        f = open(filename)
        lines = f.readlines()
        for edge in lines:
            edge = edge.replace("\n"," ")
            fr, to, val = edge.split("\t")
            fr_list.append(int(fr))
            to_list.append(int(to))
            value.append(int(val))
    num_node_u = max(fr_list)
    num_node_v = max(to_list)
    return num_node_u, num_node_v
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
    num_node_u, num_node_v = read_txt_fine(f"./copyed/{dataset_name}-{dataset_number}")
    print(num_node_u, num_node_v)
    
###
exit()
###


