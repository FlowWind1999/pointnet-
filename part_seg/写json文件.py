import json
import os
import random

def write_train():
    with open("../data/ours/train_test_split/shuffled_train_file_list.json", "w+") as f:
        path = "../data/ours/10"
        lst = os.listdir(path)
        lit = []
        for i in lst:
            i = "shape_data/10/" + i[:-4]
            lit.append(i)
        random.shuffle(lit)
        print(lit)
        json.dump(lit, f)

def write_test():
    with open("../data/UVA/train_test_split/shuffled_test_file_list.json", "w+") as f:
        path = "../data/UVA/gtdata"
        lst = os.listdir(path)
        random.shuffle(lst)
        lit = []
        for i in lst:
            i = "UVA/gtdata/" + i[:-4]
            lit.append(i)
        print(len(lit))
        json.dump(lit, f)

def write_val():
    with open("../data/UVA/train_test_split/shuffled_val_file_list.json", "w+") as f:
        path = "../data/UVA/gadata"
        lst = os.listdir(path)
        random.shuffle(lst)
        lit = []
        for i in lst:
            i = "shape_data/text/" + i[:-4]
            lit.append(i)
        print(len(lit))
        json.dump(lit, f)

def write_list():
    path_wire = "../data/UVA/gtdata"
    lst = os.listdir(path_wire)
    random.shuffle(lst)
    lit1=[]
    lit2=[]
    lit3=[]
    count=0
    f1=open("../data/UVA/train_test_split/shuffled_train_file_list.json", "w+")
    f2=open("../data/UVA/train_test_split/shuffled_test_file_list.json", "w+")
    f3=open("../data/UVA/train_test_split/shuffled_val_file_list.json", "w+")
    for item in lst:
        i="shape_data/UVA/" + item[:-4]
        print(i)
        if(count<0):
            lit1.append(i)
        elif(count<=300):
            lit2.append(i)
        else:
            lit3.append(i)
        count=count+1
    print(len(lit1),len(lit2),len(lit3))
    json.dump(lit1, f1)
    json.dump(lit2, f2)
    json.dump(lit3, f3)
    f1.close()
    f2.close()
    f3.close()


if __name__ == "__main__":
    write_list()


