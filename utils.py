
import os
import pandas as pd
import random
from collections import defaultdict


def load_data(path, data=0):
    '''
    Args:
        path(string): the directory of the data
        student(int): 0: use original test set; 1: use the training student for student model; .
    Returns:
        tableA, tableB, test: the dataframe for the two tables of records and the test set
    '''
    pathA = os.path.join(path,"tableA.csv")
    pathB = os.path.join(path,"tableB.csv")
    if data == 0:
        path_set = os.path.join(path,"test.csv")
    elif data == 1:
        path_set = os.path.join(path,"train_student.csv")
    elif data == 2:
        path_set = os.path.join(path,"test_student.csv")
    else:
        raise Exception('Invalid Dataset!') 
    tbA = pd.read_csv(pathA)
    tbB = pd.read_csv(pathB)
    tests = pd.read_csv(path_set)
    return tbA, tbB, tests


def formulate_instance(tableA, tableB, inst):
    '''
    Args:
        tableA/tableB(dataFrame): the two tables
        inst(dataFrame): one test instance
    Returns:
        item: a triplet with two entities and label
    '''
    id1 = int(inst[0])
    id2 = int(inst[1])
    header = list(tableA)
    attr_num = len(header)
    left = []
    right = []
    for idx in range(attr_num):
        if pd.isnull(tableA.iloc[id1][idx]):
            left.append("")
        else:
            left.append(str(tableA.iloc[id1][idx]))
        if pd.isnull(tableB.iloc[id2][idx]):
            right.append("")
        else:
            right.append(str(tableB.iloc[id2][idx]))
    item = (left, right, inst[2], header)
    return item


def generate_student_training_instances(instance, explains, label):
    '''
    Args:
        instance(tuple): The test instance in the format of left entity, right entity, label, header
        explains(defaultdict): The explaination for the instance: key: attribute, value: tokens
        label(str): The ground truth for student model (not the original label)
    Returns:
        res(string): a sequentialized instance for the student model
    '''

    header = instance[-1]
    if len(explains.keys()) == 0:
        left = ""
        right = ""
        for idx,attr in enumerate(header):
            left += " COL "+str(attr)+" VAL "+str(instance[0][idx])
            right += " COL "+str(attr)+" VAL "+str(instance[1][idx])
        return left.strip()+"\t"+right.strip()+"\t"+label
    
    explain_attrs = set()
    for k,_ in explains.items():
        explain_attrs.add(k)
    tmp = [i for i in range(len(header))]
    left = ""
    right = ""
    for idx,attr in enumerate(header):
        if attr in explain_attrs:
            left += " COL "+str(attr)+" VAL "+str(explains[attr])
            right += " COL "+str(attr)+" VAL "+str(instance[1][idx])
        else:
            left += " COL "+str(attr)+" VAL "+str(instance[0][idx])
            right += " COL "+str(attr)+" VAL "+str(instance[1][idx])
    res = left.strip()+"\t"+right.strip()+"\t"+label
    return res


def split_two_sets(raw_data, rate=0.6):
    '''
    Args:
        raw_data(Array): The test sets, each item is in the Ditto format
        rate: the portion of training data
    Returns:
        train/test(Array): The training and test sets for student model
    ''' 
    pos_instances = []
    neg_instances = []
    for item in raw_data:    
        if item.split(',')[2] == '1':
            pos_instances.append(item)
        else:
            neg_instances.append(item)    
    train = []
    test = []
    for i in range(len(pos_instances)):
        if i < int(rate*len(pos_instances)):
            train.append(pos_instances[i])
        else:
            test.append(pos_instances[i])
    
    for i in range(len(neg_instances)):
        if i < int(rate*len(neg_instances)):
            train.append(neg_instances[i])
        else:
            test.append(neg_instances[i])
    return train, test