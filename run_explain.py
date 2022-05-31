import pandas as pd
import numpy as np
import json
import csv
import os
import sys
import torch
import random
import argparse
import datetime
from collections import defaultdict


from ditto_explainer import DittoExplainer
from utils import load_data,formulate_instance,generate_student_training_instances,split_two_sets
from ditto_helper import student_ground_truth_ditto,load_teacher_ditto, sequentialize_ditto

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs.json")
    parser.add_argument("--datadir", type=str, default="./restaurant/")
    parser.add_argument("--modeldir", type=str, default="food_ditto.pt")
    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument("--dumpexp", type=bool, default=False)
    parser.add_argument("--expmethod", type=str, default="greedy")


    
    hp = parser.parse_args()
    
    print("Loading teacher model and original dataset...")
    tests = []
    tableA, tableB, original_set = load_data(hp.datadir)

    for _,item in original_set.iterrows():
        test_instance = sequentialize_ditto(item, tableA, tableB)
        tests.append(test_instance)
    configs = json.load(open(hp.config))
    config_index = {conf['idx'] : conf for conf in configs}
    config = config_index[hp.datadir[:-1][2:]]
    model = load_teacher_ditto(config['lm'],hp.modeldir,config, hp.gpu)
    print("Processing records...")
    res = student_ground_truth_ditto(model, tests, config)
    outputs = []
    assert(len(res) == len(tests))
    for idx,rec in original_set.iterrows():
        line = str(rec[0])+","+str(rec[1])+","+str(res[idx])
        outputs.append(line)
    assert(len(outputs) == len(tests))
    train_set, test_set = split_two_sets(outputs)

    print("Generating Explanations...")
    de = DittoExplainer(model, config)
    total_exp = 0
    training_with_exp = []
    total_time = 0.0
    explains = []
    samples = []

    for i,item in train_set.iterrows():
        # print("dealing with item: "+str(i))
        instance = formulate_instance(tableA, tableB, item)
        starttime = datetime.datetime.now()
        explain_res, eval_cnt = de.explain(instance, method=hp.expmethod)
        endtime = datetime.datetime.now()
        total_exp = total_exp + eval_cnt
        total_time = total_time + (endtime - starttime).seconds
        explains.append(explain_res[0])
        samples.append(explain_res[1])
        ground_truth_label = str(item[2])
        # inject new training instances
        student_training_instance = generate_student_training_instances(instance, explain_res[0], ground_truth_label)
        training_with_exp.append(student_training_instance)

    strain = []
    for _,item in train_set.iterrows():
        ditto_instance = sequentialize_ditto(item, tableA, tableB)
        strain.append(ditto_instance)

    tableA, tableB, test_set = load_data(hp.datadir,2)
    stest = []
    for _,item in test_set.iterrows():
        ditto_instance = sequentialize_ditto(item, tableA, tableB)
        stest.append(ditto_instance)    

    print("Dumping datasets for student model...")

    # dump explains
    if hp.dumpexp == True:
        exp_path = os.path.join(hp.datadir, "explains.json")
        with open(exp_path, 'w') as f:
            json.dump(explains, f, indent=2)
    # dump the training set, training set with explanations and test set for student model
    with open(os.path.join(hp.datadir, "train.txt"),"w") as f:
        for line in trains:
            f.write(line+"\n")

    if hp.expmethod == "greedy":
        fname = "train.txt.explain_inj"
    else:
        fname = "train.txt.explain_inj_bs"
    with open(os.path.join(hp.datadir, fname),"w") as f:
        for line in strain:
            f.write(line+"\n")
        for line in training_with_exp:
            f.write(line+"\n")
        
    with open(os.path.join(hp.datadir, "test.txt"),"w") as f:
        for line in stest:
            f.write(line+"\n")
