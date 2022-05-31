import os
import argparse
import json
import sys
import torch
import numpy as np
import random


from dataset import PretrainDataset
from pretrain import train

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured/Fodors-Zagats")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--size", type=int, default=None)

    hp = parser.parse_args()

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # only a single task for baseline
    task = hp.task

    # create the tag of the run
    run_tag = '%s_lm=%s_size=%s_id=%d' % (task, hp.lm, str(hp.size), hp.run_id)
    run_tag = run_tag.replace('/', '_')

    # load task configuration
    configs = json.load(open('student_configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]
    trainset = config['student-train']
    validset = config['student-valid']
    testset = config['student-test']

    # load train/dev/test sets
    train_dataset = PretrainDataset(trainset,
                                   lm=hp.lm,
                                   max_len=hp.max_len,
                                   size=hp.size,
                                   da=hp.da)
    valid_dataset = PretrainDataset(validset, lm=hp.lm)
    test_dataset = PretrainDataset(testset, lm=hp.lm)

    # train and evaluate the model
    train(train_dataset,
          valid_dataset,
          test_dataset,
          run_tag, hp)
