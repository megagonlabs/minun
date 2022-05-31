import os
import sys
import torch
import random
import numpy as np


from pretrain import PretrainModel
from dataset import PretrainDataset
from torch.utils import data

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sequentialize_ditto(instance, tableA, tableB):
    '''
    Args:
        instance(string):  The test instance in the format of left id, right id, label
        tableA/tableB(dataframe): The dataframe of records in the left/right table
    Returns:
        line(str): The sequentialized result in the format of Ditto
    '''
    left = ""
    right = ""
    idxA = instance[0]
    idxB = instance[1]
    label = str(instance[2])
    header = list(tableA)
    for i,attr in enumerate(header):
        left += " COL "+str(attr)+" VAL "+str(tableA.iloc[idxA, i])
        right += " COL "+str(attr)+" VAL "+str(tableB.iloc[idxB, i])
    line = left.strip()+"\t"+right.strip()+"\t"+label
    return line

    
def load_teacher_ditto(lm, model_path, config, gpu=True, fp16=False):
    if not os.path.exists(model_path):
        raise ModelNotFoundError(model_path)
    if gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'
    model = PretrainModel(device=device, lm=lm)
    saved_state = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(saved_state)
    model = model.to(device)
    if fp16 and 'cuda' in device:
        from apex import amp
        model = amp.initialize(model, opt_level='O2')
    return model
    
    
def student_ground_truth_ditto(model, records, args):
    '''
    model (MultiTaskNet): The trained Ditto model
    records (Array): The test sets, each item is in the Ditto format
    args(defaultdict): the arguments necessaqry to eval the model
    '''
    dataset = PretrainDataset(records, args['vocab'], args['name'], lm=args['lm'], max_len=args['max_len'])
    iterator = data.DataLoader(dataset=dataset,
                                 batch_size=args['batch_size'],
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=PretrainDataset.pad)
    Y_hat = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, mask, y, seqlens, taskname = batch
            taskname = taskname[0]
            _, _, y_hat = model(x, y, task=taskname)  # y_hat: (N, T)
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    results = []
    for i in range(len(records)):
        pred = dataset.idx2tag[Y_hat[i]]
        results.append(str(pred))
    return results


def predict_ditto(inputs, config, model, batch_size=64):
    set_seed(20220104)
    dataset = PretrainDataset(inputs, config['vocab'], config['name'], lm=config['lm'], max_len=config['max_len'])
    iterator = data.DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=PretrainDataset.pad)
    Y_logits = []
    Y_hat = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, mask, y, seqlens, taskname = batch
            taskname = taskname[0]
            logits, _, y_hat = model(x, y, task=taskname)  # y_hat: (N, T)
            Y_logits += logits.softmax(-1).cpu().numpy().tolist()
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    results = []
    for i in range(len(inputs)):
        pred = dataset.idx2tag[Y_hat[i]]
        results.append(pred)

    return results, Y_logits