import os
import json
import torch
import random
import numpy as np

from apex import amp
from torch.utils import data
from pretrain import evaluate, PretrainModel
from dataset import PretrainDataset


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch``
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_str(ent1, ent2, max_len=256):
    """Serialize a pair of data entries

    Args:
        ent1 (Dictionary): the 1st data entry
        ent2 (Dictionary): the 2nd data entry
        max_len (int, optional): the max sequence length

    Returns:
        string: the serialized version
    """
    content = ''
    for ent in [ent1, ent2]:
        if isinstance(ent, str):
            content += ent
        else:
            for attr in ent.keys():
                content += 'COL %s VAL %s ' % (attr, ent[attr])
        content += '\t'

    content += '0'

    # if summarizer is not None:
    #     content = summarizer.transform(content, max_len=max_len)

    new_ent1, new_ent2, _ = content.split('\t')
    # if dk_injector is not None:
    #     new_ent1 = dk_injector.transform(new_ent1)
    #     new_ent2 = dk_injector.transform(new_ent2)

    return new_ent1 + '\t' + new_ent2 + '\t0'


def classify(sentence_pairs, model,
             lm='distilbert',
             max_len=256,
             threshold=None):
    """Apply the Pretrain model on pre-processed input.

    Args:
        sentence_pairs (list of str): the sequence pairs
        model (MultiTaskNet): the model in pytorch
        max_len (int, optional): the max sequence length
        threshold (float, optional): the threshold of the 0's class

    Returns:
        list of int: the predicted labels
        list of float: the scores of the pairs
    """
    inputs = sentence_pairs
    dataset = PretrainDataset(inputs,
                           max_len=max_len,
                           lm=lm)

    iterator = data.DataLoader(dataset=dataset,
                               batch_size=len(dataset),
                               shuffle=False,
                               num_workers=0,
                               collate_fn=PretrainDataset.pad)

    # prediction
    all_probs = []
    all_logits = []
    with torch.no_grad():
        # print('Classification')
        for i, batch in enumerate(iterator):
            x, _ = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_logits += logits.cpu().numpy().tolist()

    if threshold is None:
        threshold = 0.5

    pred = [1 if p > threshold else 0 for p in all_probs]
    return pred, all_logits



def load_model(task, path, lm, use_gpu, fp16=True):
    """Load a model for a specific task.

    Args:
        task (str): the task name
        path (str): the path of the checkpoint directory
        lm (str): the language model
        use_gpu (boolean): whether to use gpu
        fp16 (boolean, optional): whether to use fp16

    Returns:
        Dictionary: the task config
        PretrainModel: the model
    """
    # load models
    checkpoint = os.path.join(path, task, 'model.pt')
    if not os.path.exists(checkpoint):
        raise ValueError("Model not found at %s" % checkpoint)

    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]
    config_list = [config]

    if use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    model = PretrainModel(device=device, lm=lm)

    saved_state = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    if 'model' in saved_state:
        model.load_state_dict(saved_state['model'])
    else:
        # temp fix
        keys = list(saved_state.keys())
        for name in keys:
            if 'fc.weight' in name:
                saved_state['fc.weight'] = saved_state[name]
                del saved_state[name]
            if 'fc.bias' in name:
                saved_state['fc.bias'] = saved_state[name]
                del saved_state[name]

        model.load_state_dict(saved_state)

    model = model.to(device)

    if fp16 and 'cuda' in device:
        model = amp.initialize(model, opt_level='O2')

    return config, model


