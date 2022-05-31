import jsonlines
import os
import shap
import pickle
import torch
import scipy as sp
import numpy as np

from tqdm import tqdm
from scipy.special import softmax
from lime.lime_text import LimeTextExplainer
from dataset import PretrainDataset
from .utils import *

total_model_calls = 0

def get_shap_explanations(input_path, model, lm, max_len):
    """Generate SHAP explanations for pairs in a .jsonl file

    Args:
        input_path (str): the path to the input .jsonl file
        model (PretrainModel): the model for inference
        lm (str): the language model name
        max_len (int): max sequence length (tokenization)

    Returns:
        Tokenizer: the tokenizer from the lm (for post-processing)
        List of tuples: the explanations (to be verified)
    """
    with jsonlines.open(input_path) as reader:
        pairs = []
        for idx, row in tqdm(enumerate(reader)):
            pairs.append(to_str(row[0], row[1], max_len))

    dataset = PretrainDataset(pairs,
                           max_len=max_len,
                           lm=lm)
    tokenizer = dataset.tokenizer

    cache_path = input_path + '.shap'
    if os.path.exists(cache_path):
        shap_exp = pickle.load(open(cache_path, 'rb'))
    else:
        # build explainer
        def f(x):
            global total_model_calls
            total_model_calls += len(x)
            with torch.no_grad():
                # padding
                inputs = [tokenizer.encode(v) for v in x]
                maxlen = max([len(x) for x in inputs])
                inputs = [xi + [0]*(maxlen - len(xi)) for xi in inputs]
                tv = torch.tensor(inputs).cuda()

                outputs = model(tv).detach().cpu().numpy()
                scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
                val = sp.special.logit(scores[:,1]) # use one vs rest logit units

            return val

        explainer = shap.Explainer(f, tokenizer)
        sequences = [tokenizer.decode(tokenizer.encode(v,
            add_special_tokens=True,
            max_length=dataset.max_len,
            truncation=True)) for v in dataset.pairs]
        shap_exp = explainer(sequences, fixed_context=1)
        pickle.dump(shap_exp, open(cache_path, 'wb'))

    return tokenizer, shap_exp


def get_lime_explanations(input_path, model, lm, max_len,
                          batch_size=512):
    """Generate LIME explanations for pairs in a .jsonl file
    """
    with jsonlines.open(input_path) as reader:
        pairs = []
        for idx, row in tqdm(enumerate(reader)):
            pairs.append(to_str(row[0], row[1], max_len))

    dataset = PretrainDataset(pairs,
                           max_len=max_len,
                           lm=lm)
    tokenizer = dataset.tokenizer

    cache_path = input_path + '.lime'
    if os.path.exists(cache_path):
        lime_exp = pickle.load(open(cache_path, 'rb'))
    else:
        def preprocess(text_pair):
            res = ''
            for tid, text in enumerate(text_pair):
                tokens = text.split(' ')
                for i, token in enumerate(tokens):
                    if token not in ['COL', 'VAL']:
                        res += '%d_%d_%s ' % (tid, i, token)

            return res

        def postprocess(text):
            res = ["", ""]
            tokens = text.split(' ')
            current_id = 0

            for token in tokens:
                LL = token.split('_')
                if LL[0].isdigit() and int(LL[0]) in [0, 1]:
                    current_id = int(LL[0])
                res[current_id] += LL[-1] + ' '

            return res[0], res[1]

        def pad(batch):
            maxlen = max([len(x) for x in batch])
            batch = [xi + [0]*(maxlen - len(xi)) for xi in batch]
            return torch.tensor(batch).cuda()

        def predict_proba(x):
            global total_model_calls
            total_model_calls += len(x)
            new_x = [postprocess(v) for v in x]

            tv = [tokenizer.encode(v) for v in new_x]
            res = []
            batch = []
            batch_seqs = []
            seen = {}

            with torch.no_grad():
                for v, elem in zip(new_x, tv):
                    if v in seen:
                        res.append(seen[v])
                    else:
                        batch.append(elem)
                        batch_seqs.append(v)

                        if len(batch) == batch_size:
                            batch = pad(batch).cuda()
                            outputs = model(batch)[0].softmax(-1).detach().cpu().numpy().tolist()
                            res += outputs
                            batch = []
                            for seq, out in zip(batch_seqs, outputs):
                                seen[seq] = out
                            batch_seqs.clear()

            if len(batch) > 0:
                batch = pad(batch).cuda()
                res += model(batch).softmax(-1).detach().cpu().numpy().tolist()
            return np.array(res)

        explainer = LimeTextExplainer(class_names=['0','1'])
        lime_exp = []
        for v in tqdm(dataset.pairs):
            lime_exp.append(explainer.explain_instance(preprocess(v),
                    predict_proba,
                    num_samples=300,
                    num_features=100))

        pickle.dump(lime_exp, open(cache_path, 'wb'))

    return tokenizer, lime_exp


def create_baseline_variants(left, right, delete=True, copy=True):
    """Inject variants by deleting spans of length 3 and copying attributes. """
    results = []
    # inject deletion variants
    if delete:
        left_tokens = left.split(' ')
        for i, token in enumerate(left_tokens):
            if token.lower() not in ['col', 'val']:
                # remove span of length 3
                new_left = ' '.join(left_tokens[:i] + left_tokens[i+3:])
                results.append((new_left, right))

    # inject copy variants
    if copy:
        ent_attrs = []
        for entity in [left, right]:
            attrs = []
            attributes = entity.split('COL')[1:]
            for attr_val in attributes:
                attr, val = attr_val.split('VAL')
                attrs.append([attr, val])
            ent_attrs.append(attrs)

        for left_av, right_av in zip(ent_attrs[0], ent_attrs[1]):
            ori_left = left_av[1]
            left_av[1] = right_av[1]

            new_left = ''
            for attr, val in ent_attrs[0]:
                new_left += 'COL %s VAL %s ' % (attr, val)
            new_left = new_left.strip()
            results.append((new_left, right))
            left_av[1] = ori_left

    return results


def create_shap_variants(shap_values, tokenizer, idx, ratio=0.5):
    """Create variants by delete tokens with highest shap values."""
    data = list(shap_values.data[idx])
    sorted_values = sorted([(shap_values.values[idx][i], i) \
                            for i in range(len(shap_values.values[idx])) \
                            if data[i].strip() not in [tokenizer.sep_token,
                                tokenizer.bos_token,
                                tokenizer.cls_token, 'COL', 'VAL']],
                           reverse=True)

    N = len(sorted_values)
    for i in range(int(N*ratio)):
        data[sorted_values[i][1]] = ''
    for i in range(N):
        if data[i] in [tokenizer.bos_token, tokenizer.cls_token]:
            data[i] = ''

    seq = ''.join(data)
    left_right = seq.split(tokenizer.sep_token)
    ans = []
    for item in left_right:
        if len(item.strip()) > 0:
            ans.append(item.strip())

    while len(ans) < 2:
        ans.append('')

    return ans[0], ans[1]


def create_lime_token_variants(lime_exp, idx, left, right, ratio=0.5):
    """Create token level explanations based on lime's output."""
    exp = lime_exp[idx].as_list()
    left_tokens = left.split(' ')
    right_tokens = right.split(' ')

    value_ids = []
    for token, value in exp:
        LL = token.split('_')
        if len(LL) == 3 and LL[0] in ['0', '1'] and \
                LL[1].isdigit() and len(LL[-1]) > 0:
            value_ids.append((value, int(LL[0]), int(LL[1])))

    value_ids.sort(reverse=True)
    N = len(value_ids)
    for i in range(int(N*ratio)):
        lid = value_ids[i][1]
        tid = value_ids[i][2]
        if lid == 0:
            if tid < len(left_tokens):
                left_tokens[tid] = ''
        else:
            if tid < len(right_tokens):
                right_tokens[tid] = ''

    return " ".join(left_tokens), " ".join(right_tokens)



def create_lime_attr_variants(lime_exp, idx, left, right, ratio=0.5):
    """Create attribute level explanations based on lime's output."""
    # Drop the most significant attribute
    attr_scores = []
    exp = lime_exp[idx].as_list()
    value_mp = {}
    for token, value in exp:
        LL = token.split('_')
        if len(LL) == 3 and LL[0] in ['0', '1'] and \
                LL[1].isdigit() and len(LL[-1]) > 0:
            value_mp[(int(LL[0]), int(LL[1]))] = value

    tokens = [left.split(' '), right.split(' ')]
    for lid, token_list in enumerate(tokens):
        attr_id = -1
        for tid, token in enumerate(token_list):
            if token == 'COL':
                attr_id += 1
                if attr_id >= len(attr_scores):
                    attr_scores.append([0.0])

            if (lid, tid) in value_mp:
                attr_scores[attr_id].append(value_mp[(lid, tid)])

    # aggregate token scores into attribute scores
    max_score = -1e8
    max_score_id = 0
    for i in range(len(attr_scores)):
        score = np.mean(attr_scores[i][:5])
        if score > max_score:
            max_score = score
            max_score_id = i


    res = []
    left_attr = ''
    for lid, token_list in enumerate(tokens):
        attr_id = -1
        target = ""

        for token in token_list:
            if token == 'COL':
                attr_id += 1
                # copy left
                if attr_id == max_score_id and lid == 1:
                    target += left_attr

            if attr_id != max_score_id:
                target += token + ' '
            else:
                if lid == 0:
                    left_attr += token + ' '
                else:
                    pass

        res.append(target)

    return res[0], res[1]
