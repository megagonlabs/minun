import torch
import torch.nn as nn
import os
import numpy as np
import jsonlines
import time
import argparse
import sklearn
import eem.explain

from tqdm import tqdm
from scipy.special import softmax

from pretrain import evaluate
from dataset import PretrainDataset

from eem.utils import load_model, set_seed, to_str, classify


def predict(input_path, output_path, config,
            model,
            batch_size=1024,
            lm='distilbert',
            max_len=256,
            threshold=None,
            inject_variants=False):
    """Run the model over the input file containing the candidate entry pairs

    Args:
        input_path (str): the input file path
        output_path (str): the output file path
        config (Dictionary): task configuration
        model (PretrainModel): the model for prediction
        batch_size (int): the batch size
        max_len (int, optional): the max sequence length
        threshold (float, optional): the threshold of the 0's class

    Returns:
        None
    """
    pairs = []

    def process_batch(rows, pairs, writer):
        predictions, logits = classify(pairs, model, lm=lm,
                                       max_len=max_len,
                                       threshold=threshold)
        # try:
        #     predictions, logits = classify(pairs, model, lm=lm,
        #                                    max_len=max_len,
        #                                    threshold=threshold)
        # except:
        #     # ignore the whole batch
        #     return
        scores = softmax(logits, axis=1)
        for row, pred, score in zip(rows, predictions, scores):
            output = {'left': row[0], 'right': row[1],
                'row_id': row[2],
                'match': pred,
                'match_confidence': score[int(pred)]}
            writer.write(output)

    # input_path can also be train/valid/test.txt
    # convert to jsonlines
    if '.txt' in input_path:
        with jsonlines.open(input_path + '.jsonl', mode='w') as writer:
            for line in open(input_path):
                writer.write(line.split('\t')[:2])
        input_path += '.jsonl'

    if inject_variants == 'shap':
        tokenizer, shap_exp = eem.explain.get_shap_explanations(input_path, model,
                              lm,
                              max_len)
    elif inject_variants == 'lime':
        _, lime_exp = eem.explain.get_lime_explanations(input_path, model,
                              lm,
                              max_len)

    # batch processing
    start_time = time.time()
    with jsonlines.open(input_path) as reader,\
         jsonlines.open(output_path, mode='w') as writer:
        pairs = []
        rows = []
        for idx, row in tqdm(enumerate(reader)):
            pairs.append(to_str(row[0], row[1], max_len))
            rows.append([row[0], row[1], idx])

            if len(pairs) == batch_size:
                process_batch(rows, pairs, writer)
                pairs.clear()
                rows.clear()

            variants = []
            if inject_variants == 'baseline':
                variants = eem.explain.create_baseline_variants(row[0], row[1])
            elif inject_variants == 'shap':
                for ratio in [0.1, 0.3, 0.5, 0.7]:
                    variants.append(eem.explain.create_shap_variants(shap_exp, tokenizer, idx,
                                                         ratio))
            elif inject_variants == 'lime':
                left, right = to_str(row[0], row[1], max_len).split('\t')[:2]
                # token level
                for ratio in [0.1, 0.3, 0.5, 0.7]:
                    variants.append(eem.explain.create_lime_token_variants(lime_exp, idx,
                                                               left, right, ratio))

                # attribute level
                variants.append(eem.explain.create_lime_attr_variants(lime_exp, idx, left, right))

            # add and predict labels for varaints
            for new_left, new_right in variants:
                eem.explain.total_model_calls += 1
                pairs.append(to_str(new_left, new_right, max_len))
                rows.append([new_left, new_right, idx])
                if len(pairs) == batch_size:
                    process_batch(rows, pairs, writer)
                    pairs.clear()
                    rows.clear()

        if len(pairs) > 0:
            process_batch(rows, pairs, writer)

    run_time = time.time() - start_time
    run_tag = '%s_lm=%s' % (config['name'], lm)
    os.system('echo %s %f >> log.txt' % (run_tag, run_time))
    os.system('echo %s %s %d >> num_model_calls.txt' % (inject_variants, config['name'], eem.explain.total_model_calls))


def tune_threshold(config, model, hp):
    """Tune the prediction threshold for a given model on a validation set.

    Args:
        config (Dictionary): the task configuration
        model (PretrainModel): the Pretrain Model for prediction
        hp (Namespace): the hyperparameters

    Returns:
        float: the threshold of the best F1 on the validation set
    """
    validset = config['validset']
    task = hp.task

    # set_seed(123)

    # load dev sets
    valid_dataset = PretrainDataset(validset,
                                 max_len=hp.max_len,
                                 lm=hp.lm)

    valid_iter = torch.utils.data.DataLoader(dataset=valid_dataset,
                                 batch_size=64,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=PretrainDataset.pad)

    # acc, prec, recall, f1, v_loss, th = eval_classifier(model, valid_iter,
    #                                                     get_threshold=True)
    f1, th = evaluate(model, valid_iter, threshold=None)

    # verify F1
    set_seed(123)
    predict(validset, "tmp.jsonl", config, model,
            max_len=hp.max_len,
            lm=hp.lm,
            threshold=th)

    predicts = []
    with jsonlines.open("tmp.jsonl", mode="r") as reader:
        for line in reader:
            predicts.append(int(line['match']))
    os.system("rm tmp.jsonl")

    labels = []
    with open(validset) as fin:
        for line in fin:
            labels.append(int(line.split('\t')[-1]))

    real_f1 = sklearn.metrics.f1_score(labels, predicts)
    print("load_f1 =", f1)
    print("real_f1 =", real_f1)

    return th


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='Structured/Beer')
    parser.add_argument("--inject_variants", type=str, default='None') # None, baseline, shap
    parser.add_argument("--input_path", type=str, default='input/candidates_small.jsonl')
    parser.add_argument("--output_path", type=str, default='output/matched_small.jsonl')
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default='checkpoints/')
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=256)
    hp = parser.parse_args()

    # load the models
    set_seed(123)
    config, model = load_model(hp.task, hp.checkpoint_path,
                       hp.lm, hp.use_gpu, hp.fp16)


    # tune threshold
    threshold = tune_threshold(config, model, hp)

    # run prediction
    predict(hp.input_path, hp.output_path, config, model,
            max_len=hp.max_len,
            lm=hp.lm,
            threshold=threshold,
            inject_variants=hp.inject_variants)
