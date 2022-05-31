import json
import os
import sys
import time


datasets = ['Structured/Amazon-Google',
            'Structured/DBLP-ACM',
            'Structured/DBLP-GoogleScholar',
            'Structured/Walmart-Amazon',
            'Textual/Abt-Buy']

gpu_id = 1
inject_variants = 'lime'

def update_configs():
    """Update configurations. """
    configs = json.load(open('configs.json'))
    names = [task['name'] for task in configs]

    for dataset in datasets:
        for subtask in ['explain', 'noexplain']:
            if dataset + '-' + subtask not in names:
                if subtask == 'explain':
                    trainset = 'train.txt.explain_inj'
                else:
                    trainset = 'train.txt.explain'
                config = {'name': dataset + '-' + subtask,
            "task_type": "classification",
            "vocab": ["0", "1"],
            "trainset": "data/er_magellan/%s/%s" % (dataset, trainset),
            "validset": "data/er_magellan/%s/train.txt.explain" % dataset,
            "testset": "data/er_magellan/%s/test.txt.explain" % dataset}
                configs.append(config)

    json.dump(configs, open('configs.json', 'w'), indent=2)


if __name__ == '__main__':
    for dataset in datasets:
        # train models if not there
        if not os.path.exists('checkpoints/%s/model.pt' % dataset):
            os.system("""CUDA_VISIBLE_DEVICES=%d python train_student.py \
                      --task %s \
                      --batch_size 64 \
                      --max_len 256 \
                      --lr 3e-5 \
                      --n_epochs 10 \
                      --lm roberta \
                      --fp16 \
                      --save_model \
                      --da del""" % (gpu_id, dataset))

        # run predictions and generate counter examples
        start = time.time()
        os.system("""CUDA_VISIBLE_DEVICES=%d python matcher.py \
		   --task %s \
                   --inject_variants %s \
		   --input_path data/er_magellan/%s/test.txt \
		   --output_path data/er_magellan/%s/test.output.jsonl \
		   --lm roberta \
		   --max_len 256 \
		   --use_gpu \
		   --fp16 \
		   --checkpoint_path checkpoints/""" % (gpu_id, dataset, inject_variants,
                       dataset, dataset))

        # create datasets
        os.system('python create_datasets.py %s' % dataset)
        runtime = time.time() - start
        os.system('echo %s %s %f >> run_time_log.txt' % (inject_variants, dataset, runtime))

        # train model w/o explanation
        os.system("""CUDA_VISIBLE_DEVICES=%d python train_ditto.py \
                  --task %s \
                  --logdir checkpoints_%s \
                  --batch_size 64 \
                  --max_len 256 \
                  --lr 3e-5 \
                  --n_epochs 40 \
                  --lm distilbert \
                  --fp16 --run_id 2""" % (gpu_id, dataset + '-noexplain', inject_variants))

        # train model w explanation
        os.system("""CUDA_VISIBLE_DEVICES=%d python train_ditto.py \
                  --task %s \
                  --logdir checkpoints_%s \
                  --batch_size 64 \
                  --max_len 256 \
                  --lr 3e-5 \
                  --n_epochs 40 \
                  --lm distilbert \
                  --fp16 --run_id 2""" % (gpu_id, dataset + '-explain', inject_variants))
