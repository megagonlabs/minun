import json
import os
import sys
import jsonlines


datasets = ['AG',
            'DA',
            'DS',
            'WA',
            'AB']
# datasets = ['AB']

def update_configs():
    """Update configurations. """
    configs = json.load(open('configs.json'))
    names = [task['name'] for task in configs]

    for dataset in datasets:
        for subtask, suf in zip(['explain', 'explain_bs', 'noexplain'],
                                ['explain_inj', 'explain_inj_bs', 'explain']):
            task_name = dataset + '-minun-' + subtask
            if task_name not in names:
                trainset = 'train.txt.%s' % suf
                config = {'name': task_name,
            "task_type": "classification",
            "vocab": ["0", "1"],
            "trainset": "data/minun/%s/%s" % (dataset, trainset),
            "validset": "data/minun/%s/train.txt.explain" % dataset,
            "testset": "data/minun/%s/test.txt.explain" % dataset}
                configs.append(config)

    json.dump(configs, open('configs.json', 'w'), indent=2)


def add_labels(gpu_id):
    """Verify and re-label the train/valid/test sets for explanations."""
    model_paths = ['Structured/Amazon-Google',
                   'Structured/DBLP-ACM',
                   'Structured/DBLP-GoogleScholar',
                   'Structured/Walmart-Amazon',
                   'Textual/Abt-Buy']
    for dataset, path in zip(datasets, model_paths):
        for suffix in ['train.txt.explain', 'train.txt.explain_inj_bs',
                       'train.txt.explain_inj', 'test.txt.explain']:
            input_path = 'data/minun/%s/%s' % (dataset, suffix)
            output_path = 'data/minun/%s/%s.output' % (dataset, suffix)

            # if True:
            if not os.path.exists(output_path):
                cmd = """CUDA_VISIBLE_DEVICES=%d python matcher.py \
                   --task %s \
                   --input_path %s \
                   --output_path %s \
                   --lm roberta \
                   --max_len 256 \
                   --use_gpu \
                   --fp16 \
                   --checkpoint_path checkpoints_minun/""" % (gpu_id, path, input_path, output_path)
                os.system(cmd)

            with jsonlines.open(output_path, 'r') as reader:
                fout = open(input_path, 'w')

                for rid, row in enumerate(reader):
                    left, right, label = row['left'], row['right'], row['match']
                    fout.write('%s\t%s\t%d\n' % (left, right, label))

                fout.close()

gpu_id = 1
update_configs()
add_labels(gpu_id)

if __name__ == '__main__':
    for dataset in datasets:
        # train model w/o explanation
        os.system("""CUDA_VISIBLE_DEVICES=%d python train_ditto.py \
                  --task %s \
                  --logdir checkpoints_minun_new \
                  --batch_size 64 \
                  --max_len 256 \
                  --lr 3e-5 \
                  --n_epochs 40 \
                  --lm distilbert \
                  --fp16""" % (gpu_id, dataset + '-minun-noexplain'))

        # train model w explanation
        os.system("""CUDA_VISIBLE_DEVICES=%d python train_ditto.py \
                  --task %s \
                  --logdir checkpoints_minun_new \
                  --batch_size 64 \
                  --max_len 256 \
                  --lr 3e-5 \
                  --n_epochs 40 \
                  --lm distilbert \
                  --fp16""" % (gpu_id, dataset + '-minun-explain'))

        # train model w explanation
        os.system("""CUDA_VISIBLE_DEVICES=%d python train_ditto.py \
                  --task %s \
                  --logdir checkpoints_minun_new \
                  --batch_size 64 \
                  --max_len 256 \
                  --lr 3e-5 \
                  --n_epochs 40 \
                  --lm distilbert \
                  --fp16""" % (gpu_id, dataset + '-minun-explain_bs'))
