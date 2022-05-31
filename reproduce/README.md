# Evaluation Framework For Entity Matching

Install required packages:

```
conda create -n minun python=3.9
conda activate minun
pip install -r requirements.txt
```

Special packages (you may need to adapt to your cuda version):
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge shap
conda install pandas
```

NVIDIA apex for fp16 training:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir .
cd ..
```


## Train a model M to be explained

For example:
```
CUDA_VISIBLE_DEVICES=0 python train_model.py \
  --task Textual/Abt-Buy \
  --batch_size 64 \
  --max_len 256 \
  --lr 3e-5 \
  --n_epochs 10 \
  --lm roberta \
  --fp16 \
  --save_model \
  --logdir checkpoints/
```

Parameters:
* ``--task``: the name of the tasks (see ``configs.json``)
* ``--batch_size``, ``--max_len``, ``--lr``, ``--n_epochs``: the batch size, max sequence length, learning rate, and the number of epochs
* ``--lm``: the language model (``distilbert`` or ``roberta``).
* ``--fp16``: whether train with the half-precision floating point optimization
* ``--save_model``: if this flag is on, then save the checkpoint to ``{logdir}/{task}/model.pt``.


## Generate explanations (LIME, SHAP, CF-Baseline)

```
CUDA_VISIBLE_DEVICES=0 python matcher.py \
   --task Textual/Abt-Buy \
   --inject_variants lime \
   --input_path data/er_magellan/Textual/Abt-Buy/test.txt \
   --output_path data/er_magellan/Textual/Abt-Buy/test.output.jsonl \
   --lm roberta \
   --max_len 256 \
   --use_gpu \
   --fp16 \
   --checkpoint_path checkpoints/
```

Parameters:
* ``--task``: the name of the tasks (same as above)
* ``--inject_variants``: the explanation methods. We currently support ``lime``, ``shap``, and ``baseline``.
* ``--input_path``: the path to the test set (instances to be explained)
* ``--output_path``: the output path of the explanations
* ``--max_len``, ``--lm``, ``--checkpoint_path``: should be the same as above
* ``--fp16``, ``--use_gpu``: always turn them on
* ``--save_model``: if this flag is on, then save the checkpoint to ``{logdir}/{task}/model.pt``.

## Create train/test sets with target model's predictions and injected explanations

```
python create_datasets.py Textual/Abt-Buy
```

Running this script will:
* split the test sets into 50:50 ``('train.txt.explain', 'test.txt.explain')``
* Use M to label the two datasets (prediction already made when running ``matcher.py``).
* Inject explanations into ``train.txt.explain`` -> ``train.txt.explain_inj``

## Train the student model to obtain the F1 scores

This step using the same training code as the one for the target model (Ditto).

No explanation:
```
CUDA_VISIBLE_DEVICES=0 python train_model.py \
  --task Textual/Abt-Buy-noexplain \
  --logdir checkpoints_lime \
  --batch_size 64 \
  --max_len 256 \
  --lr 3e-5 \
  --n_epochs 40 \
  --lm distilbert \
  --fp16
```

With explanation:
```
CUDA_VISIBLE_DEVICES=0 python train_model.py \
  --task Textual/Abt-Buy-explain \
  --logdir checkpoints_lime \
  --batch_size 64 \
  --max_len 256 \
  --lr 3e-5 \
  --n_epochs 40 \
  --lm distilbert \
  --fp16
```

## Train with Minun explanations

These tasks in ``configs.json`` with the suffix ``-minun-explain`` and ``-minum-explain_bs`` are datasets injected with Minun's explanations. ``-explain_bs`` means that the explanations are generated with binary search (otherwise greedy).

For example:
```
CUDA_VISIBLE_DEVICES=0 python train_model.py \
  --task AB-minun-explain \
  --logdir checkpoints_minun \
  --batch_size 64 \
  --max_len 256 \
  --lr 3e-5 \
  --n_epochs 40 \
  --lm distilbert \
  --fp16
```

## View sample explanations

Use the jupyter notebook ``visualize.ipynb``.

## Complete experiment scripts:

See ``run_all.py`` and ``run_all_minun.py``.
