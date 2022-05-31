import random
import jsonlines
import os
import sys

if __name__ == '__main__':
    # dataset = 'Textual/Abt-Buy/'
    dataset = sys.argv[1]

    output_path = os.path.join('data/er_magellan/', dataset, 'test.output.jsonl')
    new_train_path = os.path.join('data/er_magellan/', dataset, 'train.txt.explain')
    new_train_inj_path = os.path.join('data/er_magellan/', dataset, 'train.txt.explain_inj')
    new_test_path = os.path.join('data/er_magellan/', dataset, 'test.txt.explain')

    pairs = []
    used = {}
    counter_examples = []
    with jsonlines.open(output_path, 'r') as reader:
        for row in reader:
            left = row['left']
            right = row['right']
            label = row['match']
            row_id = row['row_id']
            if row_id not in used:
                pairs.append('\t'.join([left, right, str(label)]))
                used[row_id] = str(label)
                counter_examples.append([])
            else:
                if str(label) != used[row_id]:
                    counter_examples[row_id].append('\t'.join([left, right, str(label)]))

    # random.shuffle(pairs)
    N = len(pairs)

    # train.txt
    with open(new_train_path, 'w') as fout:
        for p in pairs[:N//2]:
            fout.write(p + '\n')

    with open(new_train_inj_path, 'w') as fout:
        for row_id, p in enumerate(pairs[:N//2]):
            fout.write(p + '\n')
            for ce in counter_examples[row_id][:1]:
                fout.write(ce + '\n')

    with open(new_test_path, 'w') as fout:
        for p in pairs[N//2:]:
            fout.write(p + '\n')

