import os
import json
import torch
from torchvision import datasets

root = 'EuroSAT_MS'
split_dir = 'splits'
os.makedirs(split_dir, exist_ok=True)

seed = 42
g = torch.Generator().manual_seed(seed)


full = datasets.ImageFolder(root=root)


from collections import defaultdict
cls_to_indices = defaultdict(list)
for idx, (_, label) in enumerate(full.samples):
    cls_to_indices[label].append(idx)

train_indices, test_indices = [], []


for label, idxs in cls_to_indices.items():

    perm = torch.randperm(len(idxs), generator=g).tolist()
    idxs = [idxs[i] for i in perm]

    n_train = int(0.8 * len(idxs))
    train_indices.extend(idxs[:n_train])
    test_indices.extend(idxs[n_train:])


train_indices = sorted(train_indices)
test_indices = sorted(test_indices)


with open(os.path.join(split_dir, 'train_indices.json'), 'w') as f:
    json.dump(train_indices, f)
with open(os.path.join(split_dir, 'test_indices.json'), 'w') as f:
    json.dump(test_indices, f)

print(f'Train: {len(train_indices)}, Test: {len(test_indices)}')
