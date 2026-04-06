import json
import cv2
import numpy as np
import os
from pathlib import Path

from torch.utils.data import Dataset


class MyDataset_depth(Dataset):
    def __init__(self, train_dir='fill100'):
        self.data = []
        self.train_dir = train_dir
        home_path = Path(__file__).parent
        self.home_path = home_path
        print('current path', home_path)
        file = Path(home_path, f'training/{self.train_dir}/prompt.json')
        with open(file.as_posix(), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        folder = Path(self.home_path, f'training/{self.train_dir}')
        sFile = Path(folder, source_filename)
        tFile = Path(folder, target_filename)
        source = cv2.imread(sFile.as_posix())
        target = cv2.imread(tFile.as_posix())

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

