import shutil
import tempfile
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmaction.core import EvalHook


class TestDataset(Dataset):

    def __init__(self):
        pass

    def evaluate(self, results, logger=None):
        return dict(test='success')

    def __getitem__(self, idx):
        results = dict(imgs=torch.ones((1, 5)).cuda())
        return results

    def __len__(self):
        return 1


class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.linear = nn.Linear(5, 5)

    def forward(self, imgs, return_loss=False):
        assert return_loss is False
        return self.linear(imgs)


def test_eval_hook():
    try:
        from torch.utils.data import DataLoader
    except ImportError:
        warnings.warn('Skipping test_save_checkpoint in the absense of torch')
        return
    import mmcv.runner
    test_dataset = TestDataset()
    loader = DataLoader(test_dataset)
    model = ExampleModel()
    eval_hook = EvalHook(test_dataset)
    tmpdir = tempfile.mkdtemp()
    runner = mmcv.runner.Runner(
        model=model.cuda(),
        batch_processor=lambda model, x, **kwargs: {
            'log_vars': {
                "accuracy": 0.98
            },
            'num_samples': 1
        },
        work_dir=tmpdir)
    runner.register_hook(eval_hook)
    runner.run([loader], [('train', 1)], 1)
    shutil.rmtree(tmpdir)
