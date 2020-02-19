import os.path as osp

from mmaction.core import mean_class_accuracy, top_k_accuracy
from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class VideoDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    ```
    some/path/000.mp4 1
    some/path/001.mp4 1
    some/path/002.mp4 2
    some/path/003.mp4 2
    some/path/004.mp4 3
    some/path/005.mp4 3
    ```
    """

    def load_annotations(self):
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                filename, label = line.split(' ')
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, label=int(label)))
        return video_infos

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 topk=(1, 5),
                 logger=None):
        """Evaluation in rawframe dataset.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            logger (obj): Training logger. Defaults: None.
            topk (tuple[int]): K value for top_k_accuracy metric.
                Defaults: (1, 5).

        Return:
            eval_results (dict): Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError('results must be a list, but got {}'.format(
                type(results)))

        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if not isinstance(topk, (int, tuple)):
            raise TypeError(
                'topk must be int or tuple of int, but got {}'.format(
                    type(topk)))

        if isinstance(topk, int):
            topk = (topk, )

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['top_k_accuracy', 'mean_class_accuracy']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gt_labels = [ann['label'] for ann in self.video_infos]

        for metric in metrics:
            if metric == 'top_k_accuracy':
                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                for k, acc in zip(topk, top_k_acc):
                    eval_results['top{}_acc'.format(k)] = acc

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc

        return eval_results
