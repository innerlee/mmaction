import copy
import os.path as osp

from .base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class RawframeDataset(BaseDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    ```
    some/directory-1 163 1
    some/directory-2 122 1
    some/directory-3 258 2
    some/directory-4 234 2
    some/directory-5 295 3
    some/directory-6 121 3
    ```

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str): Path to a directory where videos are held.
        test_mode (bool): store True when building test dataset.
        filename_tmpl (str): Template for each filename.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg'):
        super(RawframeDataset, self).__init__(ann_file, pipeline, data_prefix,
                                              test_mode)
        self.filename_tmpl = filename_tmpl

    def load_annotations(self):
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                # frame_dir, total_frames, label = line.strip().split(' ')
                results = line.strip().split(' ')
                if len(results) < 3:
                    continue
                frame_dir = results[0]
                total_frames = results[1]
                label = results[2]

                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                video_infos.append(
                    dict(
                        frame_dir=frame_dir,
                        total_frames=int(total_frames),
                        label=int(label)))
        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        return self.pipeline(results)
