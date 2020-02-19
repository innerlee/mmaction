import numpy as np
import torch

from mmaction.models.heads import I3DHead, TSNHead


def test_i3d_head_loss():
    """Test i3d head loss when truth is empty and non-empty."""
    self = I3DHead(num_classes=4)

    input_shape = (30, 2048, 4, 7, 7)
    rng = np.random.RandomState(0)
    feat = rng.rand(*input_shape)
    feat = torch.FloatTensor(feat)

    cls_scores = self.forward(feat)

    # When truth is non-empty then cls loss should be nonzero for random inputs
    gt_labels = torch.LongTensor([2] * 30).squeeze()
    losses = self.loss(cls_scores, gt_labels)
    assert 'loss_cls' in losses.keys()
    assert losses.get('loss_cls') > 0, 'cls loss should be non-zero'


def test_tsn_head_loss():
    """Test tsn head loss."""
    self = TSNHead(num_classes=4)

    input_shape = (250, 2048, 7, 7)
    rng = np.random.RandomState(0)
    feat = rng.rand(*input_shape)
    feat = torch.FloatTensor(feat)

    num_segs = input_shape[0]
    cls_scores = self.forward(feat, num_segs)

    # When truth is non-empty then cls loss should be nonzero for random inputs
    gt_labels = torch.LongTensor([2])
    losses = self.loss(cls_scores, gt_labels)
    assert 'loss_cls' in losses.keys()
    assert losses.get('loss_cls') > 0, 'cls loss should be non-zero'
