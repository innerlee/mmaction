"""
pytest tests/test_recognizers.py
"""
import copy
from os.path import dirname, exists, join

import numpy as np
import pytest
import torch
import torch.nn.functional as F


def _get_config_directory():
    """ Find the predefined recognizer config directory """
    try:
        # Assume we are running in the source mmaction repo
        repo_dpath = dirname(dirname(__file__))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmaction
        repo_dpath = dirname(dirname(mmaction.__file__))
    config_dpath = join(repo_dpath, 'config')
    if not exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def _get_config_module(fname):
    """
    Load a configuration as a python module
    """
    from xdoctest.utils import import_module_from_path
    config_dpath = _get_config_directory()
    config_fpath = join(config_dpath, fname)
    config_mod = import_module_from_path(config_fpath)
    return config_mod


def _get_recognizer_cfg(fname):
    """
    Grab configs necessary to create a recognizer. These are deep copied to
    allow for safe modification of parameters without influencing other tests.
    """
    import mmcv
    config = _get_config_module(fname)
    model = copy.deepcopy(config.model)
    train_cfg = mmcv.Config(copy.deepcopy(config.train_cfg))
    test_cfg = mmcv.Config(copy.deepcopy(config.test_cfg))
    return model, train_cfg, test_cfg


def test_tsn_forward():
    model, train_cfg, test_cfg = _get_recognizer_cfg(
        'tsn_rgb_1x1x3_r50_2d_kinetics400_100e.py')  # flake8: E501
    model['backbone']['pretrained'] = None
    # 'type' has been popped in test_config.py
    model['cls_head']['consensus'] = dict(type='AvgConsensus', dim=1)

    from mmaction.models import build_recognizer
    recognizer = build_recognizer(
        model, train_cfg=train_cfg, test_cfg=test_cfg)  # flake8: E501

    input_shape = (1, 3, 3, 224, 224)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')

    gt_labels = mm_inputs['gt_labels']
    losses = recognizer.forward(imgs, gt_labels)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img in img_list:
            result = recognizer.forward(one_img, None, return_loss=False)
            batch_results.append(result)

    cls_score = torch.rand(5, 400)
    with pytest.raises(KeyError):
        wrong_test_cfg = dict(clip='score')
        recognizer = build_recognizer(
            model, train_cfg=train_cfg, test_cfg=wrong_test_cfg)
        recognizer.average_clip(cls_score)

    with pytest.raises(ValueError):
        wrong_test_cfg = dict(average_clips='softmax')
        recognizer = build_recognizer(
            model, train_cfg=train_cfg, test_cfg=wrong_test_cfg)
        recognizer.average_clip(cls_score)

    test_cfg = dict(average_clips='score')
    recognizer = build_recognizer(
        model, train_cfg=train_cfg, test_cfg=test_cfg)
    score = recognizer.average_clip(cls_score)
    assert torch.equal(score, cls_score.mean(dim=0))

    test_cfg = dict(average_clips='prob')
    recognizer = build_recognizer(
        model, train_cfg=train_cfg, test_cfg=test_cfg)
    score = recognizer.average_clip(cls_score)
    assert torch.equal(score, F.softmax(cls_score, dim=1).mean(dim=0))


def test_i3d_forward():
    model, train_cfg, test_cfg = _get_recognizer_cfg(
        'i3d_rgb_32x2x1_r50_3d_kinetics400_100e.py')
    model['backbone']['pretrained2d'] = False
    model['backbone']['pretrained'] = None

    from mmaction.models import build_recognizer
    recognizer = build_recognizer(
        model, train_cfg=train_cfg, test_cfg=test_cfg)

    input_shape = (1, 3, 3, 8, 32, 32)
    mm_inputs = _demo_mm_inputs(input_shape, model_type='i3d')

    imgs = mm_inputs.pop('imgs')

    gt_labels = mm_inputs['gt_labels']
    losses = recognizer.forward(imgs, gt_labels)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        img_list = [g[None, :] for g in imgs]
        batch_results = []
        for one_img in img_list:
            result = recognizer.forward(one_img, None, return_loss=False)
            batch_results.append(result)

    cls_score = torch.rand(5, 400)
    with pytest.raises(KeyError):
        wrong_test_cfg = dict(clip='score')
        recognizer = build_recognizer(
            model, train_cfg=train_cfg, test_cfg=wrong_test_cfg)
        recognizer.average_clip(cls_score)

    with pytest.raises(ValueError):
        wrong_test_cfg = dict(average_clips='softmax')
        recognizer = build_recognizer(
            model, train_cfg=train_cfg, test_cfg=wrong_test_cfg)
        recognizer.average_clip(cls_score)

    test_cfg = dict(average_clips='score')
    recognizer = build_recognizer(
        model, train_cfg=train_cfg, test_cfg=test_cfg)
    score = recognizer.average_clip(cls_score)
    assert torch.equal(score, cls_score.mean(dim=0))

    test_cfg = dict(average_clips='prob')
    recognizer = build_recognizer(
        model, train_cfg=train_cfg, test_cfg=test_cfg)
    score = recognizer.average_clip(cls_score)
    assert torch.equal(score, F.softmax(cls_score, dim=1).mean(dim=0))


def _demo_mm_inputs(input_shape=(1, 3, 3, 224, 224), model_type='tsn'):
    """
    Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 250, 3, 224, 224).
        model_type (str): Model type for data generation. Default:'tsn'
    """
    if len(input_shape) == 5:
        (N, L, C, H, W) = input_shape
    elif len(input_shape) == 6:
        (N, M, C, L, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    if model_type == 'tsn':
        gt_labels = torch.LongTensor([2] * N)
    elif model_type == 'i3d':
        gt_labels = torch.LongTensor([2] * M)
    else:
        raise ValueError('Data type {} is not available'.format(model_type))

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs),
        'gt_labels': gt_labels,
    }
    return mm_inputs
