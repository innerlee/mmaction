import numpy as np
import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmaction.models import ResNet, ResNet3d


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for m in modules:
        if isinstance(m, _BatchNorm):
            if m.training != train_state:
                return False
    return True


def test_resnet_backbone():
    """Test resnet backbone"""
    with pytest.raises(KeyError):
        # ResNet depth should be in [18, 34, 50, 101, 152]
        ResNet(20)

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        ResNet(50, num_stages=0)

    with pytest.raises(AssertionError):
        # In ResNet: 1 <= num_stages <= 4
        ResNet(50, num_stages=5)

    with pytest.raises(AssertionError):
        # len(strides) == len(dilations) == num_stages
        ResNet(50, strides=(1, ), dilations=(1, 1), num_stages=3)

    with pytest.raises(TypeError):
        self = ResNet(50, pretrained=0)
        self.init_weights()

    with pytest.raises(AssertionError):
        # style must be in ['pytorch', 'caffe']
        ResNet(18, style='tensorflow')

    with pytest.raises(AssertionError):
        # assert not with_cp
        ResNet(18, with_cp=True)

    self = ResNet(18)
    self.init_weights()

    self = ResNet(50, norm_eval=True)
    self.init_weights()
    self.train()
    assert check_norm_state(self.modules(), False)

    self = ResNet(
        pretrained='torchvision://resnet50', depth=50, norm_eval=True)
    self.init_weights()
    self.train()
    assert check_norm_state(self.modules(), False)

    frozen_stages = 1
    self = ResNet(50, frozen_stages=frozen_stages)
    self.init_weights()
    self.train()
    assert self.norm1.training is False
    for m in [self.conv1, self.norm1]:
        for param in m.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        m = getattr(self, 'layer{}'.format(i))
        for mm in m.modules():
            if isinstance(mm, _BatchNorm):
                assert mm.training is False
        for param in m.parameters():
            assert param.requires_grad is False

    self = ResNet(50, norm_eval=False)
    self.init_weights()
    self.train()

    input_shape = (1, 3, 64, 64)
    imgs = _demo_inputs(input_shape)
    feat = self.forward(imgs)
    assert feat.shape == torch.Size([1, 2048, 2, 2])


def test_resnet3d_backbone():
    """Test resnet3d backbone"""
    with pytest.raises(KeyError):
        ResNet3d(18, None)

    with pytest.raises(AssertionError):
        # In ResNet3d: 1 <= num_stages <= 4
        ResNet3d(50, None, num_stages=0)

    with pytest.raises(AssertionError):
        # In ResNet3d: 1 <= num_stages <= 4
        ResNet3d(50, None, num_stages=5)

    with pytest.raises(AssertionError):
        # len(spatial_strides) == len(temporal_strides)
        # == len(dilations) == num_stages
        ResNet3d(
            50,
            None,
            spatial_strides=(1, ),
            temporal_strides=(1, 1),
            dilations=(1, 1, 1),
            num_stages=4)

    with pytest.raises(TypeError):
        self = ResNet3d(50, ['resnet', 'bninception'])
        self.init_weights()

    self = ResNet3d(50, None, pretrained2d=False, norm_eval=True)
    self.init_weights()
    self.train()
    assert check_norm_state(self.modules(), False)

    self = ResNet3d(50, 'torchvision://resnet50', norm_eval=True)
    self.init_weights()
    self.train()
    assert check_norm_state(self.modules(), False)

    self = ResNet3d(50, None, pretrained2d=False, norm_eval=False)
    self.init_weights()
    self.train()
    assert check_norm_state(self.modules(), True)

    frozen_stages = 1
    self = ResNet3d(50, None, pretrained2d=False, frozen_stages=frozen_stages)
    self.init_weights()
    self.train()
    assert self.norm1.training is False
    for m in [self.conv1, self.norm1]:
        for param in m.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        m = getattr(self, 'layer{}'.format(i))
        for mm in m.modules():
            if isinstance(mm, _BatchNorm):
                assert mm.training is False
        for param in m.parameters():
            assert param.requires_grad is False

    input_shape = (1, 3, 6, 64, 64)
    imgs = _demo_inputs(input_shape)
    feat = self.forward(imgs)
    assert feat.shape == torch.Size([1, 2048, 1, 2, 2])


def _demo_inputs(input_shape=(1, 3, 64, 64)):
    """
    Create a superset of inputs needed to run backbone.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).
    """
    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    imgs = torch.FloatTensor(imgs)

    return imgs
