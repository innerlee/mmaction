import pytest
import torch

from mmaction.core.train import parse_losses


def test_parse_loss():
    with pytest.raises(TypeError):
        losses = dict(loss=0.5)
        parse_losses(losses)

    a_loss = [torch.randn(5, 5), torch.randn(5, 5)]
    b_loss = torch.randn(5, 5)
    losses = dict(a_loss=a_loss, b_loss=b_loss)
    r_a_loss = sum(_loss.mean() for _loss in a_loss)
    r_b_loss = b_loss.mean()
    r_loss = [r_a_loss, r_b_loss]
    r_loss = sum(r_loss)

    loss, log_vars = parse_losses(losses)

    assert r_loss == loss
    assert set(log_vars.keys()) == set(['a_loss', 'b_loss', 'loss'])
    assert log_vars['a_loss'] == r_a_loss
    assert log_vars['b_loss'] == r_b_loss
    assert log_vars['loss'] == r_loss
