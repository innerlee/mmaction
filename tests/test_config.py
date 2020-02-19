import os
import os.path as osp

import mmcv


def _get_config_directory():
    """ Find the predefined recognizer config directory """
    try:
        # Assume we are running in the source mmaction repo
        repo_dir = osp.dirname(osp.dirname(__file__))
    except NameError:
        # For IPython development when this __file__ is not defined
        import mmaction
        repo_dir = osp.dirname(osp.dirname(mmaction.__file__))
    config_dpath = osp.join(repo_dir, 'config')
    if not osp.exists(config_dpath):
        raise Exception('Cannot find config path')
    return config_dpath


def test_config_build_recognizer():
    """
    Test that all mmaction models defined in the configs can be initialized.
    """
    from mmaction.models import build_recognizer

    config_dpath = _get_config_directory()
    print('Found config_dpath = {!r}'.format(config_dpath))

    import glob
    config_fpaths = list(glob.glob(osp.join(config_dpath, '*.py')))
    config_names = [os.path.relpath(p, config_dpath) for p in config_fpaths]

    print('Using {} config files'.format(len(config_names)))

    for config_fname in config_names:
        config_fpath = osp.join(config_dpath, config_fname)
        config_mod = mmcv.Config.fromfile(config_fpath)

        config_mod.model
        config_mod.train_cfg
        config_mod.test_cfg
        print('Building recognizer, config_fpath = {!r}'.format(config_fpath))

        # Remove pretrained keys to allow for testing in an offline environment
        if 'pretrained' in config_mod.model['backbone']:
            config_mod.model['backbone']['pretrained'] = None

        recognizer = build_recognizer(
            config_mod.model,
            train_cfg=config_mod.train_cfg,
            test_cfg=config_mod.test_cfg)
        assert recognizer is not None
