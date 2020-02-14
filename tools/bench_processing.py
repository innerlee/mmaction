import argparse

import mmcv
from mmcv import Config

from mmaction import __version__
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.utils import get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('MMAction-Lite Version: {}'.format(__version__))
    logger.info('Config: {}'.format(cfg.text))

    dataset = build_dataset(cfg.data.train)

    data_loader = build_dataloader(
        dataset,
        cfg.data.videos_per_gpu,
        cfg.data.workers_per_gpu,
        cfg.gpus,
        dist=False)

    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        print(data['imgs'])
        for img in data['imgs']:
            # mean = torch.FloatTensor(cfg.img_norm_cfg['mean']).cuda()
            # std = torch.FloatTensor(cfg.img_norm_cfg['std']).cuda()

            prog_bar.update()


if __name__ == '__main__':
    main()
