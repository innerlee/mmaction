# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
mc_cfg = dict(
    server_list_cfg="/mnt/lustre/share/memcached_client/server_list.conf",
    client_cfg="/mnt/lustre/share/memcached_client/client.conf",
    sys_path='/mnt/lustre/share/pymc/py3')
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='FrameSelector', io_backend='memcached', **mc_cfg),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=0,
    train=dict(
        type='RawframeDataset',
        ann_file='data/benchlist.txt',
        data_prefix='',
        pipeline=train_pipeline,
        filename_tmpl='{:05}.jpg'))
log_level = 'INFO'
workflow = [('train', 1)]
