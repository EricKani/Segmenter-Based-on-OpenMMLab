_base_ = [
    '../_base_/models/MySegmenter.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
find_unused_parameters = True
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
    backbone=dict(type='Deit',
                  final_norm=True,
                  img_size=(512, 512),
                  drop_rate=0.,
                  drop_path_rate=0.1), # deit-spectific
    decode_head=dict(# todo
            type='MySegmenterHead_maskT',
            d_encoder=768,
            n_layers=2,
            n_heads=12,
            d_model=768,
            d_ff=4*768,
            drop_path_rate=0.0,
            dropout=0.1,
            in_channels=256,
            channels=512,
            in_index=0,
            dropout_ratio=0, # no relation
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
    )

optimizer = dict(
    lr=0.001,
    weight_decay=0.0,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})) # todo?

# num_gpus: 4 -> batch_size: 8
data = dict(samples_per_gpu=2)
