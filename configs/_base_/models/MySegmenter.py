# model settings
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained= 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',  # noqa
    backbone=dict(
        type='VisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(11), #(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        interpolate_mode='bicubic'),
    decode_head=dict(# todo
        type='MySegmenterHead',
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

    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
