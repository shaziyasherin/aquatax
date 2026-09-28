# mmdet 3.x config: DINO-4scale + Swin-B on AquaTax-10 unified corpus.
# v2 "aggressive push": Objects365-pretrained backbone, 1280px multi-scale,
# 18 epochs, AquaOpt-informed optimizer, trains on the SAM copy-paste
# augmented corpus. Run sam_copypaste_augment.py BEFORE this.
# Fits an RTX 4000 Ada (20 GB) at batch size 2 per GPU (with_cp=True).

_base_ = ['mmdet::dino/dino-4scale_r50_8xb2-12e_coco.py']

# --- data root & classes ---------------------------------------------------
data_root = 'D:/aquavisionnet/data/aquatax10/'
metainfo  = dict(classes=('plastic', 'paper', 'metal', 'glass'),
                 palette=[(220, 20, 60), (0, 200, 200), (140, 140, 140), (0, 220, 0)])
num_classes = 4

# --- backbone: Swin-B, Objects365-pretrained (via Grounding-DINO Swin-B) --
# Extracted backbone-only weights (run extract_backbone.py once — mmdet's
# SwinTransformer.init_weights() bypasses init_cfg's `prefix` field, so
# loading `backbone.*` straight out of the full Grounding-DINO checkpoint
# via `prefix=` doesn't work; the prefix has to be stripped ourselves first
# into a standalone local .pth, hence a local path here instead of a URL).
# window_size=12 / pretrain_img_size=384 matches the source checkpoint (NOT
# the window7/224 ImageNet-22k pilot variant) — keep these paired.
# convert_weights=False because the extracted state dict already uses
# mmdet's native Swin key naming (it came from another mmdet model), not
# the raw Microsoft-format checkpoint that convert_weights=True expects.
pretrained = 'D:/aquavisionnet/swin_b_o365_backbone_only.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
        window_size=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.3,
        patch_norm=True, out_indices=(1, 2, 3),
        with_cp=True,                       # activation checkpointing → saves VRAM
        convert_weights=False,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[256, 512, 1024]),
    bbox_head=dict(num_classes=num_classes),
    dn_cfg=dict(group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    test_cfg=dict(max_per_img=300))

# --- augmentation & pipelines: multi-scale up to 1280 short side ----------
image_size = (1280, 1280)
_ms_scales = [(s, 1400) for s in range(480, 1281, 32)]   # 26 scales, 480..1280 step 32

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoice', transforms=[
        [dict(type='RandomChoiceResize', scales=_ms_scales, keep_ratio=True)],
        [dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)],
              keep_ratio=True),
         dict(type='RandomCrop', crop_type='absolute_range',
              crop_size=(384, 600), allow_negative_crop=True),
         dict(type='RandomChoiceResize', scales=_ms_scales, keep_ratio=True)]]),
    dict(type='PackDetInputs')]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1400, 1280), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))]

# --- dataloaders: class-balanced sampling over the SAM copy-paste corpus --
train_dataloader = dict(
    batch_size=2, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,                       # replace base's CocoDataset dict wholesale
        type='ClassBalancedDataset',
        oversample_thr=1e-3,                 # LVIS-style; boosts paper/metal/glass
        dataset=dict(
            type='CocoDataset',
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotations/train_copypaste.json',   # <- run sam_copypaste_augment.py first
            data_prefix=dict(img=''),         # image paths are absolute in JSON
            filter_cfg=dict(filter_empty_gt=False, min_size=8),
            pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset', data_root=data_root, metainfo=metainfo,
        ann_file='annotations/val.json', data_prefix=dict(img=''),
        test_mode=True, pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset', data_root=data_root, metainfo=metainfo,
        ann_file='annotations/test.json', data_prefix=dict(img=''),
        test_mode=True, pipeline=test_pipeline))

val_evaluator  = dict(type='CocoMetric', ann_file=data_root + 'annotations/val.json',
                      metric='bbox', classwise=True)
test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/test.json',
                      metric='bbox', classwise=True)

# --- schedule: 18 epochs, cosine, AquaOpt-informed warmup fraction --------
max_epochs = 18
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=3)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

# warmup_frac=0.036 (AquaOpt/Paper-2 WOA-tuned value) applied to this run's
# expected ~16000-iter budget (18 epochs x ~890 iters/epoch on the augmented
# corpus) -> ~580 warmup iterations. Re-check against the actual
# "907/907"-style iter count printed at the start of epoch 1 in the log;
# if it's off by more than ~2x, it's still fine (LinearLR warmup length is
# not sensitive to being approximately right).
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=580),
    dict(type='CosineAnnealingLR', begin=0, end=max_epochs, by_epoch=True,
         T_max=max_epochs, eta_min_ratio=0.05)]

optim_wrapper = dict(
    type='AmpOptimWrapper',                        # fp16 mixed precision
    # AquaOpt (WOA-tuned) lr/wd from Paper 2's search, reused here as an
    # informed prior rather than re-running a full WOA search on a DETR-scale
    # model (too expensive for this pass). w_harm doesn't apply (no harm loss
    # term in DINO) so it's dropped.
    optimizer=dict(type='AdamW', lr=1.415e-4, weight_decay=4.564e-5),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone':       dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)}))

# --- runtime --------------------------------------------------------------
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=3, max_keep_ckpts=2,
                    save_best='coco/bbox_mAP_50', rule='greater'),
    logger=dict(type='LoggerHook', interval=25))

work_dir = 'D:/aquavisionnet/avn_out/dino_swinb_v2'
randomness = dict(seed=20260729, deterministic=False)
auto_scale_lr = dict(enable=False, base_batch_size=16)
