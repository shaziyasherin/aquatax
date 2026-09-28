# Ablation row B - detection-pretrained backbone, WITHOUT SAM copy-paste and WITHOUT the
# class-balanced wrapper. Everything else is inherited unchanged from the reported v2 config:
# same Swin-B architecture, same multi-scale pipeline, same 18 epochs, same optimiser, same seed.
#
# The ONLY difference from the reported run is the training dataset:
#   reported : ClassBalancedDataset( CocoDataset(annotations/train_copypaste.json) )   2159 imgs
#   row B    :                       CocoDataset(annotations/train.json)               1814 imgs
#
# Verify before training:
#   python scripts\16_mmdet_run.py dump --config configs\dino_ablation_B.py --out results\ablB_config_dump.py
#   python scripts\16_mmdet_run.py dump --config D:\aquavisionnet\dino_swinb_aquavision_v2.py --out results\reported_config_dump.py
#   fc results\reported_config_dump.py results\ablB_config_dump.py > results\diff_ablB.txt
# The diff must contain only work_dir, ann_file, and the removal of the ClassBalancedDataset
# wrapper. Anything else means this row is not a controlled comparison.

_base_ = 'D:/aquavisionnet/dino_swinb_aquavision_v2.py'

work_dir = 'D:/aquavisionnet/revision/work_dirs/ablB'

train_dataloader = dict(
    dataset=dict(
        _delete_=True,                       # drop the ClassBalancedDataset wrapper entirely
        type='CocoDataset',
        data_root='D:/aquavisionnet/data/aquatax10/',
        metainfo=dict(classes=('plastic', 'paper', 'metal', 'glass'),
                      palette=[(220, 20, 60), (0, 200, 200), (140, 140, 140), (0, 220, 0)]),
        ann_file='annotations/train.json',   # original 1,814-image corpus, no synthetic images
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=8),
        pipeline={{_base_.train_pipeline}}))
