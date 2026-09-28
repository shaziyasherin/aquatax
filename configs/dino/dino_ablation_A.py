# Ablation row A - identical to row B except the Swin-B backbone starts from the ImageNet-22k
# CLASSIFICATION checkpoint instead of the Objects365/Grounding-DINO DETECTION checkpoint.
# B - A is therefore the effect of detection pretraining, isolated.
#
# The checkpoint deliberately uses window 12 / 384 px, matching the architecture the reported run
# declares (embed_dims 128, depths [2,2,18,2], window_size 12, pretrain_img_size 384). The
# window7/224 variant would force changing window_size as well, which would make this row change
# two things at once - exactly the defect Reviewer 4 attacked in the old Table 4.
#
# convert_weights=True here because this is the raw Microsoft-format checkpoint. The reported run
# uses convert_weights=False only because its weights were already extracted into mmdet's naming.
# That flag is a consequence of the checkpoint format, not an independent change.
#
# Download once (~350 MB), then point `imagenet22k` at the local file:
#   https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth

_base_ = './dino_ablation_B.py'

work_dir = 'D:/aquavisionnet/revision/work_dirs/ablA'

imagenet22k = 'D:/aquavisionnet/swin_base_patch4_window12_384_22k.pth'

model = dict(
    backbone=dict(
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=imagenet22k)))
