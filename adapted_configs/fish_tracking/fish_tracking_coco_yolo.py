_base_ = '../../configs/yolox/yolox_s_8x8_300e_coco.py'

# model settings
model = dict(
    random_size_range=(10, 20),
    backbone=dict(deepen_factor=0.33, widen_factor=0.375),
    neck=dict(in_channels=[96, 192, 384], out_channels=96),
    bbox_head=dict(in_channels=96, feat_channels=96, num_classes=1))

img_scale = (760, 1352)
# img_scale = (640, 640)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(760, 1352),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data_root = "data/2022_02_14_coco_format_select_vids"
classes = ('predator',)
train_dataset = dict(
    pipeline=train_pipeline,
    dataset=dict(
    img_prefix=f'{data_root}/images',
    classes=classes,
    ann_file=f'{data_root}/train_coco.json'))

test_dataset = dict(
    pipeline=test_pipeline,
    img_prefix=f'{data_root}/images',
    classes=classes,
    ann_file=f'{data_root}/test_coco.json')

val_dataset = dict(
    pipeline=test_pipeline,
    img_prefix=f'{data_root}/images',
    classes=classes,
    ann_file=f'{data_root}/val_coco.json')


data = dict(
    samples_per_gpu=4,
    train=train_dataset,
    val=val_dataset,
    test=test_dataset)

evaluation = dict(metric = ['bbox'],
                interval = 2)
checkpoint_config = dict(interval = 1)


seed = 0
runner = dict(max_epochs=10)

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
work_dir = 'work_dir/yolox_tiny_fish_tracking_resx05'