
_base_ = '../../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

img_scale = (1280, 720)

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1)),
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='Resize',
        # img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
        #            (1333, 768), (1333, 800)],
        img_scale=img_scale,
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='YOLOXHSVRandomAug'),
    # dict(
    #     type='RandomAffine',
    #     scaling_ratio_range=(0.1, 2),
    #     border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    # dict(
    #     type='MixUp',
    #     img_scale=img_scale,
    #     ratio_range=(0.8, 1.6),
    #     pad_val=114.0),
    
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=img_scale,
        # img_scale=[(640, 360)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, img_scale=[img_scale]),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# learning policy
optimizer = dict(lr = 0.02/8)
lr_config = dict(warmup = None)
log_config = dict(interval=10)         

evaluation = dict(metric = ['bbox'],
                interval = 1)
checkpoint_config = dict(interval = 1)
classes = ('Fish',)
data_root = "data/2022_02_28_moorea"
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        pipeline=train_pipeline,
        img_prefix=f'{data_root}/images',
        classes=classes,
        ann_file=f'{data_root}/train_coco.json'),
    val=dict(
        pipeline=test_pipeline,
        img_prefix=f'{data_root}/images',
        classes=classes,
        ann_file=f'{data_root}/val_coco.json'),
    test=dict(
        pipeline=test_pipeline,
        img_prefix=f'{data_root}/images',
        classes=classes,
        ann_file=f'{data_root}/test_coco.json'))
seed = 0
runner = dict(max_epochs=10)
# data_root = '../data/2022_02_14_coco_format_select_vids/'
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
work_dir = 'work_dir/moorea_yolox_all_transforms_1280'
