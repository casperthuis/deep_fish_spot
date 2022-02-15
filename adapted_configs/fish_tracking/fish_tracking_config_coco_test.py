# The new config inherits a base config to highlight the necessary modification
# _base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
# base = "../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py"

_base_ = [
    '../../configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/coco_detection.py', '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/schedules/schedule_1x.py'
]
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    pretrained=None,
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))

# Modify dataset related settings
dataset_type = 'COCODataset'
data_root = '../data/2022_02_14_coco_format_select_vids/'
classes = ('predator',)
data = dict(
    workers_per_gpu=1,
    train=dict(
        img_prefix='../data/2022_02_14_coco_format_select_vids/images',
        classes=classes,
        ann_file='../data/2022_02_14_coco_format_select_vids/train_coco.json'),
    val=dict(
        img_prefix='../data/2022_02_14_coco_format_select_vids/images',
        classes=classes,
        ann_file='../data/2022_02_14_coco_format_select_vids/val_coco.json'),
    test=dict(
        img_prefix='../data/2022_02_14_coco_format_select_vids/images',
        classes=classes,
        ann_file='../data/2022_02_14_coco_format_select_vids/test_coco.json'))
work_dir = '../work_dir/test_run_coco'  # Directory to save the model checkpoints and logs for the current experiments.
# train_pipeline = [dict(
#         type='Resize',
#         img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
#                    (1333, 768), (1333, 800)],
#         multiscale_mode='value',
#         keep_ratio=True)]


# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"