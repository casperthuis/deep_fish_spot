# import copy
import os.path as osp

import mmcv
import numpy as np

from mmdet.apis import set_random_seed
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset


@DATASETS.register_module()
# class KittiTinyDataset(CustomDataset):
class TrackedFishDataset(CustomDataset):

    # CLASSES = ('Car', 'Pedestrian', 'Cyclist')
    CLASSES = ('predator')

    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        image_list = mmcv.list_from_file(self.ann_file)
    
        data_infos = []
        # convert annotations to middle format
        for image_id in image_list:
            filename = f'{self.img_prefix}/{image_id}.jpeg'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]
    
            data_info = dict(filename=f'{image_id}.jpeg', width=width, height=height)
    
            # load annotations
            label_prefix = self.img_prefix.replace('image_2', 'label_2')

            
            lines = mmcv.list_from_file(osp.join(label_prefix, f'{image_id}.txt'))
    
            content = [line.strip().split(' ') for line in lines]
            bbox_names = [x[0] for x in content]
            bboxes = [[float(info) for info in x[4:8]] for x in content]
    
            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
    
            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long),
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                       dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos


def main():

    cfg = mmcv.Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')        
    
    # Modify dataset type and path
    # cfg.dataset_type = 'KittiTinyDataset'
    # cfg.data_root = 'kitti_tiny/'

    cfg.dataset_type = 'TrackedFishDataset'
    cfg.data_root = 'tracked_fish/'

    cfg.data.test.type = 'TrackedFishDataset'
    cfg.data.test.data_root = 'tracked_fish/'
    cfg.data.test.ann_file = 'test.txt'
    cfg.data.test.img_prefix = 'testing/image'

    # cfg.data.test.type = 'KittiTinyDataset'
    # cfg.data.test.data_root = 'kitti_tiny/'
    # cfg.data.test.ann_file = 'train.txt'
    # cfg.data.test.img_prefix = 'training/image_2'

    cfg.data.train.type = 'TrackedFishDataset'
    cfg.data.train.data_root = 'tracked_fish/'
    cfg.data.train.ann_file = 'train.txt'
    cfg.data.train.img_prefix = 'training/image_2'

    
    cfg.data.val.type = 'KittiTinyDataset'
    cfg.data.val.data_root = 'tracked_fish/'
    cfg.data.val.ann_file = 'val.txt'
    cfg.data.val.img_prefix = 'validation/image_2'

    # modify num classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = 1
    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = './tutorial_exps'

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optimizer.lr = 0.02 / 8
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 12
    
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 12

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)


    # We can initialize the logger for training and have a look
    # at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')


if __name__ == "__main__":  
   main()
