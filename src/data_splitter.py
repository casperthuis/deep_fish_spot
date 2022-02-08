import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split


def main(args):
    """ 
    Splits data set into the medium dataset annotation for mmdetection code.

    """
    # load data
    # loop over all subfolders
    camera_angles = ["A", "B", "C"] 
    images_dict = {}

    for s in os.listdir(args.dataset_origin):
        sub_dir = os.path.join(args.dataset_origin, s)
        images_dir = os.path.join(sub_dir, "images")
        print(images_dir)
        for c in camera_angles:
            camare_angle_dir = os.path.join(images_dir, c)
            print(camare_angle_dir)
            for i in os.listdir(camare_angle_dir):
                print(i)
            

        print(s)
    # devise naming sceme for data
    # 


    # determine ratios for train/test/val on args

    # split data 

    # Create txt files based on splits

    # Create subfolders for image and labels based on splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_origin",
        "-o",
        type=str,
        default="tracked_clips_2020",
        help="Location of the original dataset folder"
    )

    parser.add_argument(
        "--dataset_destination",
        "-d",
        type=str,
        required=True,
        help="Location of parsed dataset"
    )

    parser.add_argument(
        "--test_size",
        "-t",
        type=str,
        required=True,
        help="Location of parsed dataset"
    )

    parser.add_argument(
        "--val_size",
        "-v",
        type=str,
        help="Location of parsed dataset"
    )

    args = parser.parse_args()
    main(args)
