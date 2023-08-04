"""
Finuting script for YOLOv8
Check arguments on https://docs.ultralytics.com/modes/train/#arguments
"""

import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data', type=str, default=None, help="path to data file, i.e. coco128.yaml")
    
    # Weights
    parser.add_argument('--model', type=str, default=None, help="path to weights")
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs to train for")
    parser.add_argument('--patience', type=int, default=50, help="epochs to wait for no observable improvement for early stopping of training")
    parser.add_argument('--batch', type=int, default=16, help="number of images per batch (-1 for AutoBatch)")
    parser.add_argument('--imgsz', type=int, default=640, help="size of input images as integer")
    parser.add_argument('--save', type=bool, default=True, help="save train checkpoints and predict results")
    parser.add_argument('--save_period', type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument('--cache', type=str, default='False', help="True/ram, disk or False. Use cache for data loading")
    
    # Device and parallelism
    parser.add_argument('--device', type=str, default=None, help="device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu")
    parser.add_argument('--workers', type=int, default=8, help="number of worker threads for data loading (per RANK if DDP)")
    
    # Experiment
    parser.add_argument('--project', type=str, default=None, help="project name")
    parser.add_argument('--name', type=str, default=None, help="experiment name")
    parser.add_argument('--exist_ok', type=bool, default=False, help="whether to overwrite existing experiment")
    parser.add_argument('--pretrained', type=bool, default=False, help="whether to use a pretrained model")

    # Optimization
    parser.add_argument('--optimizer', type=str, default='auto', choices=['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'], help="optimizer to use")
    parser.add_argument('--verbose', type=bool, default=False, help="whether to print verbose output")
    parser.add_argument('--seed', type=int, default=0, help="random seed for reproducibility")
    parser.add_argument('--deterministic', type=bool, default=True, help="whether to enable deterministic mode")
    parser.add_argument('--single_cls', type=bool, default=False, help="train multi-class data as single-class")
    parser.add_argument('--rect', type=bool, default=False, help="rectangular training with each batch collated for minimum padding")
    parser.add_argument('--cos_lr', type=bool, default=False, help="use cosine learning rate scheduler")

    # Other options
    parser.add_argument('--close_mosaic', type=int, default=0, help="(int) disable mosaic augmentation for final epochs")
    parser.add_argument('--resume', type=bool, default=False, help="resume training from last checkpoint")
    parser.add_argument('--amp', type=bool, default=True, choices=[True, False], help="Automatic Mixed Precision (AMP) training")
    parser.add_argument('--fraction', type=float, default=1.0, help="dataset fraction to train on")
    parser.add_argument('--profile', type=bool, default=False, help="profile ONNX and TensorRT speeds during training for loggers")

    # Learning rate
    parser.add_argument('--lr0', type=float, default=0.01, help="initial learning rate")
    parser.add_argument('--lrf', type=float, default=0.01, help="final learning rate (lr0 * lrf)")

    # Optimizer parameters
    parser.add_argument('--momentum', type=float, default=0.937, help="SGD momentum/Adam beta1")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="optimizer weight decay")

    # Warmup
    parser.add_argument('--warmup_epochs', type=float, default=3.0, help="warmup epochs (fractions ok)")
    parser.add_argument('--warmup_momentum', type=float, default=0.8, help="warmup initial momentum")
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1, help="warmup initial bias lr")

    # Loss function weights
    parser.add_argument('--box', type=float, default=7.5, help="box loss gain")
    parser.add_argument('--cls', type=float, default=0.5, help="cls loss gain (scale with pixels)")
    parser.add_argument('--dfl', type=float, default=1.5, help="dfl loss gain")
    parser.add_argument('--pose', type=float, default=12.0, help="pose loss gain (pose-only)")
    parser.add_argument('--kobj', type=float, default=2.0, help="keypoint obj loss gain (pose-only)")
    parser.add_argument('--label_smoothing', type=float, default=0.0, help="label smoothing (fraction)")

    # Miscellaneous
    parser.add_argument('--nbs', type=int, default=64, help="nominal batch size")
    parser.add_argument('--overlap_mask', type=bool, default=True, help="masks should overlap during training (segment train only)")
    parser.add_argument('--mask_ratio', type=int, default=4, help="mask downsample ratio (segment train only)")
    parser.add_argument('--dropout', type=float, default=0.0, help="use dropout regularization (classify train only)")
    parser.add_argument('--val', type=bool, default=True, help="validate/test during training")

    return parser.parse_args()

def main(args):
    
    print('Loading YOLO model...')
    model = YOLO(args.model)

    print('Finetuning...')
    model.train(**vars(args))

if __name__ == "__main__":
    args = parse_args()
    main(args)
    