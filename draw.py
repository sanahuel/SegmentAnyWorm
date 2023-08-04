import os
import cv2
import argparse
import numpy as np
from pathlib import Path

PATH = Path(__file__).resolve().parents[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=PATH / 'data', help='path to images, default ./data')
    parser.add_argument('--masks', type=str, default=PATH / 'output', help='path to masks, default ./output')
    return parser.parse_args()

def main(args):
    imgs = os.listdir(args.data)
    masks = os.listdir(args.masks)
    for img in imgs:
        print(f'--Img: {img}')
        img_path = os.path.join(args.data, img)
        image = cv2.imread(img_path)
        overlay = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        for mask in masks:
            if mask.replace('.png', '').replace('.jpg','').split('_mask')[0] == img.replace('.png', '').replace('.jpg',''):
                
                print(f'--{mask}')
                mask_path = os.path.join(args.masks, mask)
                mask = cv2.imread(mask_path)
                overlay = cv2.bitwise_or(overlay, mask)
                
                darkness = np.sum(overlay, axis=-1)
                threshold = 113
                dark_pixel_mask = darkness > threshold
                target_color = (255, 144, 30)
                overlay[dark_pixel_mask] = target_color

        combined = cv2.addWeighted(image, 1 - 0.5, overlay, 0.5, 0)
        output_path = os.path.join(args.masks, img)
        cv2.imwrite(output_path, combined)

if __name__ == '__main__':
    args = parse_args()
    main(args)