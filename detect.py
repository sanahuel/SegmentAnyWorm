import os
import cv2
import argparse
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

PATH = Path(__file__).resolve().parents[0]

def save_images(tensor, save_path_prefix):
    # Save individual masks
    n, _, _, _ = tensor.size()
    for i in range(n):
        image_array = tensor[i, 0].cpu().numpy()
        image_array = (image_array * 255).astype('uint8')
        image = Image.fromarray(image_array, mode='L')
        image_path = f"{save_path_prefix}_mask{i}.png"
        image.save(image_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=PATH / 'data', help='path to images, default ./data')
    parser.add_argument('--output', type=str, default=PATH / 'output', help='directory to output masks, default ./output')
    parser.add_argument('--yolo', type=str, default=PATH / 'YOLOv8s_General.pt', help='path to yolo weights')
    parser.add_argument('--sam', type=str, default=PATH / 'sam_vit_h_4b8939.pth', help='path to sam weights')
    parser.add_argument('--device', type=str, default='cpu', help='device to use, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()

def main(args):
    imgs = os.listdir(args.data)

    # Load Yolo model
    model = YOLO(args.yolo)
    print('YOLO loaded successfully')

    # Load SAM model
    sam_checkpoint = args.sam
    model_type = "vit_h"
    device = args.device if args.device == 'cpu' else 'cuda'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    print('SAM loaded successfully')

    for img in imgs:
        print(f'--Img: {img}')
        img_path = os.path.join(args.data, img)
        image = cv2.imread(img_path)

        # Yolo detction
        results = model(img_path, device=args.device, verbose=False)
        boxes = results[0].boxes.xyxy
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
        print(f'Detection completed')
            
        # SAM segmentation
        predictor.set_image(image)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        print(f'Segmentation completed')

        img_prefix = img.replace('.jpg', '').replace('.png', '')
        save_images(masks, os.path.join(args.output ,img_prefix))

if __name__ == '__main__':
    args = parse_args()
    main(args)