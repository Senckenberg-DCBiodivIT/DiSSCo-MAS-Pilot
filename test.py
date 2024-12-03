import os
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import torch
import cv2
import numpy as np
import utils.utils as utils
import logging

from collections import defaultdict
from torchvision import tv_tensors
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.ops import nms
from PIL import Image
from torchvision.transforms.v2 import functional as F
from calculate_scale_pixel import get_scale_cm, calculate_scale_pixel
from torch.cuda.amp import autocast, GradScaler
from utils_helper import get_transform, clear_cuda, get_random_color, load_model_checkpoint
from model import get_model_instance_segmentation
from utils.parameters import *
from PIL import Image


class InferencePlantDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = []

        for img_file in os.listdir(os.path.join(root, 'scans')):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, 'scans', img_file)
                self.images.append(img_path)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = tv_tensors.Image(torch.tensor(img))
        img = img.clone().detach().permute(2, 0, 1) 

        if self.transforms:
            img = self.transforms(img)

        return img,img_path

def prediction(model, data_loader, device, iou_threshold=0.5, score_threshold=0.5):
    
    folder_path = 'inference'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model.eval()

    image_counter = 0
    scale_detection_counter = 0
    scale_text_recognition_counter = 0
    class_count = defaultdict(int) 
    char_accuracies = []   
    image_class_count = defaultdict(lambda: defaultdict(int))
    image_list = data_loader.dataset.images
    initial_downscale_factor = 1
    for images, img_paths in data_loader:
        for img, image_path in zip(images, img_paths):
            with autocast():
                image = img.unsqueeze(0).to(device)
                original_image = cv2.imread(image_path)
                original_height, original_width = original_image.shape[0], original_image.shape[1]
                downscale_factor = initial_downscale_factor
                downscaled_image = original_image
                fits_in_memory = False
                image_name = image_path.split('/')[-1]                       
                image_counter += 1
                print(f"Running inference on Herbarium sheet:{image_name}")           

                while not fits_in_memory and downscale_factor > 0.1:
                    try:
                        new_height, new_width = int(original_image.shape[0] * downscale_factor), int(original_image.shape[1] * downscale_factor)
                        downscaled_image = cv2.resize(original_image, (new_width, new_height))
                        img_tensor = F.to_tensor(downscaled_image).unsqueeze(0).to(device)
                        with torch.no_grad():
                            predictions = model(img_tensor)            
                        fits_in_memory = True 
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logging.warning(f"Out of memory at scale {downscale_factor}. Reducing scale and retrying.")
                            torch.cuda.empty_cache()
                            downscale_factor *= 0.8
                        else:
                            raise e
                           
                pred = predictions[0]
                boxes = pred['boxes'].cpu()
                labels = pred['labels'].cpu()
                scores = pred['scores'].cpu()
                masks = (pred['masks'] > 0.5).squeeze(1).cpu()  
                    
                high_score_indices = scores >= score_threshold
                boxes = boxes[high_score_indices]
                labels = labels[high_score_indices]
                scores = scores[high_score_indices]
                masks = masks[high_score_indices]

                nms_indices = nms(boxes, scores, iou_threshold)
                boxes = boxes[nms_indices]
                labels = labels[nms_indices]
                scores = scores[nms_indices]
                masks = masks[nms_indices]

                upscale_factor = 1 / downscale_factor
                boxes *= upscale_factor
                
                masks= torch.nn.functional.interpolate(masks.unsqueeze(1).float(), size=original_image.shape[:2], mode="bilinear", align_corners=False).squeeze(1)
                masks = masks > score_threshold  

                original_img_tensor = F.to_tensor(original_image).unsqueeze(0)
                output_image = original_img_tensor.clone().squeeze(0)
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

                one_cm_in_pixel, scale_detection_counter, boxes_scale, scale_text_recognition_counter, char_accuracy = get_scale_cm(original_image, downscaled_image, downscale_factor, score_threshold, device, scale_detection_counter, scale_text_recognition_counter)

                if char_accuracy > 0:
                    char_accuracies.append(char_accuracy)
                                    
                if boxes_scale is not None and len(boxes_scale) > 0:
                    output_image = draw_bounding_boxes(output_image, boxes_scale[0].unsqueeze(0), labels=["scale"], colors=[get_random_color()], width=30, font=font_path, font_size=100)

                for i in range(len(boxes)):
                    color = get_random_color()
                    boolean_mask = masks[i].byte().to(torch.bool)
                    class_name = HERBARIUM_CLASSES[labels[i].item()]
                    label_text = f"{class_name}: {scores[i]:.2f}"

                    class_count[class_name] += 1
                    image_class_count[image_name][class_name] += 1

                    leaf_pixel_area = masks[i].sum().item()
                    if one_cm_in_pixel > 0:
                        leaf_pixel_area_in_cm = leaf_pixel_area / ( one_cm_in_pixel * one_cm_in_pixel)
                        leaf_pixel_area_in_cm = round(leaf_pixel_area_in_cm, 1)
                        label_text = f"{HERBARIUM_CLASSES[labels[i].item()]}: {scores[i]:.2f}, Area: {leaf_pixel_area_in_cm:.1f} cm²"
                        print(f"The herbarium sheet is {image_name} and the {class_name} is {leaf_pixel_area_in_cm} cm²")
                        
                    output_image = draw_bounding_boxes(output_image, boxes[i].unsqueeze(0), labels=[label_text], colors=[color], width=10, font=font_path, font_size=100)
                    output_image = draw_segmentation_masks(output_image, boolean_mask.unsqueeze(0), alpha=0.5, colors=[color])

                base_filename = os.path.basename(image_name)
                filename_without_extension = os.path.splitext(base_filename)[0]
                image_name = f"{filename_without_extension}.png"      

                save_path = os.path.join(folder_path, image_name)

                output_image_pil = T.ToPILImage()(output_image)
                output_image_pil.save(save_path)

                print(f"Scale detection counter is {scale_detection_counter}")
                print(f"Scale text counter is {scale_text_recognition_counter}")
                    
                del img
                torch.cuda.empty_cache()
                clear_cuda()

                print("\n--- Overall Statistics ---")
                print("Class counts across all images:")
                for class_name, count in class_count.items():
                    print(f"  {class_name}: {count}")

                if char_accuracies:
                    average_char_accuracy = sum(char_accuracies) / len(char_accuracies)
                    print(f"\nOverall average OCR character accuracy: {average_char_accuracy:.2f}")
                else:
                    print("\nNo OCR character accuracy metrics collected.")



if __name__ == "__main__":
    dataset = InferencePlantDataset("test_image", get_transform(train=False))
    print(f"The total images for inference is {len(dataset)}")
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=0)
    num_classes = len(HERBARIUM_CLASSES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'plant_organ_model_checkpoint.pth'
    model = get_model_instance_segmentation(num_classes)
    if os.path.exists(checkpoint_path):
        start_epoch = load_model_checkpoint(checkpoint_path, model, device)
        model.to(device)
        prediction(model, data_loader, device)
    else: 
        print("The model checkpoint path is not found")
    
    