import torch
import cv2
import numpy as np
import os
import uuid
import requests
import logging
import websocket
import json

from torchvision.transforms import functional as F
from torchvision.transforms import v2 as T
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from io import BytesIO
from PIL import Image
from utils_helper import get_transform, clear_cuda, get_random_color, load_model_checkpoint
from calculate_scale_pixel import get_scale_cm
from model import get_model_instance_segmentation
from utils.parameters import *

folder_path = 'static/images'
os.makedirs(folder_path, exist_ok=True)

def prediction(original_image, model, device, iou_threshold=0.5, score_threshold=0.5, initial_downscale_factor=1.0):
    model.eval()
    downscale_factor = initial_downscale_factor
    downscaled_image = original_image
    fits_in_memory = False
    min_scale, max_scale = 0.1, 1.0

    while not fits_in_memory and max_scale > min_scale:
        try:
            downscale_factor = (min_scale + max_scale) / 2
            new_height, new_width = int(original_image.shape[0] * downscale_factor), int(original_image.shape[1] * downscale_factor)
            downscaled_image = cv2.resize(original_image, (new_width, new_height))
            img_tensor = F.to_tensor(downscaled_image).unsqueeze(0).to(device)
            with torch.no_grad():
                predictions = model(img_tensor)
            fits_in_memory = True 

        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.warning(f"Out of memory at scale {downscale_factor:.2f}. Adjusting scale.")
                torch.cuda.empty_cache()
                max_scale = downscale_factor
            else:
                raise e
                
        finally:
            del img_tensor
            torch.cuda.empty_cache()

    if not fits_in_memory:
        raise RuntimeError("Failed to process the image within memory constraints.")

    #output, output_image = process_predictions(device, original_image, downscaled_image, predictions, downscale_factor, score_threshold, iou_threshold)
    output = process_predictions(device, original_image, downscaled_image, predictions, downscale_factor, score_threshold, iou_threshold)

    #save_prediction_image(original_image, output_image)
    return output

def process_predictions(device, original_image, downscaled_image, predictions, downscale_factor, score_threshold, iou_threshold):
    boxes = predictions[0]['boxes'].cpu()
    labels = predictions[0]['labels'].cpu()
    scores = predictions[0]['scores'].cpu()
    masks = (predictions[0]['masks'] > 0.5).squeeze(1).cpu()  

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
    masks = torch.nn.functional.interpolate(masks.unsqueeze(1).float(), size=original_image.shape[:2], mode="bilinear", align_corners=False).squeeze(1)
    masks = masks > score_threshold  

    scale_detection_counter = 0
    scale_text_recognition_counter = 0

    one_cm_in_pixel, scale_detection_counter, boxes_scale, scale_text_recognition_counter, metrics = get_scale_cm(original_image, downscaled_image, downscale_factor, score_threshold, device, scale_detection_counter, scale_text_recognition_counter)

    output = []  
    #img_tensor = F.to_tensor(original_image).unsqueeze(0)
    #output_image = img_tensor.clone().squeeze(0)
    #font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    #if boxes_scale is not None and len(boxes_scale) > 0:
        #output_image = draw_bounding_boxes(output_image, boxes_scale[0].unsqueeze(0), labels=["scale"], colors=[get_random_color()], width=30, font=font_path, font_size=100)

    for i in range(len(boxes)):
        score = f"{scores[i]:.1f}"
        area = masks[i].sum().item()
        #boolean_mask = masks[i].byte().to(torch.bool)
        #color = get_random_color()
        #label_text = f"{HERBARIUM_CLASSES[labels[i].item()]}: {score}"
        
        pixel_area_in_cm = 0
        if one_cm_in_pixel > 0:
            pixel_area_in_cm = area / (one_cm_in_pixel ** 2)
            pixel_area_in_cm = round(pixel_area_in_cm, 1)
            #label_text = f"{HERBARIUM_CLASSES[labels[i].item()]}: {scores[i]:.2f}, Area: {pixel_area_in_cm:.1f} cm²"

        contours, _ = cv2.findContours(masks[i].cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [contour.flatten().tolist() for contour in contours]

        output.append({
            "boundingBox": boxes[i].tolist(),
            "class": HERBARIUM_CLASSES[labels[i].item()],
            "score": score,
            "areaInPixel": area,
            "one_cm_in_pixel": one_cm_in_pixel,
            "areaInCm2": pixel_area_in_cm,
            "polygon": polygons
        })

        #output_image = draw_bounding_boxes(output_image, boxes[i].unsqueeze(0), labels=[label_text], colors=[color], width=10, font=font_path, font_size=100)
        #output_image = draw_segmentation_masks(output_image, boolean_mask.unsqueeze(0), alpha=0.5, colors=[color])

    #return output, output_image
    return output

def save_prediction_image(original_image, output_image):    
    output_image_pil = T.ToPILImage()(output_image)
    output_image_path = f"{folder_path}/{uuid.uuid4().hex}.png"
    output_image_pil.save(output_image_path)

def predict_organs(image):
    num_classes = len(HERBARIUM_CLASSES)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'plant_organ_model_checkpoint.pth'
    model = get_model_instance_segmentation(num_classes)
    
    if os.path.exists(checkpoint_path):
        model = load_model_checkpoint(checkpoint_path, model, device)
        model.to(device)
        output = prediction(image, model, device)
        return output
    else:
        raise FileNotFoundError("The model checkpoint path is not found")

def process_data(data):
    image_url = data.get("message")
    image_url_response = requests.get(image_url)    
    image = Image.open(BytesIO(image_url_response.content)).convert("RGB")
    image_np = np.array(image)

    output = predict_organs(image_np)
    return {
        "image_url": image_url,
        "image_height": image_np.shape[0],
        "image_width": image_np.shape[1],
        "output": output
    }

def on_message(ws, message):
    try:
        data = json.loads(message)
        if data['message'] is None:
            return

        response_payload = process_data(data)
        requests.post("http://0.0.0.0:8000/processed_message", json=response_payload)

    except requests.RequestException as e:
        logging.error(f"An error occurred during the request: {str(e)}")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {str(e)}")

def on_error(ws, error):
    logging.error(f"WebSocket Error: {error}")

ws = websocket.WebSocketApp("ws://0.0.0.0:8000/ws/new_message", on_message=on_message, on_error=on_error)
ws.run_forever()