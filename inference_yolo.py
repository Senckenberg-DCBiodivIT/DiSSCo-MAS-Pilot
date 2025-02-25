import torch
import cv2
import numpy as np
import os
import uuid
import requests
import logging
import imghdr
import websocket
import json
from ultralytics import YOLO

from torchvision.transforms import functional as F
from torchvision.transforms import v2 as T
from torchvision.ops import nms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from io import BytesIO
from PIL import Image
from utils_helper import get_transform, clear_cuda, load_model_checkpoint
from calculate_scale_pixel import calculate_scale_pixel
from model import get_model_instance_segmentation
from utils.parameters import *
from label_recognition import get_text_labels
from pathlib import Path



model_text = YOLO('models/best_text.pt')
model_plant_organ = YOLO('models/best_plant_organ_segment.pt')
model_scale = YOLO('models/best_scale.pt')


folder_path = 'static/images'
os.makedirs(folder_path, exist_ok=True)

def get_random_color():
    return tuple(map(int, np.random.randint(0, 256, 3)))

def process_predictions(device, image, results):

    scale_counter = 0
    scale_text_recognition_counter = 0
    char_accuracies = []
    overall_class_count = {}
    output = []
    output_1 = []

    
    for idx, result in enumerate(results):
        one_cm_in_pixel = 0
        char_accuracy = 0
        pixel_area = 0
        
        if image is None or image.size == 0:
            print(f"Error: Could not load image {result.path}. Skipping.")
            continue
        
        overlay = image.copy()
        image_name = f"{uuid.uuid4().hex}.png"
        
        img_width = image.shape[1]
        scale_factor = img_width / 1024
        
        #bbox_thickness = max(1, int(5 * scale_factor))
        #font_scale = 0.5 * scale_factor
        #font_thickness = max(1, int(3 * scale_factor))
        image_class_count = {}

        print(f"Running inference for image {image_name} ({idx + 1}/{len(results)})")

        scale_result = model_scale.predict(
            source=image, 
            imgsz=1024,
            save=False, 
            save_txt=False, 
            save_conf=True)
  
        

        for class_id, c in enumerate(scale_result):
            if not c.boxes or c.boxes.conf.tolist()[0] < 0.5:
                continue

            bbox = c.boxes.xyxy.tolist()[0]
            x_min, y_min, x_max, y_max = map(int, bbox)

            cropped_img = image[y_min:y_max, x_min:x_max]
            one_cm_in_pixel, scale_text_recognition_counter, char_accuracy = calculate_scale_pixel(cropped_img, scale_text_recognition_counter)

            """if one_cm_in_pixel > 0:
                print(f"One cm in pixel is {one_cm_in_pixel}")
        
            random_color = get_random_color()
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), random_color, bbox_thickness)

            label = c.names[c.boxes.cls.tolist()[0]]
            confidence = c.boxes.conf.tolist()[0]
            confidence_text = f"{confidence:.2f}"

            label_x = x_min
            label_y = y_min - 10  

            if label_y < 0:
                label_y = y_max + 10  

            cv2.putText(image, f"{label}: {confidence_text}", (label_x, label_y), cv2.FONT_HERSHEY_COMPLEX, font_scale, random_color, font_thickness)

            scale_counter += 1
            print(f"Scale is detected for the image {image_name}")

            if char_accuracy > 0:
                char_accuracies.append(char_accuracy)"""

        text_result = model_text.predict(
            source=image, 
            imgsz=1024,
            save=False, 
            save_txt=False, 
            save_conf=True
        )  

        for class_id, c in enumerate(text_result):
            if not c.boxes or c.boxes.conf.tolist()[0] < 0.5:
                continue

            bbox = c.boxes.xyxy.tolist()[0]
            x_min, y_min, x_max, y_max = map(int, bbox)

            cropped_img = image[y_min:y_max, x_min:x_max]


            detected_labels = get_text_labels(cropped_img)


            """random_color = get_random_color()
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), random_color, bbox_thickness)

            label = c.names[c.boxes.cls.tolist()[0]]
            confidence = c.boxes.conf.tolist()[0]
            confidence_text = f"{confidence:.2f}"

            label_x = x_min
            label_y = y_min - 10  

            if label_y < 0:
                label_y = y_max + 10  

            cv2.putText(image, f"{label}: {confidence_text}", (label_x, label_y), cv2.FONT_HERSHEY_COMPLEX, font_scale, random_color, font_thickness) """

            output.append({
                "boundingBox": bbox,
                "class": c.names[c.boxes.cls.tolist()[0]],
                "score": c.boxes.conf.tolist()[0],
                "detected_labels": detected_labels
            })


        for class_id, c in enumerate(result):
            if not c.masks or not c.boxes:
                print(f"Warning: Empty mask or bounding box for class {class_id} in {image_name}")
                continue

            for i, bbox in enumerate(c.boxes.xyxy.tolist()):
                confidence = c.boxes.conf.tolist()[i]  
                if confidence < 0.5:  
                    continue  

                label = c.names[c.boxes.cls.tolist()[0]]
                """confidence = c.boxes.conf.tolist()[0]
                confidence_text = f"{confidence:.2f}"
                random_color = get_random_color()
                text = f"{label}: {confidence_text}"  

                if label not in image_class_count:
                    image_class_count[label] = 0
                image_class_count[label] += 1

                if label not in overall_class_count:
                    overall_class_count[label] = 0
                overall_class_count[label] += 1"""

            
                mask = c.masks.xy[i] if i < len(c.masks.xy) else None
                if mask is None or not np.any(mask):  
                    continue

                x_min, y_min, x_max, y_max = map(int, bbox)

                #cv2.rectangle(image, (x_min, y_min), (x_max, y_max), random_color, bbox_thickness)

                contour = np.array(mask).astype(np.int32).reshape(-1, 1, 2)
                mask_image = np.zeros_like(image[:, :, 0], dtype=np.uint8)
                #text = f'{label}: {confidence_text}'
                cv2.fillPoly(mask_image, [contour], 255)

                pixel_area = np.sum(mask_image == 255)
                if one_cm_in_pixel > 0:
                    pixel_area_in_cm = pixel_area / (one_cm_in_pixel * one_cm_in_pixel)
                    pixel_area_in_cm = round(pixel_area_in_cm, 1)
                    #text = f'{label}: {confidence_text}, Area: {pixel_area_in_cm:.1f} cm^2'
                    #print(f"The class is {label} and the pixel area is {pixel_area} is {pixel_area_in_cm} cm²")

                """label_x = x_min
                label_y = y_min - 10  
                if label_y < 0:
                    label_y = y_max + 10 
                cv2.putText(image, text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, random_color, font_thickness)
                cv2.drawContours(image, [contour], -1, random_color, thickness=cv2.FILLED) """


                output.append({
                    "boundingBox": bbox,
                    "class": label,
                    "score": confidence,
                    "areaInPixel": int(pixel_area),
                    "one_cm_in_pixel": one_cm_in_pixel,
                    "areaInCm2": pixel_area_in_cm,
                    "polygon": contour.flatten().tolist()
                })
    #with open(f"{image_name}.json", 'w') as json_file:
        #json.dump(output, json_file, indent=4)

    alpha = 0.3
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image) 

    #output_path = Path(folder_path) / f"{image_name}.jpg"
    #cv2.imwrite(str(output_path), image)

    return output

        

def predict_organs(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = model_plant_organ.predict(
            source=image, 
            imgsz=1024,
            save=False, 
            save_txt=False, 
            save_conf=True)  
    return process_predictions(device, image, results)
    

def process_data(image_url):
    try:
        image_url_response = requests.get(image_url, timeout=10)  
        image_bytes = BytesIO(image_url_response.content)
        image_format = imghdr.what(image_bytes)
        if image_format:
            image = Image.open(BytesIO(image_url_response.content)).convert("RGB")
            image_np = np.array(image)
            if isinstance(image_np, np.ndarray) and len(image_np.shape) == 3 and image_np.shape[2] == 3:
                output = predict_organs(image_np)
                return {
                    "image_url": image_url,
                    "image_height": image_np.shape[0],
                    "image_width": image_np.shape[1],
                    "output": output
                }
        else:
            requests.post("http://0.0.0.0:8000/error_message", json={"image_url": image_url, "error" : "The content is not a valid image" })
    except Exception as e:
        print(f"Error: {e}")
        requests.post("http://0.0.0.0:8000/error_message", json={"image_url": image_url, "error" : "The error occured while processing the request"})
        return False 

def on_message(ws, message):
    try:
        data = json.loads(message)
        if data['message'] is None:
            return
        
        image_url = data.get("message")

        response_payload = process_data(image_url)
        if response_payload:
            requests.post("http://0.0.0.0:8000/processed_message", json=response_payload)
    except requests.RequestException as e:
        logging.error(f"An error occurred during the request: {str(e)}")
        requests.post("http://0.0.0.0:8000/error_message", json={"image_url": image_url, "error" : "The error occured while processing the request. Please contact the administrator."})
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {str(e)}")
        requests.post("http://0.0.0.0:8000/error_message", json={"image_url": image_url, "error" : "The error occured while processing the request. Please contact the administrator."})

def on_error(ws, error):
    logging.error(f"WebSocket Error: {error}")

ws = websocket.WebSocketApp("ws://0.0.0.0:8000/ws/new_message", on_message=on_message, on_error=on_error)
ws.run_forever()