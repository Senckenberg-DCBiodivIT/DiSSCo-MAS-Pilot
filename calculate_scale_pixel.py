import cv2
import numpy as np
import pytesseract
import torch
import warnings
import logging
import os

from pytesseract import Output
from model import get_model_instance_segmentation
from PIL import Image
from utils.parameters import *
from torchvision.transforms import functional as F
from torchvision.transforms import v2 as T
from Levenshtein import distance as levenshtein_distance

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.simplefilter(action='ignore', category=FutureWarning)

ground_truth = list(range(0, 10))

def get_scale_cm(original_image, downscaled_image, downscale_factor, score_threshold, device, scale_detection_counter=0, scale_text_recognition_counter=0): 
    checkpoint_path = 'models/scale_model_checkpoint.pth'
    model = get_model_instance_segmentation(num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    img_tensor = F.to_tensor(downscaled_image).unsqueeze(0).to(device)  

    with torch.no_grad():
        prediction = model(img_tensor)  
      
    boxes = prediction[0]['boxes'].cpu()
    labels = prediction[0]['labels'].cpu()
    scores = prediction[0]['scores'].cpu()

    high_score_indices = scores >= score_threshold
    boxes = boxes[high_score_indices]        
    labels = labels[high_score_indices]
    scores = scores[high_score_indices]

    upscale_factor = 1 / downscale_factor
    boxes *= upscale_factor


    if len(boxes) > 0:
        scale_detection_counter += 1
        x_min, y_min, x_max, y_max = boxes[0].int().numpy()            
        cropped_img = original_image[y_min:y_max, x_min:x_max]
        one_cm, scale_text_recognition_counter, metrics = calculate_scale_pixel(cropped_img, scale_text_recognition_counter)
        return one_cm, scale_detection_counter, boxes, scale_text_recognition_counter, metrics
    else:
        print("No scale is detected")


    return 0, scale_detection_counter, None, scale_text_recognition_counter, 0


def rotate_if_vertical(image):  
    try:
        osd = pytesseract.image_to_osd(image)
        rotation = int(osd.split('\n')[2].split(':')[1].strip())
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) if rotation == 90 else image
    except Exception:
        logging.info("OSD failed, using aspect ratio to check orientation")
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) if image.shape[0] > image.shape[1] else image

def filter_inconsistent_digits(digits_info, threshold_ratio=0.2):
    if not digits_info:
        return []

    pixel_distances = []
    consistent_digits = []

    for i in range(len(digits_info) - 1):
        current_digit = int(digits_info[i]['digit'])
        next_digit = int(digits_info[i + 1]['digit'])
        
        if 0 <= current_digit <= 9 and 1 <= next_digit <= 10:
            if digits_info[i+1]['digit'] > digits_info[i]['digit'] and digits_info[i + 1]['centroid_x'] > digits_info[i]['centroid_x']:
                pixel_distance = abs(digits_info[i + 1]['centroid_x'] - digits_info[i]['centroid_x'])
                digit_difference = next_digit - current_digit
                
                adjusted_distance = pixel_distance / digit_difference if digit_difference > 0 else pixel_distance
                pixel_distances.append(adjusted_distance)

    if len(pixel_distances) == 0:
        return []
    
    median_distance = np.median(pixel_distances)

    for i in range(len(digits_info)):
        if i == 0:
            consistent_digits.append(digits_info[i])
        else:
            current_pixel_distance = pixel_distances[i - 1] if i - 1 < len(pixel_distances) else 0

            if abs(current_pixel_distance - median_distance) / median_distance <= threshold_ratio:
                consistent_digits.append(digits_info[i])

    return consistent_digits



def compute_metrics(detected_digits):
    detected_str = ''.join(str(item['digit']) for item in detected_digits)
    ground_truth_str = ''.join(map(str, ground_truth))

    edit_dist = levenshtein_distance(detected_str, ground_truth_str)
    max_len = max(len(detected_str), len(ground_truth_str))
    char_accuracy = (1 - edit_dist / max_len) * 100 if max_len > 0 else 0

    true_chars = list(ground_truth_str)
    pred_chars = list(detected_str)

    return char_accuracy


def calculate_scale_pixel(image, scale_text_recognition_counter):
    if image is None:
        raise FileNotFoundError("Image not found")

    rotated_image = rotate_if_vertical(image)
    gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.bilateralFilter(gray_image, 9, 75, 75)

    config = '--psm 11 --oem 3'
    result = pytesseract.image_to_data(gray_image, config=config, output_type=Output.DICT)

    confidence_threshold = 75
    digits_info = []

    for i in range(len(result['text'])):
        if result['text'][i].isdigit() and result['conf'][i] > confidence_threshold:
            digit = int(result['text'][i])
            if 0 <= digit <= 10:  
                (x, y, w, h) = (result['left'][i], result['top'][i], result['width'][i], result['height'][i])
                centroid_x = x + w / 2
                digits_info.append({'digit': digit, 'centroid_x': centroid_x, 'x': x, 'y': y, 'w': w, 'h': h})

    digits_info.sort(key=lambda k: (int(k['digit']), int(k['y']), int(k['x'])))
    #print(digits_info)
    filtered_digits = filter_inconsistent_digits(digits_info)
    
    def are_consecutive(d1, d2):
        return int(d1) + 1 == int(d2)

    consecutive_digits = []
    sequence = []

    for i in range(len(filtered_digits) - 1):
        if are_consecutive(filtered_digits[i]['digit'], filtered_digits[i + 1]['digit']) and filtered_digits[i]['x'] < filtered_digits[i + 1]['x']:
            if not sequence:
                sequence.append(filtered_digits[i])
            sequence.append(filtered_digits[i + 1])
        else:
            if sequence:
                consecutive_digits.append(sequence)
                sequence = []

    if sequence:
        consecutive_digits.append(sequence)

    pixel_distances = []
    for seq in consecutive_digits:
        for j in range(len(seq) - 1):
            if seq[j + 1]['x'] > seq[j]['x']:
                pixel_distances.append(abs(seq[j + 1]['centroid_x'] - seq[j]['centroid_x']))

    consolidated_pixel_distance = round(np.mean(pixel_distances), 1) if pixel_distances else 0

    if consecutive_digits:
        #print(f"Sequences of consecutive digits found: {consecutive_digits}")
        #print(f"Consolidated pixel distance: {consolidated_pixel_distance}")
        scale_text_recognition_counter += 1
    else:
        logging.info("No valid sequence of consecutive digits found")
        closest_pair = None
        min_distance = float('inf')

        for i in range(len(filtered_digits)):
            for j in range(i + 1, len(filtered_digits)):
                if abs(filtered_digits[i]['y'] - filtered_digits[j]['y']) < 10 and filtered_digits[j]['x'] > filtered_digits[i]['x']:
                    distance = filtered_digits[j]['centroid_x'] - filtered_digits[i]['centroid_x']
                    if distance < min_distance:
                        min_distance = distance
                        closest_pair = (filtered_digits[i], filtered_digits[j])

        if closest_pair:
            smaller_digit, larger_digit = closest_pair

            digit_diff = int(larger_digit['digit']) - int(smaller_digit['digit'])
            consolidated_pixel_distance = round((min_distance / digit_diff), 1) if digit_diff > 0 else 0

            #print(f"Using nearest smaller and larger digit for calculation: {smaller_digit['digit']} -> {larger_digit['digit']}")
            #print(f"Calculated pixel distance for 1 cm: {consolidated_pixel_distance}")
            scale_text_recognition_counter += 1
        else:
            logging.info("No suitable digit pairs found for scale detection")
            consolidated_pixel_distance = 0

    metrics = 0
    if consolidated_pixel_distance > 0 :
        metrics = compute_metrics(filtered_digits)

    return consolidated_pixel_distance, scale_text_recognition_counter, metrics
