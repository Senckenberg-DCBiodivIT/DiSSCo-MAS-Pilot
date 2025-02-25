import cv2
import numpy as np
import pytesseract
import torch
import warnings
import logging

from pytesseract import Output
from torchvision.transforms import functional as F
from Levenshtein import distance as levenshtein_distance

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_text_labels(image, score_threshold=75):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.bilateralFilter(gray_image, 9, 75, 75)
    
    config = '--psm 6 --oem 3'
    result = pytesseract.image_to_data(gray_image, config=config, output_type=Output.DICT)
    
    labels_info = []
    for i in range(len(result['text'])):
        text = result['text'][i].strip()
        conf = result['conf'][i]
        
        if text and conf > score_threshold:
            (x, y, w, h) = (result['left'][i], result['top'][i], result['width'][i], result['height'][i])
            centroid_x = x + w / 2
            labels_info.append({'label': text, 'centroid_x': centroid_x, 'x': x, 'y': y, 'w': w, 'h': h})
    
    
    text = ' '.join([label['label'] for label in labels_info])
    print(text)
    
    return text


#image = cv2.imread("/home/rajapreethi/dev/plant_organ_segmentation_inference/static/images/15507.jpg")
#detected_labels = get_text_labels(image)
