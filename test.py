import os
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import torch
import cv2
import numpy as np
import inference
import inference_yolo

from ultralytics import YOLO
from collections import defaultdict
from torchvision import tv_tensors
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.ops import nms
from PIL import Image
from torchvision.transforms.v2 import functional as F
from torch.cuda.amp import autocast, GradScaler
from utils.parameters import *
from PIL import Image
from label_recognition import get_text_labels




folder_path  = "test_image/scans"
inference_model = "yolo"
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.tif')):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        if inference_model == "yolo":
            inference_yolo.predict_organs(image)
        else:
            inference.predict_organs(image)

    
    