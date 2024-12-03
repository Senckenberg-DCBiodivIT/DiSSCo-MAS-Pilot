HERBARIUM_CLASSES = ('background','leaf', 'flower', 'fruit', 'seed', 'stem', 'root','scale')
HERBARIUM_CLASSES_LIST = ['background','leaf', 'flower', 'fruit', 'seed', 'stem', 'root','scale']
num_classes = len(HERBARIUM_CLASSES)
HRBParis_imgDir = 'dataset/HerbarParis/scans/'
HRBParis_annoDir = 'dataset/HerbarParis/annotations/'
HRBParis_json_annoDir = 'dataset/HerbarParis/json_annotations/'
HRBParis_maskDir = 'dataset/HerbarParis/masks/'
HRBParis_urls = 'dataset/HerbarParis/mnhn_urls.csv'
HRBParis_testDir = 'dataset/HerbarParis/test/'
HRBParis_Dir = 'dataset/HerbarParis/'

HRBFR_imgDir = 'dataset/HerbarFR/scans/'
HRBFR_annoDir = 'dataset/HerbarFR/annotations/'
HerbarFR_json_annoDir = 'dataset/HerbarFR/json_annotations/'
HRBFR_maskDir = 'dataset/HerbarFR/masks'
HRBFR_Dir = 'dataset/HerbarFR/'

Scale_Dir = 'dataset/ScaleTraining/'

Scale_imgDir = 'dataset/ScaleTraining/scans/'
Scale_annoDir = 'dataset/ScaleTraining/annotations/'
Scale_maskDir = 'dataset/ScaleTraining/masks'


extracted_Dir = 'dataset/Dataset/'
extracted_hrbDir = 'dataset/Dataset/Annotations/'
extracted_detections = 'dataset/Dataset/Detections/'

HRBFR_detectionsDir = 'detections/FR_Detections/'

BOUNDING_BOX_COLORS = [(0, 0, 255), (128, 0, 0), (255, 0, 255),
                       (255, 255, 0), (0, 128, 0), (128, 128, 128)]

BOUNDING_BOX_COLORS_new = [
    (0, 0, 255),     # Blue
    (255, 69, 0),    # OrangeRed
    (138, 43, 226),  # BlueViolet
    (60, 179, 113),  # MediumSeaGreen
    (255, 20, 147),  # DeepPink
    (75, 0, 130),    # Indigo
    (255, 0, 0),     # Red
    (255, 140, 0),   # DarkOrange
    (30, 144, 255),  # DodgerBlue
    (0, 191, 255)    # DeepSkyBlue
]

HEIGHT = 300 * 4
WIDTH = 200 * 4

TOTAL_HRBParis_urls = 653