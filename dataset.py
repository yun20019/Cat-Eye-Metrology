import os
import requests
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from pycocotools.coco import COCO

class COCODatasetManager:
    def __init__(self, root_dir='./data'):
        self.root_dir = root_dir
        self.ann_dir = os.path.join(root_dir, 'annotations')
        self.img_dir = os.path.join(root_dir, 'val2017')
        self.ann_file = os.path.join(self.ann_dir, 'instances_val2017.json')
        self.coco = None
        os.makedirs(self.root_dir, exist_ok=True)

    def download_annotations(self):
        zip_path = os.path.join(self.root_dir, 'annotations.zip')
        if not os.path.exists(self.ann_file):
            url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            r = requests.get(url)
            with open(zip_path, 'wb') as f:
                f.write(r.content)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.root_dir)
        else:
            print("標籤檔已存在。")

    def load_coco(self):
        if not self.coco:
            self.coco = COCO(self.ann_file)

    def get_multi_cat_samples(self, target_count=13):
        """
        篩選包含兩隻貓的圖片資訊
        """
        self.load_coco()
        catIds = self.coco.getCatIds(catNms=['cat'])
        imgIds = self.coco.getImgIds(catIds=catIds)
        samples = []
        for imgId in imgIds:
            annIds = self.coco.getAnnIds(imgIds=imgId, catIds=catIds)
            if len(annIds) == 2: 
                img_info = self.coco.loadImgs(imgId)[0]
                samples.append({
                    'id': imgId,
                    'file_name': img_info['file_name'],
                    'url': img_info['coco_url']
                })
                if len(samples) >= target_count: 
                    break
        return samples

def get_comprehensive_gt(xml_path):
    """
    解析 XML 中的真實眼睛座標
    """
    if not os.path.exists(xml_path): return {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt_map = {}
    for image in root.findall('image'):
        fname = image.get('name')
        pts = {}
        for p in image.findall('points'):
            label = p.get('label')
            coords = p.get('points').split(',')
            pts[label] = np.array([float(coords[0]), float(coords[1])])
        
        info = {'cats': []}
        if 'c1_left' in pts and 'c1_right' in pts:
            info['cats'].append({'id': 'c1', 'dist': np.linalg.norm(pts['c1_left'] - pts['c1_right']), 'center': (pts['c1_left'] + pts['c1_right'])/2, 'right': pts['c1_right']})
        if 'c2_left' in pts and 'c2_right' in pts:
            info['cats'].append({'id': 'c2', 'dist': np.linalg.norm(pts['c2_left'] - pts['c2_right']), 'center': (pts['c2_left'] + pts['c2_right'])/2, 'right': pts['c2_right']})
        if 'c1_right' in pts and 'c2_right' in pts:
            info['inter_dist'] = np.linalg.norm(pts['c1_right'] - pts['c2_right'])
        gt_map[fname] = info
    return gt_map