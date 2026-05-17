import os
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from pycocotools.coco import COCO

class CatSegmentation:
    def __init__(self, yolo_seg_path='yolov8m-seg.pt', yolo_detect_path='yolov8m.pt', sam_checkpoint='sam_vit_b_01ec64.pth'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加載模型
        self.yolo_seg = YOLO(yolo_seg_path)      # 用於 Part A 的官方評估
        self.yolo_detect = YOLO(yolo_detect_path) # 用於 Part B 的眼睛定位
        
        # 初始化 SAM
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

    # ==========================================
    # Part A: 貓咪輪廓
    # ==========================================
    def calculate_iou(self, mask1, mask2):
        """計算兩張二值化 Mask 的 IoU"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0

    def evaluate_coco_performance(self, image_dir='cat_photos', coco_json='instances_val2017.json'):
        """完整保留你要求的 COCO 評估邏輯"""
        if not os.path.exists(coco_json):
            print("正在下載 COCO 官方標註檔...")
            os.system('wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip')
            os.system('unzip -j annotations_trainval2017.zip annotations/instances_val2017.json')

        coco = COCO(coco_json)
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        results_list = []

        print(f"開始評估 {len(image_files)} 張影像的 COCO IoU...")

        for fname in image_files:
            try:
                img_id = int(fname.split('.')[0])
            except ValueError: continue
                
            img_path = os.path.join(image_dir, fname)
            pred_results = self.yolo_seg.predict(source=img_path, conf=0.3, verbose=False)[0]
            h, w = pred_results.orig_shape

            if pred_results.masks is not None:
                cat_masks = []
                for i, cls in enumerate(pred_results.boxes.cls):
                    if int(cls) == 15: # COCO 貓 ID
                        m = cv2.resize(pred_results.masks.data[i].cpu().numpy(), (w, h))
                        cat_masks.append(m > 0.5)
                pred_mask = np.any(cat_masks, axis=0) if cat_masks else np.zeros((h, w), dtype=bool)
            else:
                pred_mask = np.zeros((h, w), dtype=bool)

            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=[17]) # COCO cat GT ID=17
            anns = coco.loadAnns(ann_ids)
            gt_mask = np.zeros((h, w), dtype=bool)
            for ann in anns:
                gt_mask = np.logical_or(gt_mask, coco.annToMask(ann).astype(bool))

            iou_score = self.calculate_iou(pred_mask, gt_mask)
            results_list.append({'file_name': fname, 'IoU': round(iou_score, 4)})

        df_eval = pd.DataFrame(results_list)
        print(f"\nEvaluation Summary (mIoU: {df_eval['IoU'].mean():.4f})")
        return df_eval

    # ==========================================
    # Part B: 眼睛分割與定位 
    # ==========================================
    def run_eye_refinement(self, img_dir, df_gt, target_visualize=None):
        eval_results = []
        exclude_list = ['00000049810.jpg', '000000411665.jpg', '000000063552.jpg']


        for _, row in df_gt.iterrows():
            fname = row['file_name']
            if fname in exclude_list: continue
            
            image = cv2.imread(os.path.join(img_dir, fname))
            if image is None: continue
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = image.shape[:2]
            
            # Step 1: YOLO 抓貓
            results = self.yolo_detect.predict(source=image, conf=0.3, verbose=False)[0]
            final_masks = []

            if results.boxes is not None:
                self.sam_predictor.set_image(img_rgb)
                for box in results.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    cat_w, cat_h = x2 - x1, y2 - y1
                    
                    # Step 2: Keypoint 定位 (找 Face ROI 內的最暗區塊)
                    face_roi_y_end = y1 + int(cat_h * 0.45)
                    face_roi = image[y1:face_roi_y_end, x1:x2]
                    if face_roi.size == 0: continue
                    
                    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (9, 9), 0)
                    min_val, _, min_loc, _ = cv2.minMaxLoc(blur)
                    
                    # Step 3: SAM Refinement (點提示)
                    seed_x, seed_y = min_loc[0] + x1, min_loc[1] + y1
                    masks, _, _ = self.sam_predictor.predict(
                        point_coords=np.array([[seed_x, seed_y]]),
                        point_labels=np.array([1]),
                        multimask_output=False
                    )
                    
                    # Step 4: 空間約束與過濾
                    face_mask = np.zeros((h_orig, w_orig), dtype=bool)
                    face_mask[y1:face_roi_y_end, x1:x2] = True
                    refined_mask = np.logical_and(masks[0], face_mask)
                    
                    if 50 < np.sum(refined_mask) < (cat_w * cat_h * 0.1):
                        final_masks.append(refined_mask)

            # Step 5: ±25px 命中判定
            img_eval = {'file_name': fname}
            for label in ['c1_left', 'c1_right', 'c2_left', 'c2_right']:
                gt = row.get(label)
                if isinstance(gt, tuple):
                    gx, gy = gt
                    hit = any(np.any(m[max(0, gy-25):min(h_orig, gy+25), max(0, gx-25):min(w_orig, gx+25)]) for m in final_masks)
                    img_eval[f'{label}_hit'] = hit
                else:
                    img_eval[f'{label}_hit'] = np.nan
            eval_results.append(img_eval)

        df_res = pd.DataFrame(eval_results)
        hit_cols = [c for c in df_res.columns if '_hit' in c]
        acc = (df_res[hit_cols].sum().sum() / df_res[hit_cols].notna().sum().sum()) * 100
        print(f"\n🏆 Hybrid 最終命中率: {acc:.2f}%")
        return df_res