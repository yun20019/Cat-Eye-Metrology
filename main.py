import pandas as pd
from dataset import parse_xml_annotations 
from segmentation import CatSegmentation
from measurement import EyeMetrologyEvaluator

def main():
    # 1. 初始化路徑與工具
    image_dir = './cat_photos'
    xml_path = './data/annotations.xml'
    
    seg_tool = CatSegmentation(
        yolo_seg_path='weights/yolov8m-seg.pt',
        yolo_detect_path='weights/yolov8m.pt',
        sam_checkpoint='weights/sam_vit_b_01ec64.pth'
    )
    evaluator = EyeMetrologyEvaluator()
    df_gt = parse_xml_annotations(xml_path)

    # --- Task 1: 貓咪輪廓分割 ---
    df_iou = seg_tool.evaluate_coco_performance(image_dir)

    # --- Task 2: 眼睛分割 ---
    df_hits = seg_tool.run_eye_refinement(image_dir, df_gt)
    hit_cols = [c for c in df_hits.columns if '_hit' in c]
    total_hit_rate = df_hits[hit_cols].mean().mean()
    
    print(f"\n[Hit Rate Results]")
    print(f"Overall Eye Localization Hit Rate: {total_hit_rate:.2%}\n")

    # --- Task 3&4: 眼睛距離 ---
    eval_df = evaluator.evaluate_mae(df_hits, df_gt)
    evaluator.print_final_reports(df_hits, eval_df)

if __name__ == "__main__":
    main()