import numpy as np
import pandas as pd

class EyeMetrologyEvaluator:
    def __init__(self):
        self.target_labels = ['c1_left', 'c1_right', 'c2_left', 'c2_right']

    def calculate_metrics(self, cat_metrics):
        """
        計算單張圖內的量測數值
        """
        results = {
            'cat_count': len(cat_metrics),
            'cat1_dist': np.nan,
            'cat2_dist': np.nan,
            'inter_dist': np.nan,
            'preds': cat_metrics 
        }
        
        if len(cat_metrics) > 0:
            results['cat1_dist'] = cat_metrics[0]['dist']
        if len(cat_metrics) > 1:
            results['cat2_dist'] = cat_metrics[1]['dist']
            # 計算兩隻貓右眼間的距離
            results['inter_dist'] = np.linalg.norm(cat_metrics[0]['right'] - cat_metrics[1]['right'])
            
        return results

    def evaluate_mae(self, pred_results, gt_info):
        """
        與真實值進行比對
        """
        # 妳在 XML 的原始解析內容就在 gt_info 裡
        eval_item = {'E1': np.nan, 'E2': np.nan, 'E_Inter': np.nan}
        preds_for_match = pred_results['preds'].copy()
        matched_preds = []

        # 這裡會用到妳定義的 c1, c2 ID
        for gt_cat in gt_info.get('cats', []):
            if not preds_for_match:
                break
            
            # 找到最靠近標註中心的預測貓
            dists = [np.linalg.norm(p['center'] - gt_cat['center']) for p in preds_for_match]
            best_idx = np.argmin(dists)
            best_pred = preds_for_match.pop(best_idx)
            matched_preds.append(best_pred)

            # 使用妳的 c1/c2 邏輯區分誤差
            err = abs(best_pred['dist'] - gt_cat['dist'])
            if gt_cat['id'] == 'c1':
                eval_item['E1'] = err
            else:
                eval_item['E2'] = err

        # 間距誤差比對
        if len(matched_preds) >= 2 and 'inter_dist' in gt_info:
            current_inter = np.linalg.norm(matched_preds[0]['right'] - matched_preds[1]['right'])
            eval_item['E_Inter'] = abs(current_inter - gt_info['inter_dist'])

        return eval_item

    def print_final_reports(self, preview_df, eval_df):
        """
        產出成果
        """
        print("\n" + "="*80)
        print(f"{'檔案名稱 (已排除雜訊)':<30} | {'數量':<4} | {'貓1眼距':<10} | {'貓2眼距':<10} | {'兩貓右眼距':<10}")
        print("-" * 80)
        for _, r in preview_df.iterrows():
            print(f"{r['File']:<30} | {r['Cats']:<4} | {r['Cat1_Dist']:<10} | {r['Cat2_Dist']:<10} | {r['Inter_Dist']:<10}")
        print("="*80)

        print("\n🎯 距離預測誤差分析 (Evaluation Report)")
        print("="*85)
        print(f"{'檔案名稱':<25} | {'貓1誤差':<10} | {'貓2誤差':<10} | {'兩貓間距誤差':<12}")
        print("-" * 85)
        for _, r in eval_df.iterrows():
            e1 = f"{r['E1']:>8.2f}" if pd.notnull(r['E1']) else f"{'-':>8}"
            e2 = f"{r['E2']:>8.2f}" if pd.notnull(r['E2']) else f"{'-':>8}"
            ei = f"{r['E_Inter']:>10.2f}" if pd.notnull(r['E_Inter']) else f"{'-':>10}"
            print(f"{r['File']:<25} | {e1} | {e2} | {ei}")
        print("="*85)

        cat_errors = eval_df[['E1', 'E2']].values.flatten()
        cat_errors = cat_errors[~np.isnan(cat_errors.astype(float))]
        inter_errors = eval_df['E_Inter'].values
        inter_errors = inter_errors[~np.isnan(inter_errors.astype(float))]

        if len(cat_errors) > 0:
            print(f"\n💡 平均眼距誤差 (MAE-Intra): {np.mean(cat_errors):.2f} 像素")
        if len(inter_errors) > 0:
            print(f"💡 平均間距誤差 (MAE-Inter): {np.mean(inter_errors):.2f} 像素")