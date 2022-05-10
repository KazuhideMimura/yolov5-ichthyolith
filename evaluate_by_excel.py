import os
import numpy as np
import pandas as pd
import argparse

from utils.mrcnn_utils import compute_matches
from utils.custom_functions import get_class_names, calc_prf, crop_images

data_yaml_path = './data/ichthyolith_detection.yaml'

def evaluate(gt_excel_path, pred_excel_path, save_excel_path,
             iou_threshold, model_name, ignore_classes = ['noise'], 
             score_threshold = 0.5, crop = True):
    df_gt = pd.read_excel(gt_excel_path)
    df_pred = pd.read_excel(pred_excel_path)
    # todo thresholding by size and score

    df_performance = pd.DataFrame([])
    sn1 = list(set(df_gt['slide']))
    sn2 = list(set(df_pred['slide']))
    slide_names = sorted(list(set(sn1 + sn2)))   
    class_names = get_class_names(data_yaml_path)
    class_ids = list(range(len(class_names)))
    if crop:
        sample_dir = os.path.dirname(pred_excel_path) # for 
    index = 0

    for cid in class_ids:
        c_name = class_names[cid]
        df_performance.loc[index, 'class'] = c_name
        if c_name not in ignore_classes:
            df_gt_cid = df_gt[df_gt['class_no'] == cid]
            df_pred_cid = df_pred[df_pred['class_no'] == cid]
            TP_c, FP_c, FN_c = 0, 0, 0
            for s in slide_names:
                df_gt_cid_s = df_gt_cid[df_gt_cid['slide'] == s]
                df_pred_cid_s = df_pred_cid[df_pred_cid['slide'] == s]
                gt_boxes = np.array(df_gt_cid_s.loc[:, ['Y1', 'X1', 'Y2', 'X2']])
                gt_class_ids = np.array(df_gt_cid_s.loc[:, 'class_no'])
                pred_boxes = np.array(df_pred_cid_s.loc[:, ['Y1', 'X1', 'Y2', 'X2']])
                pred_class_ids = np.array(df_pred_cid_s.loc[:, 'class_no'])
                pred_scores = np.array(df_pred_cid_s.loc[:, 'confidence'])
                gt_match, pred_match, _ = compute_matches(gt_boxes, gt_class_ids,
                        pred_boxes, pred_class_ids, pred_scores,
                        iou_threshold=iou_threshold, 
                        score_threshold = score_threshold)
                gt_matched = gt_match > -1
                pred_matched = pred_match > -1
                
                TP_c += np.sum(gt_matched)
                FP_c += np.sum(pred_match == -1)
                FN_c += np.sum(gt_match == -1)
                # save to gt and detection excel
                for i, x in enumerate(list(gt_matched)):
                    gt_index = list(df_gt_cid_s.index)[i]
                    df_gt.loc[gt_index, model_name] = x
                for i, x in enumerate(list(pred_matched)):
                    pred_index = list(df_pred_cid_s.index)[i]
                    df_pred.loc[pred_index, 'correct'] = x
            p, r, f = calc_prf(TP_c, FP_c, FN_c)
            if f == -1:
                print(f"CAUTION: divided by zero for class {c_name}")
            # write to excel
            df_performance.loc[index, 'score threshold'] = score_threshold
            for i, x in enumerate([TP_c, FP_c, FN_c]):
                param_name = ['TP', 'FP', 'FN'][i]
                df_performance.loc[index, param_name] = x
            for i, x in enumerate([p, r, f]):
                param_name = ['precision', 'recall', 'f1 score'][i]
                df_performance.loc[index, param_name] = x

            # crop image
            if crop:
                df_gt_cid_s['matched'] = gt_matched
                df_pred_cid_s['matched'] = pred_matched
                # tp
                df_pred_cid_s_tp = df_pred_cid_s[df_pred_cid_s['matched'] == 1]
                save_img_dir = os.path.join(sample_dir, 'TPs', c_name)
                crop_images(df_pred_cid_s_tp, sample_dir = '', margin = 30, 
                            mag = 3, extention = '.jpg', save_dir = save_img_dir)
                # fp
                df_pred_cid_s_fp = df_pred_cid_s[df_pred_cid_s['matched'] == 0]
                save_img_dir = os.path.join(sample_dir, 'FPs', c_name)
                crop_images(df_pred_cid_s_fp, sample_dir = '', margin = 30, 
                            mag = 3, extention = '.jpg', save_dir = save_img_dir)
                # fn
                df_gt_cid_s_fn = df_gt_cid_s[df_gt_cid_s['matched'] == 0]
                save_img_dir = os.path.join(sample_dir, 'FNs', c_name)
                crop_images(df_gt_cid_s_fn, sample_dir = '', margin = 30, 
                            mag = 3, extention = '.jpg', save_dir = save_img_dir)

        index += 1
        
    # total performance
    df_performance.loc[index, 'score threshold'] = score_threshold
    df_performance.loc[index, 'class'] = '(total)'
    TP = np.sum(df_performance.loc[index - len(class_names):index, 'TP'])
    FP = np.sum(df_performance.loc[index - len(class_names):index, 'FP'])
    FN = np.sum(df_performance.loc[index - len(class_names):index, 'FN'])
    for i, x in enumerate([TP, FP, FN]):
        param_name = ['TP', 'FP', 'FN'][i]
        df_performance.loc[index, param_name] = x
    p, r, f = calc_prf(TP, FP, FN)
    for i, x in enumerate([p, r, f]):
        param_name = ['precision', 'recall', 'f1 score'][i]
        df_performance.loc[index, param_name] = x
    index += 1
    df_performance.loc[index, 'class'] = ''
    index += 1
    
    # save excel
    df_performance.to_excel(save_excel_path)
    print(f"excel saved: {save_excel_path}")   

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=str, help='site name')
    parser.add_argument('--sample', type=str, help='sample id')
    parser.add_argument('--model', type=str, help='sample id')
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--second', type=str, default=None, help='path/to/2nd/classifier')
    opt = parser.parse_args()

    if opt.second is None:
        model_name = opt.model
    else:
        second_model = os.path.basename(os.path.dirname(os.path.dirname(opt.second)))
        model_name = f"{opt.model}__{second_model}"

    gt_excel_path = f"./data/GTs/{opt.sample}_gts.xlsx"
    pred_excel_path = f"./runs/detect/{model_name}/{opt.site}/{opt.sample}/detections.xlsx"
    for p in [gt_excel_path, pred_excel_path]:
        assert os.path.exists(p), f"path not exists: {p}"
    save_excel_path = f"./runs/detect/{model_name}/{opt.site}/{opt.sample}/performance.xlsx"
    if os.path.exists(save_excel_path):
        assert input('Already evaluated. Try again? (y for yes) ') == 'y'
    # main
    evaluate(gt_excel_path, pred_excel_path, save_excel_path,
             opt.iou_threshold, model_name)