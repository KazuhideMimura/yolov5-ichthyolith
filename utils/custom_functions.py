import os
import numpy as np
import pandas as pd
from scipy.stats import hmean
from glob import glob
import cv2
import yaml

from utils.general import xywhn2xyxy, xywh2xyxy, load_second_classname

def get_class_names(class_file_path, print_path = True):
    assert os.path.exists(class_file_path), f"{class_file_path} not exists"
    if print_path:
        print(f"loading class names from {class_file_path}")
    
    # .yaml
    if class_file_path[-5:] == '.yaml':
        with open(class_file_path, errors='ignore') as f:
            class_names = yaml.safe_load(f)['names']  # class names
    else:
        second_model_path = os.path.join(os.path.dirname(class_file_path), 'weights', 'best.pt')
        class_names = load_second_classname(second_model_path)
    return(class_names)

# todo: change parameter name
# data_yaml_path --> class file path
def generate_excel(sample_dir, site_name, data_yaml_path, W = 1200, H = 1200, 
                   nms_iou = None, extention = '.jpg', crop_img = False, 
                   l_thresh = 0, l_factor = 0.93):
    os.makedirs(sample_dir, exist_ok=True)
    sample_name = os.path.basename(sample_dir)
    excel_path = os.path.join(sample_dir, 'detections.xlsx')
        
    df = pd.DataFrame([])
    row = 0
    
    if data_yaml_path is not None:  # data.yaml path (optional)
        class_names = get_class_names(data_yaml_path)  # class names
    else:
        class_names = None
    print('class names: ', class_names)

    results_dirs = [p for p in glob(os.path.join(sample_dir, '*')) if os.path.isdir(p)]
    results_dirs.sort()
    for results_dir in results_dirs:
        print(results_dir)
        txt_paths = sorted(glob(os.path.join(results_dir, 'labels', '*.txt')))
        for path in txt_paths:
            split = os.path.basename(path)[:-4].split('_')
            slide_name = '_'.join(split[:-2])
            slide_num, x_start, y_start = int(split[-3]), int(split[-1]), int(split[-2])
            
            with open(path) as f:
                detections = f.read().split('\n')
                for det in detections:
                    df_temp = pd.DataFrame([], columns=['filename', 'site', 'sample', 'slide', 'slide_no', 'x_start', 'y_start', 
                                                        'x1', 'y1', 'x2', 'y2', 'X1', 'Y1', 'X2', 'Y2', 'duplication_check',
                                                        'class_no', 'class', 'x_center', 'y_center', 'width', 'height', 'confidence', 'note'])
                    if len(det) != 0:
                        df_temp.loc[row, 'filename'] = os.path.basename(path).replace('.txt', extention)
                        df_temp.loc[row, 'site'] = site_name
                        df_temp.loc[row, 'sample'] = sample_name
                        df_temp.loc[row, 'slide_no'] = slide_num
                        df_temp.loc[row, 'slide'] = slide_name                  
                        df_temp.loc[row, ['x_start', 'y_start']] = [x_start, y_start]
                        df_temp.loc[row, 'duplication_check'] = 0
                        df_temp.loc[row, ['class_no', 'x_center', 'y_center', 
                                        'width', 'height', 'confidence']] = det.split()
                        if class_names:
                            df_temp.loc[row, 'class'] = class_names[int(det.split()[0])]
                        xywh = list(map(float, det.split()[1:5]))
                        xywh = np.array([xywh])
                        xyxy = xywhn2xyxy(xywh, w = W, h = H)[0]
                        df_temp.loc[row, ['x1', 'y1', 'x2', 'y2']] = np.round(xyxy).astype(int)
                        X1, X2 = xyxy[0] + x_start, xyxy[2] + x_start
                        Y1, Y2 = xyxy[1] + y_start, xyxy[3] + y_start
                        df_temp.loc[row, ['X1', 'Y1', 'X2', 'Y2']] = [X1, Y1, X2, Y2]
                        row += 1
                    df = pd.concat([df, df_temp])
    
    # predict length
    w, h = np.array(df['width']).astype(float), np.array(df['height']).astype(float)
    l = np.sqrt((w * W) ** 2 + (h * H) ** 2) * l_factor
    df['predicted_length'] = l
    df = df[df['predicted_length'] >= l_thresh]
    
    # format
    for col in ['slide_no', 'class_no', 'duplication_check', 'x1', 'y1', 'x2', 'y2', 'X1', 'Y1', 'X2', 'Y2']:
        df[col] = df[col].astype(int)
    for col in ['x_start', 'y_start']:
        df[col] = df[col].astype(np.int32)
    for col in ['x_center', 'y_center', 'width', 'height', 'confidence']:
        df[col] = df[col].astype(float)
    df = df.sort_values('x_start')
    df = df.sort_values('y_start')
    df = df.sort_values('slide_no')
    df = df.reset_index(drop=True)    
    
    # non-max suppression
    if nms_iou is not None:
        assert all([0 <= nms_iou, nms_iou <= 1]), "nms_iou should be in 0~1"
        df = non_max_suppression(df, nms_iou)
    
    # crop
    if crop_img:
        df = crop_images(df, sample_dir, margin = 30, mag = 3, extention=extention)

    # save
    df.to_excel(excel_path)
    print('saved: ', excel_path)

def non_max_suppression(df, nms_iou):
    slide_names = sorted(set(df['slide']))
    for s in sorted(slide_names):
        dfs = df[df['slide'] == s]
        dfs = dfs.sort_values('confidence', ascending=False)
        while sum(dfs['duplication_check'] == 0) >= 1:
            df0 = dfs[dfs['duplication_check'] == 0]
            first_index = list(df0.index)[0]
            df.loc[first_index, 'duplication_check'] = 1 # checked
            dfs.loc[first_index, 'duplication_check'] = 1
            X1, Y1, X2, Y2 = df0.loc[first_index, ['X1', 'Y1', 'X2', 'Y2']]
            flag_x = np.logical_and(df0['X1'] <= X2, X1 <= df0['X2'])
            flag_y = np.logical_and(df0['Y1'] <= Y2, Y1 <= df0['Y2'])
            flag = np.logical_and(flag_x, flag_y)
            df_possible_duplication = df0[flag]
            for index, item in df_possible_duplication.iterrows():
                if index != first_index:
                    box1, box2 = np.array([X1, Y1, X2, Y2]), np.array(item[['X1', 'Y1','X2', 'Y2']])
                    iou = calc_iou(box1, box2)
                    if iou >= nms_iou:
                        df.loc[index, 'duplication_check'] = 2 # duplicated
                        dfs.loc[index, 'duplication_check'] = 2
    
    # remove duplications
    df = df[df['duplication_check'] == 1]
    df = df.sort_index()
    df = df.reset_index(drop=True) 
    return(df)

def calc_iou(box1, box2, eps = 1e-7):
    """
    modified from function 'bbox_iou' in utils.metrics
    box1, box2: list or numpy.array, [x_min, y_min, x_max, y_max]
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    # Intersection area
    inter =  (min(b1_x2, b2_x2) - max(b1_x1, b2_x1)) * (min(b1_y2, b2_y2) - max(b1_y1, b2_y1))
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    return(iou)

def crop_images(df, sample_dir, margin = 30, mag = 3, 
                extention = '.jpg', save_dir = None):
    use_default_savedir = save_dir is None
    slide_names = set(df['slide'])
    img_path_prev = ''
    for s in sorted(slide_names):
        dfs = df[df['slide'] == s]
        site_name = list(dfs['site'])[0]
        sample_name = list(dfs['sample'])[0]
        # todo: fix img name
        if '1179C_24_3_25' in s:
            s = s.replace('1179C_24_3_25', '1179C_24_03_25')
            sample_name = sample_name.replace('1179C_24_3_25', '1179C_24_03_25')
        if '1179C_25_03_00' in s:
            s = s.replace('1179C_25_03_00', '1179C_25_03_50_00')

        for index, item in dfs.iterrows():
            img_name = item['filename']
            img_path = f"../ichthyolith-slides/{site_name}/{sample_name}/{s}/{img_name}"
            assert os.path.exists(img_path), img_path
            if img_path != img_path_prev:
                img = cv2.imread(img_path)
                img_path_prev = img_path
            x1, y1, x2, y2 = item[['x1', 'y1', 'x2', 'y2']]
            x_start, y_start = item['x_start'], item['y_start']
            # trim
            left, right = max(0, x1 - margin), min(img.shape[1], x2 + margin)
            top, bottom = max(0, y1 - margin), min(img.shape[0], y2 + margin)
            trimmed = img[top:bottom, left:right]
            LEFT, RIGHT = left + x_start, right + x_start # absolute location
            TOP, BOTTOM = top + y_start, bottom + y_start
            # resize
            trimmed = cv2.resize(trimmed, dsize=None, fx = mag, fy = mag)
            # save
            class_name = item['class']
            if use_default_savedir:
                save_dir = os.path.join(sample_dir, 'cropped_images', class_name)
            os.makedirs(save_dir, exist_ok=True)
            save_name = f"{item['slide']}_{TOP}_{LEFT}{extention}"
            df.loc[index, 'trimmed_imgName'] = save_name
            df.loc[index, 'magnification'] = mag
            cv2.imwrite(os.path.join(save_dir, save_name), trimmed)
    return(df)

def calc_prf(tp, fp, fn):
    """
    calculate precision, recall, f1 score
    return -1 when 0 divided by 0
    """
    # precision
    if (tp + fp) > 0:
        p = tp / (tp + fp)
    else:
        p = -1
    # recall
    if (tp + fn) > 0:
        r = tp / (tp + fn)
    else:
        r = -1
    # f1 score
    if min(p, r) >= 0: # both precision and recall are valid
        f = hmean([p, r])
    else:
        f = -1
    return(p, r, f)