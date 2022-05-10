from glob import glob
import os
from utils.custom_functions import generate_excel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--site', type=str, help='site name')
parser.add_argument('--samples', type=str, default=None, help='list of samples ("sampleA sampleB ...")')
parser.add_argument('--model', type=str, help='model name')
parser.add_argument('--length_threshold', type=float, default=0.0, help='minimum length to save to excel')
parser.add_argument('--nocrop', action='store_true', help='do not save cropped images')
parser.add_argument('--second', type=str, default=None, help='path/to/2nd/classifier')
opt = parser.parse_args()

site_name = opt.site
data_yaml_path = './data/ichthyolith_detection.yaml'
crop = not opt.nocrop

if __name__ == '__main__':
    if opt.second is None:
        model_name = opt.model
    else:
        second_model = os.path.basename(os.path.dirname(os.path.dirname(opt.second)))
        model_name = f"{opt.model}__{second_model}"
    print(model_name)

    if opt.samples is None:
        sample_dirs = sorted(glob(f"./runs/detect/{model_name}/{site_name}/*"))
    else:
        sample_names = opt.samples.split()
        sample_dirs = [f"./runs/detect/{model_name}/{site_name}/{sn}" for sn in sample_names]

    print(sample_dirs)
    for sample_dir in sample_dirs:
        if os.path.isdir(sample_dir):
            if os.path.exists(os.path.join(sample_dir, 'detections.xlsx')):
                print(f"skip: {os.path.basename(sample_dir)}")
            else:
                generate_excel(sample_dir, site_name, data_yaml_path, W = 1200, H = 1200, 
                            nms_iou = 0.2, crop_img = crop, l_thresh = opt.length_threshold)
        else:
            if os.path.exists(sample_dir):
                print(f"{sample_dir} is not a directory")
            else:
                print(f"path not exists: {sample_dir}")

    print('Done!')