import sys
sys.path.append('.')
from utils.plots import plot_results
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp_num', help='1 for exp/...')
args = parser.parse_args()


exp_num = int(args.exp_num)
if exp_num == 1:
    exp_num == ''
csv_path = f"./runs/train/exp{exp_num}/results.csv"
plot_results(csv_path)
