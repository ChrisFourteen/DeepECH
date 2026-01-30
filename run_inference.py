import torch
import pickle
import os
from tqdm import tqdm
import numpy as np
import scipy.io
import argparse
from models.transpace_cls import AttenLstmPosSpaceScale as ClsModel
from models.transpace import AttenLstmPosSpaceScale as RegModel

def calculate_values(L_combi, MLT_combi):
    """计算位置特征向量"""
    cos_MLT = np.cos((MLT_combi / 12) * np.pi)
    sin_MLT = np.sin((MLT_combi / 12) * np.pi)
    cos_MLAT_power_6 = np.cos(0) ** 6  # 示例中固定为 0 度
    return L_combi, cos_MLT, sin_MLT, cos_MLAT_power_6

def replace_value(values, base_data):
    """将计算出的位置特征替换到基础数据的特征向量前4位"""
    tmp = base_data.copy()
    tmp[:4] = values
    return tmp

def main():
    parser = argparse.ArgumentParser(description="Space Physics Inference (Regression & Classification)")
    parser.add_argument('--reg_ckpt', type=str, default='checkpoints/regression_model.pth', help='Regression model checkpoint')
    parser.add_argument('--cls_ckpt', type=str, default='checkpoints/classification_model.pth', help='Classification model checkpoint')
    parser.add_argument('--reg_data', type=str, default='data/sample_regression.pkl', help='Sample data for regression')
    parser.add_argument('--cls_data', type=str, default='data/sample_classification.pkl', help='Sample data for classification')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")

    # 1. 加载数据
    print("Loading sample data...")
    with open(args.reg_data, 'rb') as f:
        reg_data_dict = pickle.load(f)
    with open(args.cls_data, 'rb') as f:
        cls_data_dict = pickle.load(f)

    # 选取示例索引 (参考 infer_dif.py)
    reg_base = reg_data_dict[119]
    cls_base = cls_data_dict[119]

    # 2. 加载模型
    print("Loading models...")
    reg_model = RegModel().to(device)
    reg_model.load_state_dict(torch.load(args.reg_ckpt, map_location=device))
    reg_model.eval()

    cls_model = ClsModel().to(device)
    cls_model.load_state_dict(torch.load(args.cls_ckpt, map_location=device))
    cls_model.eval()

    # 3. 构造网格数据进行推理
    L_values = np.arange(1.2, 6.1, 0.1)
    MLT_values = np.arange(0, 24.1, 0.1)
    
    reg_matrix = np.zeros((len(L_values), len(MLT_values)))
    cls_matrix = np.zeros((len(L_values), len(MLT_values)))

    print("Running inference grid...")
    with torch.no_grad():
        for i, L in enumerate(tqdm(L_values, desc="L-shell")):
            for j, MLT in enumerate(MLT_values):
                pos_vals = calculate_values(L, MLT)
                
                # 回归推理
                reg_input = torch.tensor(replace_value(pos_vals, reg_base), dtype=torch.float).unsqueeze(0).to(device)
                reg_matrix[i][j] = reg_model(reg_input).cpu().item()

                # 分类推理
                cls_input = torch.tensor(replace_value(pos_vals, cls_base), dtype=torch.float).unsqueeze(0).to(device)
                cls_prob = cls_model(cls_input).cpu().item()
                cls_matrix[i][j] = 1 if cls_prob > 0.5 else 0

    # 4. 保存结果
    reg_out = os.path.join(args.output_dir, 'regression_grid.mat')
    cls_out = os.path.join(args.output_dir, 'classification_grid.mat')
    scipy.io.savemat(reg_out, mdict={'data': reg_matrix})
    scipy.io.savemat(cls_out, mdict={'data': cls_matrix})
    
    print(f"Inference complete.\nResults saved to:\n - {reg_out}\n - {cls_out}")

if __name__ == "__main__":
    main()
