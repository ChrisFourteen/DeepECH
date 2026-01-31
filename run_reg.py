import torch
import pickle
import argparse
import numpy as np
from models.transpace import AttenLstmPosSpaceScale as RegModel

def main():
    parser = argparse.ArgumentParser(description="ECH Regression Inference (Single Sample)")
    parser.add_argument('--ckpt', type=str, default='checkpoints/regression_model.pth', help='Model checkpoint')
    parser.add_argument('--data', type=str, default='data/sample_data.pkl', help='Data file (.pkl)')
    parser.add_argument('--index', type=int, default=60, help='Sample index to infer')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    model = RegModel().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # 2. 加载数据并提取单个样本
    with open(args.data, 'rb') as f:
        data = pickle.load(f)

    # 兼容 ndarray 或 dict 格式
    if isinstance(data, dict):
        sample = list(data.values())[args.index]
    else:
        sample = data[args.index]

    # 3. 推理
    with torch.no_grad():
        input_tensor = torch.tensor(sample, dtype=torch.float).unsqueeze(0).to(device)
        intensity = model(input_tensor).cpu().item()

    # 4. Print Results
    print("-" * 50)
    print(f"ECH Regression Result (Sample Index: {args.index})")
    print("-" * 50)

    # Position Input
    print("1. Position Input:")
    print(f"   L-shell: {sample[0]:.2f}")
    print(f"   cos(MLT): {sample[1]:.2f}")
    print(f"   sin(MLT): {sample[2]:.2f}")
    print(f"   cos(MLAT)^6: {sample[3]:.2f}")

    # The regression network outputs the predicted wave amplitude in dB
    print(f"Predicted ECH Wave Intensity: {intensity:.4f} dB")
    print("-" * 50)

if __name__ == "__main__":
    main()
