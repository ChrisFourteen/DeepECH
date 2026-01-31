import torch
import pickle
import argparse
import numpy as np
from models.transpace_cls import AttenLstmPosSpaceScale as ClsModel

def main():
    parser = argparse.ArgumentParser(description="ECH Classification Inference (Single Sample)")
    parser.add_argument('--ckpt', type=str, default='checkpoints/classification_model.pth', help='Model checkpoint')
    parser.add_argument('--data', type=str, default='data/sample_classification.pkl', help='Data file (.pkl)')
    parser.add_argument('--index', type=int, default=60, help='Sample index to infer')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    model = ClsModel().to(device)
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
        prob = model(input_tensor).cpu().item()
        has_ech = prob > 0.5

    # 4. 打印结果
    print("-" * 50)
    print(f"ECH Classification Result (Sample Index: {args.index})")
    print("-" * 50)
    
    # Position Input (前4位)
    print("1. Position Input:")
    print(f"   [0] L-shell: {sample[0]:.2f}")
    print(f"   [1] cos(MLT): {sample[1]:.2f}")
    print(f"   [2] sin(MLT): {sample[2]:.2f}")
    print(f"   [3] cos(MLAT)^6: {sample[3]:.2f}")
    
    # Sequence Input (后续序列)
    print("\n2. Sequence Input:")

    seq1 = sample[4:53]
    seq2 = sample[53:102]
    print(f"   Sequence 1 (Indices 4-52, Length {len(seq1)}):")
    print(f"      Mean: {np.mean(seq1):.4f}, Std: {np.std(seq1):.4f}")
    print(f"      First 5 values: {seq1[:5]}")
    print(f"   Sequence 2 (Indices 53-101, Length {len(seq2)}):")
    print(f"      Mean: {np.mean(seq2):.4f}, Std: {np.std(seq2):.4f}")
    print(f"      First 5 values: {seq2[:5]}")
    
    print("-" * 50)
    print(f"Raw Probability (Logit): {prob:.4f}")
    print(f"ECH Wave Presence: {'YES' if has_ech else 'NO'}")
    print("-" * 50)

if __name__ == "__main__":
    main()
