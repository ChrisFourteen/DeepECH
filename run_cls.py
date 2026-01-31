import torch
import pickle
import argparse
import numpy as np
from models.transpace_cls import AttenLstmPosSpaceScale as ClsModel

def main():
    parser = argparse.ArgumentParser(description="ECH Classification Inference (Single Sample)")
    parser.add_argument('--ckpt', type=str, default='checkpoints/classification_model.pth', help='Model checkpoint')
    parser.add_argument('--data', type=str, default='data/sample_data.pkl', help='Data file (.pkl)')
    parser.add_argument('--index', type=int, default=0, help='Sample index to infer')
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

    # 3. Inference
    with torch.no_grad():
        input_tensor = torch.tensor(sample, dtype=torch.float).unsqueeze(0).to(device)
        output = model(input_tensor).cpu().item()
        
        # Convert Logit to Probability using Sigmoid function
        probability = 1 / (1 + np.exp(-output))
        
        # Threshold: 0.5 (Based on the manuscript)
        has_ech = probability > 0.5

    # 4. Print Results
    print("-" * 50)
    print(f"ECH Classification Result (Sample Index: {args.index})")
    print("-" * 50)
    
    # Position Input
    print("1. Position Input:")
    print(f"   L-shell: {sample[0]:.2f}")
    print(f"   cos(MLT): {sample[1]:.2f}")
    print(f"   sin(MLT): {sample[2]:.2f}")
    print(f"   cos(MLAT)^6: {sample[3]:.2f}")
    
    # Sequence Input (Summary)
    print("\n2. Sequence Input (Summary):")
    # Sequence 1-4 are used as temporal context (SYM-H and SME indices)
    print(f"   Sequence 1 & 2 (Indices 4-101): Hourly & Minute resolution context.")
    if len(sample) > 102:
        print(f"   Sequence 3 & 4 (Indices 102-223): Additional multi-scale context.")
    
    print("-" * 50)
    print(f"Raw Model Score (Logit): {output:.4f}")
    print(f"Probability: {probability:.4f}")
    print(f"ECH Wave Presence: {'YES' if has_ech else 'NO'}")
    print("-" * 50)

if __name__ == "__main__":
    main()
