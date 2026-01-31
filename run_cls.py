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

    model = ClsModel().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    
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
    print(f"   L-shell: {sample[0]:.2f}")
    print(f"   cos(MLT): {sample[1]:.2f}")
    print(f"   sin(MLT): {sample[2]:.2f}")
    print(f"   cos(MLAT)^6: {sample[3]:.2f}")
    
    print(f"Raw Model Score (Logit): {output:.4f}")
    print(f"Probability: {probability:.4f}")
    print(f"ECH Wave Presence: {'YES' if has_ech else 'NO'}")

if __name__ == "__main__":
    main()
