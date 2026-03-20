import torch
import time
from sklearn.metrics import f1_score, recall_score
from models.architectures import LightweightIDSModel
from utils.metrics import calculate_mmd
from data_factory import load_and_align_data

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, tgt_loader = load_and_align_data(window_size=10, batch_size=64)
    
    model = LightweightIDSModel(input_size=42).to(device)
    model.load_state_dict(torch.load("weights/final_model.pth"))
    model.eval()

    all_p, all_l = [], []
    with torch.no_grad():
        for x, y in tgt_loader:
            x = x.to(device)
            out, _ = model(x)
            all_p.extend(torch.argmax(out, 1).cpu().numpy())
            all_l.extend(y.numpy())

    print(f"F1: {f1_score(all_l, all_p)*100:.2f}%")
    
    # 变体鲁棒性测试 (用于雷达图数据)
    correct_adv = 0
    with torch.no_grad():
        for x, y in tgt_loader:
            x_adv = x.to(device) + 0.1 * torch.randn_like(x).to(device)
            out, _ = model(x_adv)
            correct_adv += (torch.argmax(out, 1).cpu() == y).sum().item()
    print(f"Robust Recall: {(correct_adv/len(all_l))*100:.2f}%")

if __name__ == "__main__":
    evaluate()