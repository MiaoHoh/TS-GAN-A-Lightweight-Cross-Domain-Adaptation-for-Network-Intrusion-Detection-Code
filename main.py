import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
from scipy.linalg import sqrtm
from thop import profile
from sklearn.metrics import f1_score, accuracy_score, recall_score
from data_factory import load_and_align_data
from models import LightweightIDSModel, TSGenerator, DomainDiscriminator

def calculate_mmd(source, target):
    def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    batch_size = int(source.size(0))
    kernels = gaussian_kernel(source, target)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    return torch.mean(XX + YY - XY - YX)

def calculate_fid(real_feat, gen_feat):
    mu1, sigma1 = real_feat.mean(axis=0), np.cov(real_feat, rowvar=False)
    mu2, sigma2 = gen_feat.mean(axis=0), np.cov(gen_feat, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean): covmean = covmean.real
    return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "weights"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    BATCH_SIZE = 64
    EPOCHS = 70
    INPUT_DIM = 42
    
    src_loader, tgt_loader = load_and_align_data(window_size=10, batch_size=BATCH_SIZE)

    model = LightweightIDSModel(input_size=INPUT_DIM).to(device)
    generator = TSGenerator(input_size=INPUT_DIM).to(device)
    discriminator = DomainDiscriminator(feat_size=128).to(device)

    dummy_input = torch.randn(1, 10, 42).to(device)
    flops, params = profile(model, inputs=(dummy_input, ), verbose=False)
    print(f"Params: {params/1e6:.2f}M, FLOPs: {flops/1e9:.4f}G")

    opt_M = optim.Adam(model.parameters(), lr=1e-4)
    opt_G = optim.Adam(generator.parameters(), lr=1e-4)
    opt_D = optim.Adam(discriminator.parameters(), lr=5e-5)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_gan = nn.BCELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for (src_data, src_label), (tgt_data, _) in zip(src_loader, tgt_loader):
            src_data, src_label = src_data.to(device), src_label.to(device)
            tgt_data = tgt_data.to(device)

            opt_D.zero_grad()
            _, s_feat = model(src_data)
            t_gen_data = generator(tgt_data)
            _, t_feat = model(t_gen_data)
            d_real = discriminator(s_feat.detach())
            d_fake = discriminator(t_feat.detach())
            loss_D = criterion_gan(d_real, torch.ones_like(d_real)) + criterion_gan(d_fake, torch.zeros_like(d_fake))
            loss_D.backward()
            opt_D.step()

            opt_M.zero_grad()
            opt_G.zero_grad()
            logits, _ = model(src_data)
            loss_cls = criterion_cls(logits, src_label)
            _, t_feat_new = model(generator(tgt_data))
            d_fake_adv = discriminator(t_feat_new)
            loss_adv = criterion_gan(d_fake_adv, torch.ones_like(d_fake_adv))
            (loss_cls + 0.1 * loss_adv).backward()
            opt_M.step()
            opt_G.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{70}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, label in tgt_loader:
            data = data.to(device)
            logits, _ = model(data)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(label.numpy())
    
    f1 = f1_score(all_labels, all_preds) * 100
    acc = accuracy_score(all_labels, all_preds) * 100
    rec = recall_score(all_labels, all_preds) * 100
    
    with torch.no_grad():
        _, s_feat_fin = model(src_data)
        _, t_feat_fin = model(generator(tgt_data))
        mmd_dist = calculate_mmd(s_feat_fin, t_feat_fin)
        fid_val = calculate_fid(s_feat_fin.cpu().numpy(), t_feat_fin.cpu().numpy())

    dummy_single = torch.randn(1, 10, 42).to(device)
    for _ in range(100): _ = model(dummy_single)
    t_start = time.perf_counter()
    for _ in range(1000): _ = model(dummy_single)
    latency = (time.perf_counter() - t_start)
    qps = 1000 / latency
    ei = (f1 * qps) / 10000

    correct_adv = 0
    with torch.no_grad():
        for data, label in tgt_loader:
            data = data.to(device)
            data_adv = data + 0.1 * torch.randn_like(data)
            logits, _ = model(data_adv)
            correct_adv += (torch.argmax(logits, dim=1).cpu() == label).sum().item()
    rob_recall = (correct_adv / len(all_labels)) * 100

    print("-" * 30)
    print(f"Accuracy: {acc:.2f}%")
    print(f"Recall: {rec:.2f}%")
    print(f"F1: {f1:.2f}%")
    print(f"MMD: {mmd_dist:.4f}")
    print(f"FID: {fid_val:.2f}")
    print(f"Latency: {latency:.2f}ms")
    print(f"QPS: {qps:.0f}")
    print(f"EI: {ei:.2f}")
    print(f"Robust Recall: {rob_recall:.2f}%")
    print(f"Decay: {rec - rob_recall:.2f}%")
    print("-" * 30)

    torch.save(model.state_dict(), os.path.join(save_dir, "ts_gan_final.pth"))

if __name__ == "__main__":
    main()