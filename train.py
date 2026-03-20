import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_factory import load_and_align_data
from models.architectures import LightweightIDSModel, TSGenerator, DomainDiscriminator

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_loader, tgt_loader = load_and_align_data(window_size=10, batch_size=64)
    
    model = LightweightIDSModel(input_size=42).to(device)
    gen = TSGenerator(input_size=42).to(device)
    disc = DomainDiscriminator(feat_size=128).to(device)
    
    opt_M = optim.Adam(model.parameters(), lr=1e-4)
    opt_G = optim.Adam(gen.parameters(), lr=1e-4)
    opt_D = optim.Adam(disc.parameters(), lr=5e-5)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_gan = nn.BCELoss()

    for epoch in range(1, 71):
        model.train()
        for (src_x, src_y), (tgt_x, _) in zip(src_loader, tgt_loader):
            src_x, src_y, tgt_x = src_x.to(device), src_y.to(device), tgt_x.to(device)
            
            # Update D
            opt_D.zero_grad()
            _, s_f = model(src_x)
            _, t_f = model(gen(tgt_x))
            loss_D = criterion_gan(disc(s_f.detach()), torch.ones(s_f.size(0),1).to(device)) + \
                     criterion_gan(disc(t_f.detach()), torch.zeros(t_f.size(0),1).to(device))
            loss_D.backward()
            opt_D.step()

            # Update M & G
            opt_M.zero_grad(); opt_G.zero_grad()
            logits, _ = model(src_x)
            _, t_f_new = model(gen(tgt_x))
            loss_adv = criterion_gan(disc(t_f_new), torch.ones(t_f_new.size(0),1).to(device))
            (criterion_cls(logits, src_y) + 0.1 * loss_adv).backward()
            opt_M.step(); opt_G.step()
            
        if epoch % 10 == 0: print(f"Epoch {epoch} Done")
    
    torch.save(model.state_dict(), "weights/final_model.pth")

if __name__ == "__main__":
    train()