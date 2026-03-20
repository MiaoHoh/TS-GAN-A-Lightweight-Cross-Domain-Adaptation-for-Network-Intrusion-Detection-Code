import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


UNSW_PATH = r'C:\Users\Administrator\Desktop\返修意见\大论文\data\UNSW-NB15'
CIC_PATH = r'C:\Users\Administrator\Desktop\返修意见\大论文\data\CIC-IDS2017'

FEATURE_MAPPING = {
    # --- (Basic Flow Features) ---
    'dur': 'Flow Duration',
    'sbytes': 'Total Fwd Packets',
    'dbytes': 'Total Backward Packets',
    'sttl': 'Fwd Header Length',
    'dttl': 'Bwd Header Length',
    'sloss': 'Fwd Packets/s',
    'dloss': 'Bwd Packets/s',
    'Sload': 'Flow Packets/s',
    'Dload': 'Flow Bytes/s',
    'spkts': 'Total Length of Fwd Packets',
    'dpkts': 'Total Length of Bwd Packets',
    'swit': 'Fwd IAT Total',
    'dwit': 'Bwd IAT Total',
    
    # ---  (Content & Traffic Stats) ---
    'stcpb': 'Fwd PSH Flags',
    'dtcpb': 'Bwd PSH Flags',
    'smeansz': 'Fwd Packet Length Mean',
    'dmeansz': 'Bwd Packet Length Mean',
    'trans_depth': 'Fwd IAT Mean',
    'res_bdy_len': 'Bwd IAT Mean',
    'sjit': 'Fwd IAT Std',
    'djit': 'Bwd IAT Std',
    'sinpkt': 'Flow IAT Mean',
    'dinpkt': 'Flow IAT Std',
    'tcprtt': 'Flow IAT Max',
    'synack': 'Flow IAT Min',
    'ackdat': 'Fwd IAT Max',
    
    # ---  (Time-based & Count Features) ---
    'is_sm_ips_ports': 'Fwd IAT Min',
    'ct_state_ttl': 'Bwd IAT Max',
    'ct_flw_http_mthd': 'Bwd IAT Min',
    'is_ftp_login': 'Fwd Header Length.1', 
    'ct_ftp_cmd': 'Fwd Packets/s',
    'ct_srv_src': 'Bwd Packets/s',
    'ct_srv_dst': 'Min Packet Length',
    'ct_dst_ltm': 'Max Packet Length',
    'ct_src_ltm': 'Packet Length Mean',
    'ct_src_dport_ltm': 'Packet Length Std',
    'ct_dst_sport_ltm': 'Packet Length Variance',
    'ct_dst_src_ltm': 'Average Packet Size',
    'ct_flw_http_mthd': 'Avg Fwd Segment Size',
    'ct_ftp_cmd': 'Avg Bwd Segment Size',
    'ct_srv_src': 'Subflow Fwd Packets',
    'ct_srv_dst': 'Subflow Bwd Packets'
}
def load_and_align_data(window_size=10, batch_size=64):
    print("📂 正在从桌面路径读取源域与目标域数据...")
    
    cic.replace([np.inf, -np.inf], np.nan, inplace=True)
    cic.fillna(0, inplace=True)
    

    unsw = pd.read_csv(os.path.join(UNSW_PATH, 'UNSW_NB15_training_set.csv'))
    cic = pd.read_csv(os.path.join(CIC_PATH, 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'))

    unsw_cols = list(FEATURE_MAPPING.keys())
    cic_cols = list(FEATURE_MAPPING.values())

    X_src = unsw[unsw_cols].copy()
    X_tgt = cic[cic_cols].copy()
    X_tgt.columns = unsw_cols 


    y_src = unsw['label'].values 
    y_tgt = cic[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1).values


    X_src_scaled = scaler.fit_transform(X_src)
    X_tgt_scaled = scaler.transform(X_tgt) # 

 
    def to_sequences(data, labels, window):
        x, y = [], []
        for i in range(len(data) - window):
            x.append(data[i:i+window])
            y.append(labels[i+window])
        return torch.FloatTensor(np.array(x)), torch.LongTensor(np.array(y))

    print("⏳ 正在进行时序窗口切片 (Window Size = 10)...")
    src_x, src_y = to_sequences(X_src_scaled, y_src, window_size)
    tgt_x, tgt_y = to_sequences(X_tgt_scaled, y_tgt, window_size)


    src_loader = DataLoader(TensorDataset(src_x, src_y), batch_size=batch_size, shuffle=True)
    tgt_loader = DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=batch_size, shuffle=True)

    print(f"✅ 数据对齐完成！源域样本: {len(src_x)}, 目标域样本: {len(tgt_x)}")
    return src_loader, tgt_loader