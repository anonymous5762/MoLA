import deepchem as dc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import time

from model_MoLA import MoLA

from utils import compute_auc

import warnings
warnings.filterwarnings('ignore')

classification_loaders = {
    "bace_classification": dc.molnet.load_bace_classification,
    "bbbp": dc.molnet.load_bbbp,
    "clintox": dc.molnet.load_clintox,
    "muv": dc.molnet.load_muv,
    "sider": dc.molnet.load_sider,
    "tox21": dc.molnet.load_tox21,
}

import h5py
def load_Molformer(output_dir_path):
    path = os.path.join(output_dir_path, 'Molformer_Emb_2025.h5')
    if not os.path.exists(path):
        print(f"Output file not found: {path}")
        return 0, 0, 0
    
    with h5py.File(path, "r") as f:
        train_fp = f["train_fp"][:]   # [:] 读为 numpy array
        valid_fp = f["valid_fp"][:]
        test_fp = f["test_fp"][:]
    return train_fp, valid_fp, test_fp

# 提取 FingerPrints 和 SMILES
def extract_fp_and_smiles(dataset):
    fingerprints = []
    smiles_list = []

    for mol in dataset.ids:  # dataset.ids 通常是 SMILES
        mol_obj = Chem.MolFromSmiles(mol)
        if mol_obj is None:
            print(f"Invalid SMILES: {mol}")
            continue

        # 提取 FingerPrint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol_obj, radius=2, nBits=2048)
        fingerprints.append(fp)

        # 存储原始 SMILES
        smiles_list.append(mol)

    return fingerprints, smiles_list

# 构建字符到索引的映射
def build_vocab(smiles_list):
    unique_chars = sorted(set("".join(smiles_list)))  # 排序确保顺序一致
    vocab = {char: idx + 1 for idx, char in enumerate(unique_chars)}
    vocab["<pad>"] = 0
    return vocab

def prepare_data(dataset, fingerprints, fp_molformer, smiles, vocab, max_sm_len=100):
    data_list = []
    for i, (graph, y, _) in enumerate(zip(dataset.X, dataset.y, dataset.w)):
        x = torch.tensor(graph.node_features, dtype=torch.float32)
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
        fp = torch.tensor(fingerprints[i], dtype=torch.float32)
        fp2 = torch.tensor(fp_molformer[i], dtype=torch.float32)
        sm_idx = [vocab.get(char, 0) for char in smiles[i]]
        sm = torch.tensor(sm_idx[:max_sm_len], dtype=torch.long)
        sm = torch.cat([sm, torch.zeros(max_sm_len - len(sm), dtype=torch.long)])
        y = torch.tensor(y, dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, fp=fp.unsqueeze(0), fp2=fp2.unsqueeze(0), sm=sm.unsqueeze(0), y=y)
        data_list.append(data)
    return data_list


def train(model, train_loader, criterion, optimizer, device, is_multilabel):
    model.train()
    total_loss = 0.0
    y_true, y_scores = [], []
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        outputs = outputs[-1]  # 获取最后一层输出
        if is_multilabel or outputs.shape[1] == 1:
            target = batch.y.view(-1, outputs.shape[1])
        else:
            target = batch.y.argmax(dim=1)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if is_multilabel:
            probs = torch.sigmoid(outputs)
        else:
            probs = torch.sigmoid(outputs) if outputs.shape[1] == 1 else torch.softmax(outputs, dim=1)
        y_scores.extend(probs.cpu().detach().numpy())
        y_true.extend(target.cpu().detach().numpy())
    prc, roc = compute_auc(y_true, y_scores)
    return total_loss / len(train_loader), prc, roc

def validate(model, loader, criterion, device, is_multilabel):
    model.eval()
    total_loss = 0.0
    y_true, y_scores = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            outputs = outputs[-1]  # 获取最后一层输出
            if is_multilabel or outputs.shape[1] == 1:
                target = batch.y.view(-1, outputs.shape[1])
            else:
                target = batch.y.argmax(dim=1)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            if is_multilabel:
                probs = torch.sigmoid(outputs)
            else:
                probs = torch.sigmoid(outputs) if outputs.shape[1] == 1 else torch.softmax(outputs, dim=1)
            y_scores.extend(probs.cpu().numpy())
            y_true.extend(target.cpu().numpy())
    prc, roc = compute_auc(y_true, y_scores)
    return total_loss / len(loader), prc, roc

def validate_mm(model, loader, criterion, device, is_multilabel, missing_modalities=None):
    """
    missing_modalities: list of strings, e.g. ['fp'], ['fp2','sm'], or None
    取值必须跟 Data 对象里的字段名一致：'x','edge_index','fp','fp2','sm'
    """
    model.eval()
    total_loss = 0.0
    y_true, y_scores = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # —— 在这里把指定模态置零 —— 
            if missing_modalities:
                for mod in missing_modalities:
                    if hasattr(batch, mod):
                        tensor = getattr(batch, mod)
                        # 保持形状不变，全置零
                        zeroed = torch.zeros_like(tensor)
                        setattr(batch, mod, zeroed)

            outputs = model(batch)
            outputs = outputs[-1]  # 获取最后一层输出

            # 构造 target
            if is_multilabel or outputs.shape[1] == 1:
                target = batch.y.view(-1, outputs.shape[1])
            else:
                target = batch.y.argmax(dim=1)

            loss = criterion(outputs, target)
            total_loss += loss.item()

            # 计算概率
            if is_multilabel:
                probs = torch.sigmoid(outputs)
            else:
                if outputs.shape[1] == 1:
                    probs = torch.sigmoid(outputs)
                else:
                    probs = torch.softmax(outputs, dim=1)

            y_scores.extend(probs.cpu().numpy())
            y_true.extend(target.cpu().numpy())

    prc, roc = compute_auc(y_true, y_scores)
    return total_loss / len(loader), prc, roc


# 主实验逻辑
def run_experiment(loader_func, model_save_path, epochs=10, batch_size=32, embed_dim=128, max_len=100, dataset_name="", device='cpu'):
    torch.cuda.empty_cache()
    tasks, datasets, transformers = loader_func(featurizer=dc.feat.MolGraphConvFeaturizer(), splitter="random", splitter_seed=2025,\
                                                reload=True, data_dir="./datasets/raw/", save_dir="./datasets/featurized/")
    train_dataset, valid_dataset, test_dataset = datasets
    output_dir_path = os.path.join("./datasets/MHG_OUTPUT", dataset_name)
    train_fp_molformer, valid_fp_molformer, test_fp_molformer = load_Molformer(output_dir_path)
    train_fp, train_smiles = extract_fp_and_smiles(train_dataset)
    valid_fp, valid_smiles = extract_fp_and_smiles(valid_dataset)
    test_fp, test_smiles = extract_fp_and_smiles(test_dataset)
    smiles_vocab = build_vocab(train_smiles + valid_smiles + test_smiles)
    train_data = prepare_data(train_dataset, train_fp, train_fp_molformer, train_smiles, smiles_vocab)
    valid_data = prepare_data(valid_dataset, valid_fp, valid_fp_molformer, valid_smiles, smiles_vocab)
    test_data = prepare_data(test_dataset, test_fp, test_fp_molformer, test_smiles, smiles_vocab)
    num_classes = train_dataset.y.shape[1]
    is_multilabel = num_classes > 1 and (train_dataset.y.sum(axis=1) > 1).any()
    out_size = num_classes if is_multilabel else (1 if num_classes == 1 else num_classes)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = MoLA(
        graph_dim=train_data[0].x.size(1),
        fp_dim=train_data[0].fp.size(1),
        sm_vocab_size=len(smiles_vocab),
        hidden_dim=embed_dim,
        output_dim=out_size,
        num_layers=4
    ).to(device)

    criterion = nn.BCEWithLogitsLoss() if is_multilabel or out_size == 1 else nn.CrossEntropyLoss()
    print(f"Output Size: {out_size}, Is Multilabel: {is_multilabel}")
    print(f"Output Size: {out_size}, Is Multilabel: {is_multilabel}", file=file)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    patience = 10
    num_not_improve_max = 30
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=patience, verbose=True)

    best_val_auc = 0
    num_not_improve = 0

    for epoch in range(epochs):
        time_start = time.time()
        train_loss, train_prc, train_roc = train(model, train_loader, criterion, optimizer, device, is_multilabel)
        valid_loss, valid_prc, valid_roc = validate(model, valid_loader, criterion, device, is_multilabel)
        test_loss, test_prc, test_roc = validate(model, test_loader, criterion, device, is_multilabel)

        train_auc = train_prc if "muv" in dataset_name else train_roc
        valid_auc = valid_prc if "muv" in dataset_name else valid_roc
        test_auc = test_prc if "muv" in dataset_name else test_roc
        time_cost = time.time() - time_start
        
        # 动态调整学习率
        scheduler.step(valid_auc)
        current_lr = optimizer.param_groups[0]['lr']

        if valid_auc > best_val_auc:
            best_val_auc = valid_auc
            num_not_improve = 0
            torch.save(model, model_save_path)
        else:
            num_not_improve += 1
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Auc: {train_auc:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, "
              f"Valid Auc: {valid_auc:.4f}, Best Valid Auc: {best_val_auc:.4f}, "
              f"Test Auc: {test_auc:.4f}, Learning Rate: {current_lr:.6f}, Cost Time: {time_cost:.4f}")
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Auc: {train_auc:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, "
              f"Valid Auc: {valid_auc:.4f}, Best Valid Auc: {best_val_auc:.4f}, "
              f"Test Auc: {test_auc:.4f}, Learning Rate: {current_lr:.6f}, Cost Time: {time_cost:.4f}", file=file)
        if num_not_improve >= num_not_improve_max:
            break

    model = torch.load(model_save_path)
    test_loss, test_prc, test_roc = validate(model, test_loader, criterion, device, is_multilabel)
    test_auc = test_prc if "muv" in dataset_name else test_roc

    print(f"Test Loss: {test_loss:.4f}, Test Auc: {test_auc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Auc: {test_auc:.4f}", file=file)


# 示例运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
debug = False
run_times = 10
model_name = "MoLA_Test"

if __name__ == "__main__":
    result_path = "./result_Cla/"+model_name
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for i in range(run_times):
        file_name = os.path.join(result_path, "result_{}_{}.txt".format(model_name, i+1))
        print(file_name)
        if os.path.exists(file_name) and not debug:
            print("文件已存在！")
            continue
        
        file = open(file_name, "w")
        file.close()
    
        for dataset_name, loader in classification_loaders.items():
            file = open(file_name, "a+")
            print(f"\nRunning {model_name} on {dataset_name}")
            print(f"\nDataset: {dataset_name}", file=file)
            model_save_path = f"./weights/{model_name}_{dataset_name}_{str(i+1)}.pth"
            run_experiment(loader, model_save_path, epochs=300, batch_size=256, embed_dim=256, dataset_name=dataset_name, device=device)
            file.close()
