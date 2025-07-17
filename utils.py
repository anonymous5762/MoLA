import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def train(model, train_loader, criterion, optimizer, device, is_multilabel):
    model.train()
    total_loss = 0.0
    y_true, y_scores = [], []
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
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

# 验证函数
def validate(model, loader, criterion, device, is_multilabel):
    model.eval()
    total_loss = 0.0
    y_true, y_scores = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
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

def train_SMILES(model, train_loader, criterion, optimizer, device, is_multilabel):
    model.train()
    total_loss = 0.0
    y_true, y_scores = [], []
    for X_batch, y_batch, _ in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        if is_multilabel or outputs.shape[1] == 1:
            target = y_batch.view(-1, outputs.shape[1])
        else:
            target = y_batch.argmax(dim=1)
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

# 验证函数
def validate_SMILES(model, valid_loader, criterion, device, is_multilabel):
    model.eval()
    total_loss = 0.0
    y_true, y_scores = [], []
    with torch.no_grad():
        for X_batch, y_batch, _ in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            if is_multilabel or outputs.shape[1] == 1:
                target = y_batch.view(-1, outputs.shape[1])
            else:
                target = y_batch.argmax(dim=1)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            if is_multilabel:
                probs = torch.sigmoid(outputs)
            else:
                probs = torch.sigmoid(outputs) if outputs.shape[1] == 1 else torch.softmax(outputs, dim=1)
            y_scores.extend(probs.cpu().numpy())
            y_true.extend(target.cpu().numpy())
    prc, roc = compute_auc(y_true, y_scores)
    return total_loss / len(valid_loader), prc, roc

def compute_auc(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    prcs, rocs = [], []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0:
            try:
                prcs.append(average_precision_score(y_true[:, i], y_scores[:, i]))
            except:
                prcs.append(0)
            try:
                rocs.append(roc_auc_score(y_true[:, i], y_scores[:, i]))
            except:
                rocs.append(0)
    return np.mean(prcs), np.mean(rocs)


from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_GNN_Reg(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch).squeeze()
        batch.y = batch.y.squeeze()  # 确保 y 的形状正确
        # print(f"outputs.shape: {outputs.shape}, batch.y.shape: {batch.y.shape}")
        loss = criterion(outputs, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(outputs.cpu().detach().numpy())
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return total_loss / len(train_loader), rmse, mae

def validate_GNN_Reg(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            outputs = model(batch).squeeze()
            loss = criterion(outputs, batch.y)
            total_loss += loss.item()
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(outputs.cpu().detach().numpy())
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return total_loss / len(valid_loader), rmse, mae
