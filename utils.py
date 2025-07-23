import numpy as np
import pandas as pd
import torch

def normalize_adj_matrix(adj_matrix):
    adj_matrix = np.array(adj_matrix)
    non_zero_elements = adj_matrix[adj_matrix > 0]
    if len(non_zero_elements) == 0:
        return adj_matrix
    min_val = non_zero_elements.min()
    max_val = non_zero_elements.max()
    normalized_matrix = np.where(adj_matrix > 0, (adj_matrix - min_val) / (max_val - min_val), 0)
    identity_matrix = np.eye(normalized_matrix.shape[0])
    normalized_matrix = normalized_matrix + identity_matrix
    return normalized_matrix

def get_normalized_adj(A):
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A = np.multiply(np.multiply(diag.reshape((-1, 1)), A), diag.reshape((1, -1)))
    return A

def load_data(args):
    a_path = f"data/{args.dataset}/adj_{args.dataset}_distance.pkl"
    x_path = f"data/{args.dataset}/{args.dataset}.npz"
    A = pd.read_pickle(a_path)
    S = torch.from_numpy(A).to(device=args.device).to(dtype=torch.float32)
    if args.dataset == "PEMS03" or args.dataset == "PEMS07":
        S = S*1000
    X = np.load(x_path)
    A = normalize_adj_matrix(A)
    A = get_normalized_adj(A)
    A = torch.from_numpy(A).to(device=args.device).to(dtype=torch.float32)
    X = X["data"]
    X = X.astype(np.float32)
    X = np.transpose(X, (1, 2, 0))
    num_elements = X.shape[2]
    num_selected = int(0.6 * num_elements)
    selected_data = X[:, :, :num_selected]
    means = np.mean(selected_data, axis=(0, 2))
    stds = np.std(selected_data, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    X = X / stds.reshape(1, -1, 1)
    X = X[:, :1, :]
    V = torch.FloatTensor(A.shape[0], A.shape[0]).uniform_(args.max_speed, args.max_speed).to(args.device)
    return A, X, V, S, means, stds

def time_index_emb(X):
    tempX = torch.tensor(np.transpose(X, (2, 0, 1)))
    cyclic_indices = (torch.arange(tempX.size(0)) % 288).long()
    cyclic_indices_expanded = cyclic_indices.unsqueeze(1).repeat(1, tempX.size(1))
    X = torch.cat((tempX, cyclic_indices_expanded.view(tempX.size(0), tempX.size(1), 1)), dim=2)
    cycle_length = 288
    max_value = 6
    tempX = torch.zeros((X.shape[0], X.shape[1], X.shape[2] + 1))
    tempX[:, :, :-1] = X
    cycle_values = (torch.arange(X.shape[0]) // cycle_length) % (max_value + 1)
    cycle_values = cycle_values.unsqueeze(1).repeat(1, X.shape[1])
    tempX[:, :, -1] = cycle_values
    tempX = np.transpose(tempX, (1, 2, 0))
    tempX = tempX.numpy()
    return tempX

def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])
    return torch.from_numpy(np.array(features)), \
        torch.from_numpy(np.array(target))

def rowNormalization(x):
    means = x.mean(1, keepdim=True).detach()
    x_centered = x - means
    variance = torch.var(x_centered, dim=1, keepdim=True, unbiased=False)
    variance_clamped = torch.clamp(variance, min=0)
    stdev = torch.sqrt(variance_clamped + 1e-5)
    x_normalized = x_centered / stdev
    x = x_normalized
    return x, means, stdev

def get_normalized_flow(X):
    x_flow = X[:, :, :, :1]
    x_flow = x_flow.squeeze(3).permute(0, 2, 1)
    x_flow, means, stdev = rowNormalization(x_flow)
    return x_flow, means, stdev


def concatenate_feature(x_flow_restored, X):
    X = X.permute(0, 3, 1, 2)
    x_flow_restored = x_flow_restored.permute(0, 2, 1).unsqueeze(1)
    x_cat = torch.cat((x_flow_restored, X), dim=1)
    return x_cat.permute(0, 2, 3, 1)

def compute_metrics(pred, target, threshold=5.0, epsilon=1e-5):
    mae = np.mean(np.abs(pred - target))
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    mape = np.abs(pred - target) / (np.abs(target) + epsilon)
    valid_mask = (~np.isnan(mape)) & (~np.isinf(mape)) & (mape <= threshold)
    mape = np.mean(mape[valid_mask]) * 100 if np.any(valid_mask) else 0.0
    return mae, rmse, mape
