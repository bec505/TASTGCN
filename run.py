import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import model
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="TASTGCN for Traffic Forecasting")
    parser.add_argument("--dataset", default="PEMS08", help="Dataset name")
    parser.add_argument("--num_input", type=int, default=12)
    parser.add_argument("--num_output", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_speed", type=int, default=120)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=12)
    parser.add_argument("--space_emb_dim", type=int, default=3)
    parser.add_argument("--day_emb_dim", type=int, default=32)
    parser.add_argument("--week_emb_dim", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return logging.getLogger(__name__)

def train(model, optimizer, criterion, train_input, train_target, A, S, V, batch_size, device, means, stds):
    model.train()
    losses, maes, rmses, mapes = [], [], [], []
    permutation = torch.randperm(train_input.shape[0])
    for i in range(0, train_input.shape[0], batch_size):
        indices = permutation[i:i + batch_size]
        X_batch = train_input[indices].to(device)
        Y_batch = train_target[indices].to(device)
        optimizer.zero_grad()
        outputs = model(X_batch, A, S, V)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        out = outputs.detach().cpu().numpy() * stds[0] + means[0]
        target = Y_batch.detach().cpu().numpy() * stds[0] + means[0]
        mae, rmse, mape = utils.compute_metrics(out, target)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
    return np.mean(losses), np.mean(maes), np.mean(rmses), np.mean(mapes)


def evaluate(model, input_data, target_data, means, stds, batch_size, device, A, S, V):
    model.eval()
    losses, maes, rmses, mapes = [], [], [], []
    criterion = nn.L1Loss().to(device)
    permutation = torch.randperm(input_data.shape[0])
    for i in range(0, input_data.shape[0], batch_size):
        indices = permutation[i:i + batch_size]
        X_batch = input_data[indices].to(device)
        Y_batch = target_data[indices].to(device)
        outputs = model(X_batch, A, S, V)
        loss = criterion(outputs, Y_batch).item()
        losses.append(loss)
        out = outputs.detach().cpu().numpy() * stds[0] + means[0]
        target = Y_batch.detach().cpu().numpy() * stds[0] + means[0]
        mae, rmse, mape = utils.compute_metrics(out, target)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
    return np.mean(losses), np.mean(maes), np.mean(rmses), np.mean(mapes)


def main():
    args = parse_args()
    log = setup_logging()
    torch.manual_seed(args.seed)
    A, X, V, S, means, stds = utils.load_data(args)
    X = utils.time_index_emb(X)
    # Split dataset
    total_len = X.shape[2]
    train_len = int(total_len * 0.6)
    val_len = int(total_len * 0.8)
    train_data = X[:, :, :train_len]
    val_data = X[:, :, train_len:val_len]
    test_data = X[:, :, val_len:]
    train_input, train_target = utils.generate_dataset(train_data, args.num_input, args.num_output)
    val_input, val_target = utils.generate_dataset(val_data, args.num_input, args.num_output)
    test_input, test_target = utils.generate_dataset(test_data, args.num_input, args.num_output)
    model_instance = model.TASTGCN(args, A.size(0)).to(args.device)
    optimizer = torch.optim.Adam(model_instance.parameters(), lr=args.learning_rate)
    criterion = nn.L1Loss().to(args.device)
    best_mae, best_rmse, best_mape = float("inf"), float("inf"), float("inf")
    for epoch in range(args.epochs):
        train_loss, train_mae, train_rmse, train_mape = train(
            model_instance, optimizer, criterion, train_input, train_target,
            A, S, V, args.batch_size, args.device, means, stds
        )
        val_loss, val_mae, val_rmse, val_mape = evaluate(
            model_instance, val_input, val_target, means, stds, args.batch_size, args.device, A, S, V
        )
        test_loss, test_mae, test_rmse, test_mape = evaluate(
            model_instance, test_input, test_target, means, stds, args.batch_size, args.device, A, S, V
        )
        log.info(f"Epoch {epoch + 1}:")
        log.info(f"  Train      -> MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.2f}%")
        log.info(f"  Validation -> MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.2f}%")
        if test_mae < best_mae:
            best_mae, best_rmse, best_mape = test_mae, test_rmse, test_mape
    log.info(f"Best Test Result     -> MAE: {best_mae:.4f}, RMSE: {best_rmse:.4f}, MAPE: {best_mape:.2f}%")

if __name__ == "__main__":
    main()
