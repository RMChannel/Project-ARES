# lstm_final.py - COPIA COMPLETA
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class CornerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class ImitationLSTM(nn.Module):
    def __init__(self, input_dim=7):  # 7 feats
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 3), nn.Tanh()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        act = self.fc(out[:, -1, :])
        steer = act[:, 0]
        throttle = (act[:, 1] + 1) / 2
        brake = (act[:, 2] + 1) / 2
        return steer, throttle, brake


def build_sequences(df):
    curve_df = df[df["is_curve"]].copy()
    feats = ["speed_norm", "steer", "brake", "throttle", "norm_pos", "vx", "vy"]
    actions = ["steer", "throttle", "brake"]

    print(f"Step in curve: {len(curve_df)}")
    curve_df[feats] = curve_df[feats].fillna(0)

    X, y = [], []
    values = curve_df[feats].values
    acts = curve_df[actions].values

    for i in range(len(curve_df) - 20):
        X.append(values[i:i + 20])
        y.append(acts[i + 20])

    return np.array(X), np.array(y)


def train():
    df = pd.read_parquet("parquet//kaggle_for_lstm.parquet")
    X, y = build_sequences(df)

    dataset = CornerDataset(X, y)
    loader = DataLoader(dataset, 32, True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImitationLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.MSELoss()

    losses = []
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            steer_p, thr_p, brk_p = model(xb)
            y_pred = torch.stack([steer_p, thr_p, brk_p], 1)
            loss = loss_fn(y_pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(loader))
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={losses[-1]:.4f}")

    torch.save(model.state_dict(), "models/lstm_imitation_trained.pt")
    print("✅ Modello salvato!")

    plt.plot(losses)
    plt.title("LSTM Training")
    plt.savefig("models/loss.png")
    plt.show()


if __name__ == "__main__":
    train()
