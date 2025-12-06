import numpy as np
import torch
import torch.nn as nn

from soft_fbeta_loss import SoftF1Loss, make_weighted_bce_loss

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

def train_model(
    train_loader,
    val_loader,
    input_dim,
    loss_type="soft_f1",
    beta=2.0,
    pos_weight=3.8,
    lr=1e-3,
    epochs=25,
    device=None
):
    from sklearn.metrics import precision_recall_fscore_support

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPClassifier(input_dim).to(device)

    if loss_type == "soft_f1":
        criterion = SoftF1Loss(beta=beta)
    elif loss_type == "bce":
        criterion = make_weighted_bce_loss(pos_weight=pos_weight)
    else:
        raise ValueError("loss_type must be 'soft_f1' or 'bce'")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        bin_preds = (all_preds >= 0.5).astype(int)
        _, _, f1s, _ = precision_recall_fscore_support(
            all_targets, bin_preds, average=None, labels=[0, 1]
        )
        f1_apis = f1s[1]
        mean_train_loss = np.mean(train_losses)

        print(
            f"Epoch {epoch:02d} | train loss = {mean_train_loss:.4f} | "
            f"val F1_Apis = {f1_apis:.4f}"
        )

        if f1_apis > best_val_f1:
            best_val_f1 = f1_apis
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
