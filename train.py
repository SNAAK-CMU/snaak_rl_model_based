import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from torch.utils.data import DataLoader, random_split
from load_data import NPZSequenceDataset

EPOCHS = 100
BATCH_SIZE = 100
DATA_DIR = "dataset"
LR = 1e-3
CHECKPOINT_PATH = "best_model.pth"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, loss, optimizer
    model = Model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Dataset + split
    dataset = NPZSequenceDataset(DATA_DIR)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            rgb, depth, weight, action, next_weight = [b.to(device) for b in batch]

            pred_weights = model(rgb, depth, weight, action)

            loss = criterion(pred_weights, next_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * rgb.size(0)

        train_loss /= len(train_dataloader.dataset)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                rgb, depth, weight, action, next_weight = [b.to(device) for b in batch]
                pred_weights = model(rgb, depth, weight, action)
                loss = criterion(pred_weights, next_weight)
                val_loss += loss.item() * rgb.size(0)

        val_loss /= len(val_dataloader.dataset)

        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ---------------- CHECKPOINT ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, CHECKPOINT_PATH)
            print(f"Saved new best model at epoch {epoch+1} with val_loss={val_loss:.4f}")
