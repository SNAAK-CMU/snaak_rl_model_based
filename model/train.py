import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from torch.utils.data import DataLoader, random_split
from load_data import NPZSequenceDataset
from tqdm import tqdm

EPOCHS = 70
BATCH_SIZE = 12
DATA_DIR = "../rl_dataset"
LR = 1e-3
CHECKPOINT_PATH = "model.pth"


if __name__ == "__main__":
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, loss, optimizer
    model = Model().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Dataset + split
    dataset = NPZSequenceDataset(DATA_DIR)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0
        train_loader_iter = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for batch in train_loader_iter:
            rgb, depth, weight, action, next_weight = [b.to(device) for b in batch]
            pred_weights = model(rgb, depth, weight, action)
            loss = criterion(pred_weights, next_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * rgb.size(0)

            # Update progress bar postfix with current loss
            train_loader_iter.set_postfix(loss=loss.item())

            del rgb, depth, weight, action, next_weight, pred_weights, loss
            torch.cuda.empty_cache()

        train_loss /= len(train_dataloader.dataset)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        val_loader_iter = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_loader_iter:
                rgb, depth, weight, action, next_weight = [b.to(device) for b in batch]
                pred_weights = model(rgb, depth, weight, action)
                loss = criterion(pred_weights, next_weight)
                val_loss += loss.item() * rgb.size(0)

                val_loader_iter.set_postfix(loss=loss.item())

                del rgb, depth, weight, action, next_weight, pred_weights, loss
                torch.cuda.empty_cache()

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
            print(f"Saved new best model at epoch {epoch + 1} with val_loss={val_loss:.4f}")

        # Clear cache at the end of the epoch too
        torch.cuda.empty_cache()
