import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt

from vae import VAEImagePredictionNet, vae_loss_function, DinoDataset


def visualize_predictions(s_t, s_t_plus_1, s_t_plus_2_true, s_t_plus_3_true, pred_s_t_plus_2, pred_s_t_plus_3, epoch, save_dir):
    s_t = s_t[0].cpu().squeeze().numpy()
    s_t_plus_1 = s_t_plus_1[0].cpu().squeeze().numpy()
    s_t_plus_2_true = s_t_plus_2_true[0].cpu().squeeze().numpy()
    s_t_plus_3_true = s_t_plus_3_true[0].cpu().squeeze().numpy()
    pred_s_t_plus_2 = pred_s_t_plus_2[0].detach().cpu().squeeze().numpy()
    pred_s_t_plus_3 = pred_s_t_plus_3[0].detach().cpu().squeeze().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle(f"Epoch {epoch+1} - Prediction vs Ground Truth")
    
    axes[0, 0].imshow(s_t, cmap='gray')
    axes[0, 0].set_title("s_t")
    axes[0, 1].imshow(s_t_plus_1, cmap='gray')
    axes[0, 1].set_title("s_t+1")
    axes[0, 2].imshow(s_t_plus_2_true, cmap='gray')
    axes[0, 2].set_title("True s_t+2")

    axes[1, 0].imshow(pred_s_t_plus_2, cmap='gray')
    axes[1, 0].set_title("Pred s_t+2")
    axes[1, 1].imshow(s_t_plus_3_true, cmap='gray')
    axes[1, 1].set_title("True s_t+3")
    axes[1, 2].imshow(pred_s_t_plus_3, cmap='gray')
    axes[1, 2].set_title("Pred s_t+3")

    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)
    plt.savefig(os.path.join(save_dir, "samples", f"epoch_{epoch+1}.png"))
    plt.show()
    plt.close()


def train_vae(model, dataloader, optimizer, num_epochs, device, save_dir="vae_models", patience=20):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    train_losses = []
    best_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (s_t, s_t_plus_1, action, s_t_plus_2_true, s_t_plus_3_true) in enumerate(dataloader):
            s_t = s_t.to(device)
            s_t_plus_1 = s_t_plus_1.to(device)
            action = action.to(device)
            s_t_plus_2_true = s_t_plus_2_true.to(device)
            s_t_plus_3_true = s_t_plus_3_true.to(device)

            optimizer.zero_grad()
            pred_s_t_plus_2, pred_s_t_plus_3, mu, logvar = model(s_t, s_t_plus_1, action)
            loss, recon_loss, kl_loss = vae_loss_function(pred_s_t_plus_2, s_t_plus_2_true,
                                                           pred_s_t_plus_3, s_t_plus_3_true,
                                                           mu, logvar)
            beta = min(1.0, epoch / 50)  # 第 50 個 epoch 才變成 1
            loss = recon_loss + beta * kl_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0:
                visualize_predictions(s_t, s_t_plus_1, s_t_plus_2_true, s_t_plus_3_true,
                                      pred_s_t_plus_2, pred_s_t_plus_3, epoch, save_dir)

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_vae_model.pth'))
            print(f"New best model saved with loss: {best_loss:.6f}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    torch.save(model.state_dict(), os.path.join(save_dir, 'final_vae_model.pth'))

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title("VAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.show()


if __name__ == "__main__":
    DATASET_NPZ_PATH = "dino_dataset.npz"
    SAVE_DIR = "vae_models"
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 300
    PATIENCE = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(DATASET_NPZ_PATH):
        print(f"Error: Dataset file not found at '{DATASET_NPZ_PATH}'")
    else:
        dataset = DinoDataset(npz_path=DATASET_NPZ_PATH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        model = VAEImagePredictionNet(action_size=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_vae(model, dataloader, optimizer, NUM_EPOCHS, device, save_dir=SAVE_DIR, patience=PATIENCE)
