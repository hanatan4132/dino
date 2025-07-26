import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
class VAEImagePredictionNet(nn.Module):
    def __init__(self, action_size=3, action_embedding_dim=64, latent_dim=256):
        super(VAEImagePredictionNet, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(512 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(512 * 16 * 16, latent_dim)

        # Action embedding
        self.action_embed = nn.Embedding(action_size, action_embedding_dim)

        # Decoder input
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim + action_embedding_dim, 512 * 16 * 16),
            nn.ReLU()
        )

        # Decoder (shared architecture for both future frame predictions)
        def make_decoder():
            return nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
                
            )

        self.decoder1 = make_decoder()
        self.decoder2 = make_decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, s_t, s_t_plus_1, action):
        x = torch.cat([s_t, s_t_plus_1], dim=1)  # [B, 2, 128, 128]
        features = self.encoder_conv(x)
        flat = self.flatten(features)

        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        z = self.reparameterize(mu, logvar)

        a_embed = self.action_embed(action.squeeze(1))  # [B, action_embedding_dim]
        z_action = torch.cat([z, a_embed], dim=1)

        dec_input = self.decoder_input(z_action)
        dec_input = dec_input.view(-1, 512, 16, 16)

        pred_s_t_plus_2 = self.decoder1(dec_input)
        pred_s_t_plus_3 = self.decoder2(dec_input)

        return pred_s_t_plus_2, pred_s_t_plus_3, mu, logvar


def vae_loss_function(pred1, target1, pred2, target2, mu, logvar):
    recon_loss = nn.functional.mse_loss(pred1, target1, reduction='mean') + \
                 nn.functional.mse_loss(pred2, target2, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss
class DinoDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.s_t = data['s_t']
        self.s_t_plus_1 = data['s_t_plus_1']
        self.a_t = data['a_t']
        self.s_t_plus_2 = data['s_t_plus_2']
        self.s_t_plus_3 = data['s_t_plus_3']
        print(f"Dataset loaded. Number of samples: {len(self.a_t)}")

    def __len__(self):
        return len(self.a_t)

    def __getitem__(self, idx):
        def process_frame(frame):
            if frame.ndim == 3 and frame.shape[2] == 1:
                frame = frame.squeeze(2)
            tensor = torch.from_numpy(frame).unsqueeze(0).float() / 255.0
            return tensor

        s_t = process_frame(self.s_t[idx])
        s_t_plus_1 = process_frame(self.s_t_plus_1[idx])
        s_t_plus_2 = process_frame(self.s_t_plus_2[idx])
        s_t_plus_3 = process_frame(self.s_t_plus_3[idx])

        action = torch.tensor(self.a_t[idx], dtype=torch.long).unsqueeze(0)
        return s_t, s_t_plus_1, action, s_t_plus_2, s_t_plus_3
