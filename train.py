import torch
import torch.nn as nn
import torch.optim as optim

from models import Encoder, Decoder, LatentDiffusionModel
from cwt_utils import apply_cwt

# --- Step 4: Training Pipeline ---
def train_pipeline(eeg_signals, epochs=10, latent_dim=128):
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    diffusion_model = LatentDiffusionModel(latent_dim)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(diffusion_model.parameters()), lr=0.001)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        for eeg_signal in eeg_signals:
            eeg_scalogram = apply_cwt(eeg_signal).unsqueeze(0).unsqueeze(0)  # Add batch & channel dim
            mu, logvar = encoder(eeg_scalogram)
            latent_code = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            generated_latent = diffusion_model(latent_code)
            reconstructed_scalogram = decoder(generated_latent)
            
            loss = loss_fn(reconstructed_scalogram, eeg_scalogram)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
