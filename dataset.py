import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pywt
import mne
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --- Step 1: Load EEG-MEG Dataset from MNE Sample ---
def load_dataset(data_dir="data"):  # <-- Specifica il percorso desiderato
    dataset = mne.datasets.sample.data_path(path=data_dir, verbose=True)
    raw = mne.io.read_raw_fif(dataset + '/MEG/sample/sample_audvis_raw.fif', preload=True)
    raw.pick_types(meg=True, eeg=True, exclude='bads')
    return raw

# --- Step 2: Apply Continuous Wavelet Transform (CWT) ---
def apply_cwt(signal, wavelet='morl', scales=np.arange(1, 128)):
    coeffs, freqs = pywt.cwt(signal, scales, wavelet)
    return coeffs, freqs

# --- Step 3: Generate CWT Image using PIL ---
def generate_cwt_pil_image(time, freqs, cwtmatr, output_width=224, output_height=224):
    dpi = 100
    figsize = (output_width / dpi, output_height / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    pcm = ax.pcolormesh(time, freqs, np.abs(cwtmatr), shading='auto', cmap='viridis')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.array(canvas.buffer_rgba())[:, :, :3]
    pil_img = Image.fromarray(img)
    plt.close(fig)
    
    return pil_img

# --- Step 4: Test the pipeline ---
def process_and_visualize():
    raw = load_dataset()
    data, times = raw[:2, :1000]  # Get EEG & MEG signals for visualization
    
    signal = data[0]  # Example signal
    coeffs, freqs = apply_cwt(signal)
    
    img = generate_cwt_pil_image(times[:coeffs.shape[1]], freqs, coeffs)
    img.show()

if __name__ == '__main__':
    # Run the pipeline
    # process_and_visualize()
    raw = load_dataset(data_dir="data")
