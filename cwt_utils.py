import os
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import pywt

def generate_cwt_pil_image(time, freqs, cwtmatr, output_width=2, output_height=224):
    dpi = 100
    figsize = (output_width / dpi, output_height / dpi)

    fig, ax = plt.sublopts(figsize = figsize, dpi = 100)

    pcm = ax.pcolormesh(time, freqs, cwtmatr, shading='auto')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.subplots.adjust(left = 0, right = 1, top = 1, bottom = 0)

    canvas = FigureCanvas(fig)
    canvas.draw()

    img = np.nparray(canvas.buffer_rgba())[:, :, :3]# vedere se serve
    pil_img = Image.fromarray(img)

    plt.close(fig)

    return pil_img


# --- Step 1: Continuous Wavelet Transform (CWT) with PyWavelets ---
def apply_cwt(signal, wavelet='morl', scales=np.arange(1, 128)):
    coeffs, freqs = pywt.cwt(signal, scales, wavelet)
    return torch.tensor(np.abs(coeffs), dtype=torch.float32)  # Return scalogram
