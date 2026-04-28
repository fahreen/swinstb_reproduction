"""
Visualization utilities for spectrograms and training results.

Functions:
    show_spectrogram(rgb_array, title=None) -> None
        Display a single 256x256x3 spectrogram with matplotlib.
    
    show_sequence(rgb_sequence, n_frames=5) -> None
        Display a sequence of spectrograms side by side.
    
    plot_framewise_metrics(metrics_dict, output_path=None) -> None
        Plot the per-frame MSE/PSNR/SSIM/LPIPS curves (like paper Fig. 7).
    
    plot_training_curves(log_csv, output_path=None) -> None
        Plot train/val loss over epochs.
"""

# TODO: implement
