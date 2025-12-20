import cv2
import numpy as np

def apply_filter(frame, ftype):
    """Apply visual filters based on user selection"""
    if ftype == "none":
        return frame

    f = frame.copy()

    # --- Your Existing Filters ---
    if ftype == "cinematic":
        f = cv2.convertScaleAbs(f, alpha=1.2, beta=-15)
        f[:, :, 0] = np.clip(f[:, :, 0] * 0.9, 0, 255)
        f[:, :, 2] = np.clip(f[:, :, 2] * 1.1, 0, 255)
    elif ftype == "vintage":
        kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
        f = cv2.transform(f, kernel)
    
    # --- New Styles from your list ---
    elif ftype == "horror":
        # Desaturate and add a cold blue/green tint
        f = cv2.convertScaleAbs(f, alpha=0.8, beta=-20)
        f[:, :, 2] = np.clip(f[:, :, 2] * 0.8, 0, 255) # Reduce Red
    elif ftype == "product":
        # High sharpness and vivid colors
        f = cv2.detailEnhance(f, sigma_s=10, sigma_r=0.15)
        f = cv2.convertScaleAbs(f, alpha=1.1, beta=10)
    elif ftype == "vlog":
        # Warm, skin-tone friendly brightness
        f = cv2.convertScaleAbs(f, alpha=1.05, beta=5)
        f[:, :, 2] = np.clip(f[:, :, 2] * 1.05, 0, 255) # Boost Red/Warmth
    elif ftype == "documentary":
        # Naturalistic, slightly flat contrast
        f = cv2.convertScaleAbs(f, alpha=0.9, beta=0)
    elif ftype == "fast_reel":
        # High contrast and slight motion blur simulation
        f = cv2.convertScaleAbs(f, alpha=1.3, beta=-10)

    return np.clip(f, 0, 255).astype(np.uint8)
