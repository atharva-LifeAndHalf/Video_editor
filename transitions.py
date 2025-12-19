import cv2
import numpy as np 
def crossfade(a, b, n):
    out = []
    for i in range(n):
        alpha = 0.5 - 0.5 * np.cos(np.pi * i / n)
        out.append(cv2.addWeighted(a, 1 - alpha, b, alpha, 0))
    return out


def dissolve(a, b, n):
    out = []
    for i in range(n):
        alpha = i / n
        out.append(cv2.addWeighted(a, 1 - alpha, b, alpha, 0))
    return out


def fade_to_black(a, b, n):
    out = []
    black = np.zeros_like(a)
    half = n // 2
    for i in range(half):
        alpha = i / half
        out.append(cv2.addWeighted(a, 1 - alpha, black, alpha, 0))
    for i in range(half):
        alpha = i / half
        out.append(cv2.addWeighted(black, 1 - alpha, b, alpha, 0))
    return out
