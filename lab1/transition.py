import numpy as np
from PIL import Image
import os

def rgb_components(img_path, out_dir):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    r = arr.copy()
    r[..., 1:] = 0
    Image.fromarray(r).save(os.path.join(out_dir, 'R.png'))
    g = arr.copy()
    g[..., [0,2]] = 0
    Image.fromarray(g).save(os.path.join(out_dir, 'G.png'))
    b = arr.copy()
    b[..., :2] = 0
    Image.fromarray(b).save(os.path.join(out_dir, 'B.png'))

def rgb_to_hsi(img_path, out_dir):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255
    R, G, B = arr[...,0], arr[...,1], arr[...,2]
    I = (R + G + B) / 3
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-8
    theta = np.arccos(np.clip(num / den, -1, 1))
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - 3 * min_rgb / (R + G + B + 1e-8)
    hsi = np.stack([H, S, I], axis=-1)
    I_img = (I * 255).astype(np.uint8)
    Image.fromarray(I_img).save(os.path.join(out_dir, 'HSI_I.png'))
    return I_img

def invert_intensity(img_path, out_dir):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255
    R, G, B = arr[...,0], arr[...,1], arr[...,2]
    I = (R + G + B) / 3
    I_inv = 1 - I
    scale = I_inv / (I + 1e-8)
    arr_inv = arr * scale[...,None]
    arr_inv = np.clip(arr_inv * 255, 0, 255).astype(np.uint8)
    Image.fromarray(arr_inv).save(os.path.join(out_dir, 'inverted_intensity.png'))

def stretch(img_path, out_dir, M):
    img = Image.open(img_path)
    arr = np.array(img)
    h, w = arr.shape[:2]
    new_h, new_w = int(h * M), int(w * M)
    stretched = np.zeros((new_h, new_w, arr.shape[2]), dtype=arr.dtype)
    for i in range(new_h):
        for j in range(new_w):
            src_i = i / M
            src_j = j / M
            i0, j0 = int(src_i), int(src_j)
            i1, j1 = min(i0+1, h-1), min(j0+1, w-1)
            di, dj = src_i - i0, src_j - j0
            for c in range(arr.shape[2]):
                val = (arr[i0,j0,c]*(1-di)*(1-dj) + arr[i1,j0,c]*di*(1-dj) + arr[i0,j1,c]*(1-di)*dj + arr[i1,j1,c]*di*dj)
                stretched[i,j,c] = int(val)
    Image.fromarray(stretched).save(os.path.join(out_dir, f'stretched_{M}.png'))
    return stretched

def compress(img_path, out_dir, N):
    img = Image.open(img_path)
    arr = np.array(img)
    h, w = arr.shape[:2]
    new_h, new_w = int(h / N), int(w / N)
    compressed = np.zeros((new_h, new_w, arr.shape[2]), dtype=arr.dtype)
    for i in range(new_h):
        for j in range(new_w):
            block = arr[int(i*N):int((i+1)*N), int(j*N):int((j+1)*N)]
            compressed[i,j] = block.mean(axis=(0,1))
    Image.fromarray(compressed.astype(np.uint8)).save(os.path.join(out_dir, f'compressed_{N}.png'))
    return compressed

def resample_two_pass(img_path, out_dir, M, N):
    stretched = stretch(img_path, out_dir, M)
    h, w = stretched.shape[:2]
    new_h, new_w = int(h / N), int(w / N)
    compressed = np.zeros((new_h, new_w, stretched.shape[2]), dtype=stretched.dtype)
    for i in range(new_h):
        for j in range(new_w):
            block = stretched[int(i*N):int((i+1)*N), int(j*N):int((j+1)*N)]
            compressed[i,j] = block.mean(axis=(0,1))
    Image.fromarray(compressed.astype(np.uint8)).save(os.path.join(out_dir, f'resample_two_pass_{M}_{N}.png'))
    return compressed

def resample_one_pass(img_path, out_dir, K):
    img = Image.open(img_path)
    arr = np.array(img)
    h, w = arr.shape[:2]
    new_h, new_w = int(h * K), int(w * K)
    resampled = np.zeros((new_h, new_w, arr.shape[2]), dtype=arr.dtype)
    scale = 1 / K if K < 1 else K
    for i in range(new_h):
        for j in range(new_w):
            src_i = i / K
            src_j = j / K
            i0, j0 = int(src_i), int(src_j)
            i1, j1 = min(i0+1, h-1), min(j0+1, w-1)
            di, dj = src_i - i0, src_j - j0
            for c in range(arr.shape[2]):
                val = (arr[i0,j0,c]*(1-di)*(1-dj) + arr[i1,j0,c]*di*(1-dj) + arr[i0,j1,c]*(1-di)*dj + arr[i1,j1,c]*di*dj)
                resampled[i,j,c] = int(val)
    Image.fromarray(resampled).save(os.path.join(out_dir, f'resample_one_pass_{K}.png'))
    return resampled

if __name__ == "__main__":
    img_path = "src/image.png"
    out_dir = "results"
    rgb_components(img_path, out_dir)
    rgb_to_hsi(img_path, out_dir)
    invert_intensity(img_path, out_dir)
    M = 4
    N = 3
    K = M / N
    stretch(img_path, out_dir, M)
    compress(img_path, out_dir, N)
    resample_two_pass(img_path, out_dir, M, N)
    resample_one_pass(img_path, out_dir, K)
