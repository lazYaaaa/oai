
import requests
import os

origin = "https://www.slavcorpora.ru"
sample_id = "b008ae91-32cf-4d7d-84e4-996144e4edb7"
sample_data = requests.get(f"{origin}/api/samples/{sample_id}").json()
image_paths = [f"{origin}/images/{page['filename']}" for page in sample_data["pages"][:10]]

os.makedirs("src", exist_ok=True)
for url in image_paths:
	filename = url.split("/")[-1]
	r = requests.get(url)
	with open(f"src/{filename}", "wb") as f:
		f.write(r.content)

from PIL import Image

def rgb_to_grayscale(input_path, output_path):
	img = Image.open(input_path).convert('RGB')
	w, h = img.size
	gray = Image.new('L', (w, h))
	pixels = gray.load()
	for y in range(h):
		for x in range(w):
			r, g, b = img.getpixel((x, y))
			y_val = int(0.299 * r + 0.587 * g + 0.114 * b)
			pixels[x, y] = y_val
	gray.save(output_path)

import numpy as np

def nick_binarization(input_path, output_path, window_size=15, k=-0.2):
	img = Image.open(input_path).convert('L')
	arr = np.array(img, dtype=np.float32)
	h, w = arr.shape
	r = window_size // 2
	padded = np.pad(arr, r, mode='reflect')
	result = np.zeros_like(arr, dtype=np.uint8)
	N = window_size * window_size
	for y in range(h):
		for x in range(w):
			window = padded[y:y+window_size, x:x+window_size]
			m = window.mean()
			m2 = (window**2).mean()
			T = m + k * np.sqrt(m2 - m*m)
			result[y, x] = 255 if arr[y, x] > T else 0
	Image.fromarray(result).save(output_path)

os.makedirs('results', exist_ok=True)
for fname in os.listdir('src'):
	if fname.lower().endswith(('.png', '.bmp', '.jpeg', '.jpg')):
		gray_path = f'results/{fname}_gray.bmp'
		rgb_to_grayscale(f'src/{fname}', gray_path)
		bin_path = f'results/{fname}_nick.bmp'
		nick_binarization(gray_path, bin_path)
