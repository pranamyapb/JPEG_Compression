from PIL import Image
import numpy as np
def rgb_to_ycbcr(img):
    # img: HxWx3, dtype float32
    matrix = np.array([
        [ 0.299,    0.587,   0.114],
        [-0.1687,  -0.3313,  0.5],
        [ 0.5,     -0.4187, -0.0813]
    ])
    shift = np.array([0, 128, 128])

    return np.tensordot(img, matrix.T, axes=1) + shift
def ycbcr_to_rgb(img):
    matrix = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.34414, -0.71414],
        [1.0, 1.772, 0.0]
    ])
    shift = np.array([0, -128, -128])
    return np.clip(np.tensordot(img + shift, matrix.T, axes=1), 0, 255)
def downsample(channel):
    return channel[::2, ::2]

def upsample(channel):
    return channel.repeat(2, axis=0).repeat(2, axis=1)


def pad_image(img, block_size=8):
    h, w = img.shape
    new_h = h + (block_size - h % block_size) % block_size
    new_w = w + (block_size - w % block_size) % block_size
    padded = np.zeros((new_h, new_w), dtype=np.float32)
    padded[:h, :w] = img
    return padded

def blockify(img, block_size=8):
    h, w = img.shape
    return img.reshape(h//block_size, block_size, w//block_size, block_size).swapaxes(1,2).reshape(-1, block_size, block_size)

from scipy.fftpack import dct, idct

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

Q50 = np.array([
  [16,11,10,16,24,40,51,61],
  [12,12,14,19,26,58,60,55],
  [14,13,16,24,40,57,69,56],
  [14,17,22,29,51,87,80,62],
  [18,22,37,56,68,109,103,77],
  [24,35,55,64,81,104,113,92],
  [49,64,78,87,103,121,120,101],
  [72,92,95,98,112,100,103,99]
], dtype=np.float32)

def quantize(block, qmatrix):
    return np.round(block / qmatrix)

def dequantize(block, qmatrix):
    return block * qmatrix

def compress(img, qmatrix):
    padded = pad_image(img)
    blocks = blockify(padded)
    dct_blocks = [dct2(block - 128) for block in blocks]  # JPEG shifts range to [-128,127]
    quantized = [quantize(block, qmatrix) for block in dct_blocks]
    return quantized, padded.shape

def decompress(quantized_blocks, shape, qmatrix):
    idct_blocks = [idct2(dequantize(block, qmatrix)) + 128 for block in quantized_blocks]
    h_blocks = shape[0] // 8
    w_blocks = shape[1] // 8
    arr = np.zeros(shape, dtype=np.float32)
    for idx, block in enumerate(idct_blocks):
        i = idx // w_blocks
        j = idx % w_blocks
        arr[i*8:(i+1)*8, j*8:(j+1)*8] = block
    return np.clip(arr, 0, 255).astype(np.uint8)
def save_image(arr, path):
    img = Image.fromarray(arr)
    img.save(path)

def compress_color_image(img_rgb, qmatrix=Q50):
    ycbcr = rgb_to_ycbcr(img_rgb.astype(np.float32))

    Y = ycbcr[:, :, 0]
    Cb = downsample(ycbcr[:, :, 1])
    Cr = downsample(ycbcr[:, :, 2])

    Y_q, Y_shape = compress(Y, qmatrix)
    Cb_q, Cb_shape = compress(Cb, qmatrix)
    Cr_q, Cr_shape = compress(Cr, qmatrix)

    return {
        'Y': Y_q, 'Cb': Cb_q, 'Cr': Cr_q,
        'shapes': (Y_shape, Cb_shape, Cr_shape)
    }
def decompress_color_image(data, qmatrix=Q50):
    Y = decompress(data['Y'], data['shapes'][0], qmatrix)
    Cb = upsample(decompress(data['Cb'], data['shapes'][1], qmatrix))
    Cr = upsample(decompress(data['Cr'], data['shapes'][2], qmatrix))

    ycbcr = np.stack([Y, Cb, Cr], axis=-1)
    return ycbcr_to_rgb(ycbcr).astype(np.uint8)


import os

img = Image.open('input.png').convert('RGB')
img_rgb = np.array(img)

compressed = compress_color_image(img_rgb)
reconstructed = decompress_color_image(compressed)

Image.fromarray(reconstructed).save('output_color.jpg')
import os

original_size = os.path.getsize('input.png')
compressed_size = os.path.getsize('output_color.jpg')
compression_ratio = original_size / compressed_size

print(f"Original size: {original_size / 1024:.2f} KB")
print(f"Compressed size: {compressed_size / 1024:.2f} KB")
print(f"Compression ratio: {compression_ratio:.2f}x")
