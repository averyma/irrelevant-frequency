import torch
# import cv2
import numpy as np
from scipy import signal
import ipdb

def rgb2gray(rgb_input):
    """
        reference: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    """
    # batch operation
    if len(rgb_input.shape) == 4:
        gray_output = rgb_input[:, 0, :, :]*0.299 + \
                        rgb_input[:, 1, :, :]*0.587 + \
                        rgb_input[:, 2, :, :]*0.114
    # single image operation
    elif len(rgb_input.shape) == 3:
        gray_output = rgb_input[0, :, :]*0.299 + \
                        rgb_input[1, :, :]*0.587 + \
                        rgb_input[2, :, :]*0.114
 
    else:
        raise  NotImplementedError("Input dimension not supported. Check tensor shape!")
 
    return gray_output

def getDCTmatrix(size):
    """
        Computed using C_{jk}^{N} found in the following link:
        https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
        Verified with cv2.dct(), error less than 1.1260e-06.
        
        output: DCT matrix with shape (size,size)
    """
    dct_dir = '/h/ama/workspace/irrelevant-frequency/dct_matrix/{}.pt'
    dct_matrix = torch.zeros([size, size])
    if size in [784, 224, 28, 32]:
        dct_matrix = torch.load(dct_dir.format(size))
    else:
        for i in range(0, size):
            for j in range(0, size):
                if j == 0:
                    dct_matrix[i, j] = np.sqrt(1/size)*np.cos(np.pi*(2*i+1)*j/2/size)
                else:
                    dct_matrix[i, j] = np.sqrt(2/size)*np.cos(np.pi*(2*i+1)*j/2/size)

    return dct_matrix

def dct(input_tensor):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    size = input_tensor.shape[0]

    dct_matrix = getDCTmatrix(size).to(input_tensor.device)
    dct_output = torch.mm(dct_matrix.transpose(0, 1), input_tensor)

    return dct_output

def batch_dct(input_tensor, dct_matrix):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    m = input_tensor.shape[0]
    d = input_tensor.shape[1]

    dct_matrix = dct_matrix.to(input_tensor.device).expand(m,-1,-1)
    dct_output = torch.bmm(dct_matrix.transpose(1, 2), input_tensor.view(m,d,1)).squeeze()
    
    return dct_output

def batch_idct(input_tensor, dct_matrix):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    m = input_tensor.shape[0]
    d = input_tensor.shape[1]

    idct_matrix = torch.inverse(dct_matrix).to(input_tensor.device).expand(m,-1,-1)
    idct_output = torch.bmm(idct_matrix.transpose(1, 2), input_tensor.view(m,d,1)).squeeze()
    
    return idct_output

def idct(input_tensor):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    size = input_tensor.shape[0]
    dct_matrix = getDCTmatrix(size).to(input_tensor.device)
    idct_matrix = torch.inverse(dct_matrix)
    idct_output = torch.mm(idct_matrix.transpose(0, 1), input_tensor)

    return idct_output

def dct2(input_tensor):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
        
        Note that this operation is performed on a single image
        input shape: (size,size)
        output shape: (size, size)
    """ 
    
    size = input_tensor.shape[0]
    dct_matrix = getDCTmatrix(size).to(input_tensor.device)
    dct_output = torch.mm(torch.mm(dct_matrix.transpose(0, 1), input_tensor),dct_matrix)

    return dct_output

def idct2(input_tensor):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
        
        Note that this operation is performed on a single image
        input shape: (size,size)
        output shape: (size, size)
    """ 
    size = input_tensor.shape[0]
    idct_matrix = torch.inverse(getDCTmatrix(size)).to(input_tensor.device)
    idct_output = torch.mm(torch.mm(idct_matrix.transpose(0, 1), input_tensor.squeeze()),idct_matrix)

    return idct_output

def batch_dct2(input_tensor, dct_matrix):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    batch_size = input_tensor.shape[0]
    d = input_tensor.shape[2]

    dct_matrix = dct_matrix.to(input_tensor.device).expand(batch_size,-1,-1)
    dct2_output = torch.bmm(torch.bmm(dct_matrix.transpose(1, 2), input_tensor.view(batch_size,d,d)), dct_matrix)
    
    return dct2_output

def batch_idct2(input_tensor, dct_matrix):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """ 
    batch_size = input_tensor.shape[0]
    d = input_tensor.shape[2]

    idct_matrix = torch.inverse(dct_matrix).to(input_tensor.device).expand(batch_size,-1,-1)
    idct2_output = torch.bmm(torch.bmm(idct_matrix.transpose(1, 2), input_tensor.view(batch_size,d,d)), idct_matrix)
    
    return idct2_output

def batch_dct2_3channel(inverse, input_tensor, dct_matrix):
    """
        reference: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    """
    # make sure batch input and 3chennels
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    assert len(input_tensor.shape) == 4, "Input tensor must be of shape (batch, 3, height, width)"

    batch_size, channels, height, width = input_tensor.shape
    output = torch.zeros_like(input_tensor, device=input_tensor.device)
    d = input_tensor.shape[2]
    if inverse:
        matrix = torch.inverse(dct_matrix).to(input_tensor.device).expand(batch_size, -1, -1)
    else:
        matrix = dct_matrix.to(input_tensor.device).expand(batch_size, -1, -1)

    for i in range(3):
        output[:, i, :, :] = torch.bmm(torch.bmm(matrix.transpose(1, 2), input_tensor[:,i,:,:].view(batch_size,d,d)), matrix)

    # Reshape back to the original tensor format
    if batch_size == 1:
        output = output.view(channels, height, width)
    else:
        output = output.view(batch_size, channels, height, width)

    return output

def batch_dct2_3channel_optimized(inverse, input_tensor, dct_matrix):
    # make sure batch input and 3chennels
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    assert len(input_tensor.shape) == 4, "Input tensor must be of shape (batch, 3, height, width), now: {}".format(input_tensor.shape)

    # No need to expand the DCT matrix; PyTorch's broadcasting handles it
    if inverse:
        matrix = torch.inverse(dct_matrix).to(input_tensor.device)
    else:
        matrix = dct_matrix.to(input_tensor.device)

    # Reshape input tensor to combine batch and channel dimensions for batch processing
    batch_size, channels, height, width = input_tensor.shape
    input_reshaped = input_tensor.view(batch_size * channels, height, width)

    # Apply DCT or inverse DCT in a batched manner
    transformed = torch.bmm(matrix @ input_reshaped, matrix.unsqueeze(0).expand(batch_size * channels, -1, -1))

    # Reshape back to the original tensor format
    if batch_size == 1:
        output = transformed.view(channels, height, width)
    else:
        output = transformed.view(batch_size, channels, height, width)

    return output

def filter_based_on_freq(batch_input, dct_matrix, mask):

    batch_dct = batch_dct2_3channel_optimized(False, batch_input, dct_matrix)
    batch_dct.mul_(mask.to(batch_input.device))
    output = batch_dct2_3channel_optimized(True, batch_dct, dct_matrix)
    return output

def filter_based_on_amp(batch_input, dct_matrix, threshold):

    batch_dct = batch_dct2_3channel_optimized(False, batch_input, dct_matrix)

    batch_dct_abs = batch_dct.abs()
    threshold_for_each_sample = torch.quantile(batch_dct_abs.flatten(start_dim=1),
                                               1.-threshold/100., dim=1, keepdim=False)
    mask = batch_dct_abs >= threshold_for_each_sample.view(batch_input.shape[0], 1, 1, 1)
    batch_dct.mul_(mask.to(batch_input.device))

    output = batch_dct2_3channel_optimized(True, batch_dct, dct_matrix)
    return output

def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0
    
def distance_from_top_left(i, j):
    dis = np.sqrt((i) ** 2 + (j) ** 2)
    return dis
#     if dis < r:
#         return 1.0
#     else:
#         return 0

# this generates a binary mask which sqrt(i^2+j^2)<r is 1
def mask_radial(size, r):
#     rows, cols = img.shape
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            mask[i, j] = distance_from_top_left(i, j) < r
    return mask

# this generates a binary mask which sqrt(i^2+j^2)<r is 1
def mask_radial_multiple_radius(size, r_list):
#     rows, cols = img.shape
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            flag = False
            for k, r in enumerate(r_list):
                if distance_from_top_left(i, j) < r:
                    mask[i, j] = k
                    flag = True
                    break
            if not flag:
                mask[i, j] = len(r_list)
    return mask

# this generates a binary mask which (sqrt(i^2+j^2)-r).abs()<threshold is 1
def equal_dist_from_top_left(size, r):
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            dis = np.sqrt((i) ** 2 + (j) ** 2)
            if np.abs(dis-r) < 0.5:
                mask[i, j] = 1.0
            else:
                mask[i, j] = 0
    return mask
