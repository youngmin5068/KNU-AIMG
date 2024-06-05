import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def partition_batch(ls, size):
    i = 0
    partitioned_lists = []
    while i < len(ls):
        partitioned_lists.append(ls[i: i+size])
        i += size

    return partitioned_lists


def make_sure_in_range(val, min_val, max_val):
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val


def crop(original_img, crop_shape, crop_position, method="center",
         in_place=False, background_val="min"):
    I, J = original_img.shape
    crop_x, crop_y = crop_position
    x_delta, y_delta = crop_shape

    if method == "center":
        min_x = int(np.round(crop_x - x_delta / 2))
        max_x = int(np.round(crop_x + x_delta / 2))
        min_y = int(np.round(crop_y - y_delta / 2))
        max_y = int(np.round(crop_y + y_delta/2))
    elif method == "upper_left":
        min_x = int(np.round(crop_x))
        max_x = int(np.round(crop_x + x_delta))
        min_y = int(np.round(crop_y))
        max_y = int(np.round(crop_y + y_delta))

    min_x = make_sure_in_range(min_x, 0, I)
    max_x = make_sure_in_range(max_x, 0, I)
    min_y = make_sure_in_range(min_y, 0, J)
    max_y = make_sure_in_range(max_y, 0, J)

    if in_place:
        original_img[min_x:max_x, min_y:max_y] = 1.0
    else:
        if background_val == "min":
            output = np.ones(crop_shape) * np.min(original_img)
        else:
            output = np.ones(crop_shape) * background_val
        real_x_delta = max_x - min_x
        real_y_delta = max_y - min_y
        origin_x = crop_shape[0] - real_x_delta
        origin_y = crop_shape[1] - real_y_delta
        output[origin_x:, origin_y:] = original_img[min_x:max_x, min_y:max_y]
        return output


def get_crop_mask(loc, crop_shape, image_shape, method, indicator=True):
    crop_map = np.zeros(image_shape)
    for crop_loc in loc:

        if indicator:
            crop_map[int(crop_loc[0]), int(crop_loc[1])] = 999.0

        crop(crop_map, crop_shape, crop_loc, method=method, in_place=True)
    return crop_map

def crop_pytorch(original_img_pytorch, crop_shape, crop_position, out,
                 method="center", background_val="min"):

    H, W = original_img_pytorch.shape
    crop_x, crop_y = crop_position
    x_delta, y_delta = crop_shape

    if method == "center":
        min_x = int(np.round(crop_x - x_delta / 2))
        max_x = int(np.round(crop_x + x_delta / 2))
        min_y = int(np.round(crop_y - y_delta / 2))
        max_y = int(np.round(crop_y + y_delta / 2))
    elif method == "upper_left":
        min_x = int(np.round(crop_x))
        max_x = int(np.round(crop_x + x_delta))
        min_y = int(np.round(crop_y))
        max_y = int(np.round(crop_y + y_delta))

    min_x = make_sure_in_range(min_x, 0, H)
    max_x = make_sure_in_range(max_x, 0, H)
    min_y = make_sure_in_range(min_y, 0, W)
    max_y = make_sure_in_range(max_y, 0, W)

    if background_val == "min":
        out[:, :] = original_img_pytorch.min()
    else:
        out[:, :] = background_val
    real_x_delta = max_x - min_x
    real_y_delta = max_y - min_y
    origin_x = crop_shape[0] - real_x_delta
    origin_y = crop_shape[1] - real_y_delta
    out[origin_x:, origin_y:] = original_img_pytorch[min_x:max_x, min_y:max_y]


def get_max_window(input_image, window_shape, pooling_logic="avg"):
    N, C, H, W = input_image.size()
    if pooling_logic == "avg":

        pool_map = torch.nn.functional.avg_pool2d(input_image, window_shape, stride=1)
    elif pooling_logic in ["std", "avg_entropy"]:

        output_size = (H - window_shape[0] + 1, W - window_shape[1] + 1)
        sliding_windows = F.unfold(input_image, kernel_size=window_shape).view(N,C, window_shape[0]*window_shape[1], -1)

        if pooling_logic == "std":
            agg_res = sliding_windows.std(dim=2, keepdim=False)
        elif pooling_logic == "avg_entropy":
            agg_res = -sliding_windows*torch.log(sliding_windows)-(1-sliding_windows)*torch.log(1-sliding_windows)
            agg_res = agg_res.mean(dim=2, keepdim=False)

        pool_map = F.fold(agg_res, kernel_size=(1, 1), output_size=output_size)
    _, _, _, W_map = pool_map.size()

    _, max_linear_idx = torch.max(pool_map.view(N, C, -1), -1)

    max_idx_x = max_linear_idx // W_map
    max_idx_y = max_linear_idx - max_idx_x * W_map

    upper_left_points = torch.cat([max_idx_x.unsqueeze(-1), max_idx_y.unsqueeze(-1)], dim=-1)
    return upper_left_points


def generate_mask_uplft(input_image, window_shape, upper_left_points, gpu_number):

    N, C, H, W = input_image.size()
    window_h, window_w = window_shape

    mask_x_min = upper_left_points[:,:,0]
    mask_x_max = upper_left_points[:,:,0] + window_h
    mask_y_min = upper_left_points[:,:,1]
    mask_y_max = upper_left_points[:,:,1] + window_w

    mask_x = Variable(torch.arange(0, H).view(-1, 1).repeat(N, C, 1, W))
    mask_y = Variable(torch.arange(0, W).view(1, -1).repeat(N, C, H, 1))
    if gpu_number is not None:
        device = torch.device("cuda:{}".format(gpu_number))
        mask_x = mask_x.cuda().to(device)
        mask_y = mask_y.cuda().to(device)
    x_gt_min = mask_x.float() >= mask_x_min.unsqueeze(-1).unsqueeze(-1).float()
    x_ls_max = mask_x.float() < mask_x_max.unsqueeze(-1).unsqueeze(-1).float()
    y_gt_min = mask_y.float() >= mask_y_min.unsqueeze(-1).unsqueeze(-1).float()
    y_ls_max = mask_y.float() < mask_y_max.unsqueeze(-1).unsqueeze(-1).float()

    selected_x = x_gt_min * x_ls_max
    selected_y = y_gt_min * y_ls_max
    selected = selected_x * selected_y
    mask = 1 - selected.float()
    return mask