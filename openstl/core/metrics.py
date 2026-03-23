
import cv2
import numpy as np
import torch

try:
    import lpips
    from skimage.metrics import structural_similarity as cal_ssim
except:
    lpips = None
    cal_ssim = None


def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1

def _threshold(x, y, t):
    t = np.greater_equal(x, t).astype(np.float32)
    p = np.greater_equal(y, t).astype(np.float32)
    is_nan = np.logical_or(np.isnan(x), np.isnan(y))
    t = np.where(is_nan, np.zeros_like(t, dtype=np.float32), t)
    p = np.where(is_nan, np.zeros_like(p, dtype=np.float32), p)
    return t, p

def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred-true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred-true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred-true)**2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred-true)**2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred-true)**2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred-true)**2 / norm, axis=(0, 1)).sum())


def PSNR(pred, true, min_max_norm=True):

    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    else:
        if min_max_norm:  # [0, 1] normalized by min and max
            return 20. * np.log10(1. / np.sqrt(mse))  # i.e., -10. * np.log10(mse)
        else:
            return 20. * np.log10(255. / np.sqrt(mse))  # [-1, 1] normalized by mean and std

"""def PSNR(pred, true):
    mse = np.mean((np.uint8(pred * 255) - np.uint8(true * 255)) ** 2)
    return 20 * np.log10(255) - 10 * np.log10(mse)"""



def SNR(pred, true):
    """Signal-to-Noise Ratio.

    Ref: https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    """
    signal = ((true)**2).mean()
    noise = ((true - pred)**2).mean()
    return 10. * np.log10(signal / noise)


def SSIM(pred, true, **kwargs):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = pred.astype(np.float64)
    img2 = true.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def POD(hits, misses, eps=1e-6):
    """
    probability_of_detection
    Inputs:
    Outputs:
        pod = hits / (hits + misses) averaged over the T channels
        
    """
    pod = (hits + eps) / (hits + misses + eps)
    return np.mean(pod)

def SUCR(hits, fas, eps=1e-6):
    """
    success_rate
    Inputs:
    Outputs:
        sucr = hits / (hits + false_alarms) averaged over the D channels
    """
    sucr = (hits + eps) / (hits + fas + eps)
    return np.mean(sucr)

def CSI(hits, fas, misses, eps=1e-6):
    """
    critical_success_index 
    Inputs:
    Outputs:
        csi = hits / (hits + false_alarms + misses) averaged over the D channels
    """
    csi = (hits + eps) / (hits + misses + fas + eps)
    return np.mean(csi)

def sevir_metrics(pred, true, threshold):
    """
    calcaulate t, p, hits, fas, misses
    Inputs:
    pred: [N, T, C, L, L]
    true: [N, T, C, L, L]
    threshold: float
    """
    pred = pred.transpose(1, 0, 2, 3, 4)
    true = true.transpose(1, 0, 2, 3, 4)
    hits, fas, misses = [], [], []
    for i in range(pred.shape[0]):
        t, p = _threshold(pred[i], true[i], threshold)
        hits.append(np.sum(t * p))
        fas.append(np.sum((1 - t) * p))
        misses.append(np.sum(t * (1 - p)))
    return np.array(hits), np.array(fas), np.array(misses)


class LPIPS(torch.nn.Module):
    """Learned Perceptual Image Patch Similarity, LPIPS.

    Modified from
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
    """

    def __init__(self, net='alex', use_gpu=True):
        super().__init__()
        assert net in ['alex', 'squeeze', 'vgg']
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.loss_fn = lpips.LPIPS(net=net)
        if use_gpu:
            self.loss_fn.cuda()

    def forward(self, img1, img2):
        # Load images, which are min-max norm to [0, 1]
        img1 = lpips.im2tensor(img1 * 255)  # RGB image from [-1,1]
        img2 = lpips.im2tensor(img2 * 255)
        if self.use_gpu:
            img1, img2 = img1.cuda(), img2.cuda()
        return self.loss_fn.forward(img1, img2).squeeze().detach().cpu().numpy()



def _to_numpy(x):
    """Convert torch.Tensor / list / scalar to numpy (or keep numpy)."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return x  # numpy array or scalar


def _pred_channel_dim(pred: np.ndarray) -> int:
    """
    Infer channel dimension index for pred/true.
    Common:
      - (N, T, C, H, W) -> C dim = 2
      - (N, C, H, W) -> C dim = 1
    """
    if pred.ndim == 5:
        return 2
    if pred.ndim == 4:
        return 1
    raise ValueError(f"Unsupported pred ndim={pred.ndim}, expected 4 or 5.")


def _reshape_stats_to_broadcastable(stats: np.ndarray, pred: np.ndarray, name: str) -> np.ndarray:
    """
    Make mean/std broadcastable with pred for common cases.
    Target shape: (1, C, 1, 1) so it broadcasts to:
      - pred (N, C, H, W)
      - pred (N, T, C, H, W)  (it aligns to last 4 dims: T,C,H,W)
    """
    stats = np.asarray(stats)

    # scalar -> keep scalar (broadcastable)
    if stats.ndim == 0:
        return stats

    # squeeze only leading/trailing singleton dims cautiously
    # (1, C) -> (C,)
    if stats.ndim == 2 and stats.shape[0] == 1:
        stats = stats.reshape(-1)

    # (C,) -> (1, C, 1, 1)
    if stats.ndim == 1:
        return stats.reshape(1, -1, 1, 1)

    # already (1, C, 1, 1)
    if stats.ndim == 4:
        # sanity: channel is dim=1 by convention here
        if stats.shape[0] != 1 or stats.shape[2] != 1 or stats.shape[3] != 1:
            # still can be broadcastable, but keep it and let numpy try
            return stats
        return stats

    # occasionally people store as (C, 1, 1)
    if stats.ndim == 3 and stats.shape[1] == 1 and stats.shape[2] == 1:
        return stats.reshape(1, stats.shape[0], 1, 1)

    # last resort: let numpy try, but keep explicit error messages later
    return stats


def _auto_expand_mv_stats(mean4: np.ndarray, std4: np.ndarray, pred_c: int) -> tuple[np.ndarray, np.ndarray]:
    """
    If stats channel count != pred channel count, but divisible, repeat along channel dim.
    This is the MV fix: e.g. pred C=12, mean/std C=4 -> repeat 3x to C=12.
    """
    # scalar stats, no channel concept
    if np.ndim(mean4) == 0 or np.ndim(std4) == 0:
        return mean4, std4

    # We normalize stats into (1, C, 1, 1) in most cases, so channel dim=1
    if mean4.ndim >= 2:
        mean_c = mean4.shape[1]
    else:
        return mean4, std4

    if mean_c == pred_c:
        return mean4, std4

    if mean_c <= 0:
        raise ValueError(f"Invalid mean channel size: {mean_c}")

    if pred_c % mean_c != 0:
        raise ValueError(
            f"[Denorm] Channel mismatch not divisible: pred_C={pred_c}, stats_C={mean_c}. "
            f"Cannot auto-expand (to avoid silent wrong metrics)."
        )

    repeat = pred_c // mean_c
    # repeat along channel dim
    mean4 = np.repeat(mean4, repeat, axis=1)
    std4 = np.repeat(std4, repeat, axis=1)
    return mean4, std4


def _denormalize_with_stats(pred, true, mean, std):
    """
    Unified denormalization for both standard and MV tasks.
    - Convert inputs to numpy if needed
    - Reshape mean/std to broadcastable form
    - Auto-expand MV (repeat stats on channel dim) only when needed
    """
    pred = _to_numpy(pred)
    true = _to_numpy(true)
    mean = _to_numpy(mean)
    std = _to_numpy(std)

    if mean is None or std is None:
        return pred, true

    # ensure numpy arrays/scalars
    mean = np.asarray(mean)
    std = np.asarray(std)

    cdim = _pred_channel_dim(pred)
    pred_c = pred.shape[cdim]

    mean_b = _reshape_stats_to_broadcastable(mean, pred, "mean")
    std_b = _reshape_stats_to_broadcastable(std, pred, "std")

    # Try direct broadcast first (works for most non-MV)
    try:
        pred_dn = pred * std_b + mean_b
        true_dn = true * std_b + mean_b
        return pred_dn, true_dn
    except Exception:
        # If failed, try MV auto-expand (only when it makes sense)
        mean_b2, std_b2 = _auto_expand_mv_stats(mean_b, std_b, pred_c)
        pred_dn = pred * std_b2 + mean_b2
        true_dn = true * std_b2 + mean_b2
        return pred_dn, true_dn


def _cal_ssim_compat(img1_hw_c, img2_hw_c):
    if cal_ssim is None:
        raise ImportError("scikit-image is not available, cannot compute SSIM.")
    try:
        return cal_ssim(img1_hw_c, img2_hw_c, channel_axis=-1)
    except TypeError:
        return cal_ssim(img1_hw_c, img2_hw_c, multichannel=True)


def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse'],
           clip_range=[0, 1], channel_names=None,
           spatial_norm=False, return_log=True, threshold=74.0):

    pred, true = _denormalize_with_stats(pred, true, mean, std)

    eval_res = {}
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'psnr', 'snr', 'lpips', 'pod', 'sucr', 'csi']

    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')

    # channel grouping logic stays the same
    if isinstance(channel_names, list):
        assert pred.shape[2] % len(channel_names) == 0 and len(channel_names) > 1
        c_group = len(channel_names)
        c_width = pred.shape[2] // c_group
    else:
        channel_names, c_group, c_width = None, None, None

    # MSE / MAE / RMSE
    if 'mse' in metrics:
        if channel_names is None:
            eval_res['mse'] = MSE(pred, true, spatial_norm)
        else:
            mse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mse_{str(c_name)}'] = MSE(
                    pred[:, :, i*c_width:(i+1)*c_width, ...],
                    true[:, :, i*c_width:(i+1)*c_width, ...],
                    spatial_norm
                )
                mse_sum += eval_res[f'mse_{str(c_name)}']
            eval_res['mse'] = mse_sum / c_group

    if 'mae' in metrics:
        if channel_names is None:
            eval_res['mae'] = MAE(pred, true, spatial_norm)
        else:
            mae_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mae_{str(c_name)}'] = MAE(
                    pred[:, :, i*c_width:(i+1)*c_width, ...],
                    true[:, :, i*c_width:(i+1)*c_width, ...],
                    spatial_norm
                )
                mae_sum += eval_res[f'mae_{str(c_name)}']
            eval_res['mae'] = mae_sum / c_group

    if 'rmse' in metrics:
        if channel_names is None:
            eval_res['rmse'] = RMSE(pred, true, spatial_norm)
        else:
            rmse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'rmse_{str(c_name)}'] = RMSE(
                    pred[:, :, i*c_width:(i+1)*c_width, ...],
                    true[:, :, i*c_width:(i+1)*c_width, ...],
                    spatial_norm
                )
                rmse_sum += eval_res[f'rmse_{str(c_name)}']
            eval_res['rmse'] = rmse_sum / c_group

    # SEVIR metrics
    if 'pod' in metrics:
        hits, fas, misses = sevir_metrics(pred, true, threshold)
        eval_res['pod'] = POD(hits, misses)
        eval_res['sucr'] = SUCR(hits, fas)
        eval_res['csi'] = CSI(hits, fas, misses)

    need_clip = any(m in metrics for m in ['ssim', 'psnr', 'snr', 'lpips'])
    if need_clip and clip_range is not None:
        pred_clip = np.clip(pred, clip_range[0], clip_range[1])
        true_clip = np.clip(true, clip_range[0], clip_range[1])
    else:
        pred_clip, true_clip = pred, true

    # SSIM
    if 'ssim' in metrics:
        ssim = 0.0
        for b in range(pred_clip.shape[0]):
            for f in range(pred_clip.shape[1]):
                # (C,H,W) -> (H,W,C)
                p = pred_clip[b, f].swapaxes(0, 2)
                t = true_clip[b, f].swapaxes(0, 2)
                ssim += _cal_ssim_compat(p, t)
        eval_res['ssim'] = ssim / (pred_clip.shape[0] * pred_clip.shape[1])

    # PSNR
    if 'psnr' in metrics:
        psnr_v = 0.0
        for b in range(pred_clip.shape[0]):
            for f in range(pred_clip.shape[1]):
                psnr_v += PSNR(pred_clip[b, f], true_clip[b, f])
        eval_res['psnr'] = psnr_v / (pred_clip.shape[0] * pred_clip.shape[1])

    # SNR
    if 'snr' in metrics:
        snr_v = 0.0
        for b in range(pred_clip.shape[0]):
            for f in range(pred_clip.shape[1]):
                snr_v += SNR(pred_clip[b, f], true_clip[b, f])
        eval_res['snr'] = snr_v / (pred_clip.shape[0] * pred_clip.shape[1])

    # LPIPS (keep your behavior, just avoid name shadowing)
    if 'lpips' in metrics:
        if lpips is None:
            raise ImportError("lpips is not available, cannot compute LPIPS.")
        lpips_score = 0.0
        cal_lpips = LPIPS(net='alex', use_gpu=False)
        pred_hw_c = pred_clip.transpose(0, 1, 3, 4, 2)
        true_hw_c = true_clip.transpose(0, 1, 3, 4, 2)
        for b in range(pred_hw_c.shape[0]):
            for f in range(pred_hw_c.shape[1]):
                lpips_score += cal_lpips(pred_hw_c[b, f], true_hw_c[b, f])
        eval_res['lpips'] = lpips_score / (pred_hw_c.shape[0] * pred_hw_c.shape[1])

    if return_log:
        for k, v in eval_res.items():
            eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
            eval_log += eval_str

    return eval_res, eval_log







"""
def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse'],
           clip_range=[0, 1], channel_names=None,
           spatial_norm=False, return_log=True, threshold=74.0):
    if mean is not None and std is not None:
        pred = pred * std + mean
        true = true * std + mean
    eval_res = {}
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'psnr', 'snr', 'lpips', 'pod', 'sucr', 'csi']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')
    if isinstance(channel_names, list):
        assert pred.shape[2] % len(channel_names) == 0 and len(channel_names) > 1
        c_group = len(channel_names)
        c_width = pred.shape[2] // c_group
    else:
        channel_names, c_group, c_width = None, None, None

    if 'mse' in metrics:
        if channel_names is None:
            eval_res['mse'] = MSE(pred, true, spatial_norm)
        else:
            mse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mse_{str(c_name)}'] = MSE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                     true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                mse_sum += eval_res[f'mse_{str(c_name)}']
            eval_res['mse'] = mse_sum / c_group

    if 'mae' in metrics:
        if channel_names is None:
            eval_res['mae'] = MAE(pred, true, spatial_norm)
        else:
            mae_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mae_{str(c_name)}'] = MAE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                     true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                mae_sum += eval_res[f'mae_{str(c_name)}']
            eval_res['mae'] = mae_sum / c_group

    if 'rmse' in metrics:
        if channel_names is None:
            eval_res['rmse'] = RMSE(pred, true, spatial_norm)
        else:
            rmse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'rmse_{str(c_name)}'] = RMSE(pred[:, :, i*c_width: (i+1)*c_width, ...],
                                                       true[:, :, i*c_width: (i+1)*c_width, ...], spatial_norm)
                rmse_sum += eval_res[f'rmse_{str(c_name)}']
            eval_res['rmse'] = rmse_sum / c_group

    if 'pod' in metrics:
        hits, fas, misses = sevir_metrics(pred, true, threshold)
        eval_res['pod'] = POD(hits, misses)
        eval_res['sucr'] = SUCR(hits, fas)
        eval_res['csi'] = CSI(hits, fas, misses) 
        
    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])
    if 'ssim' in metrics:
        ssim = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim += cal_ssim(pred[b, f].swapaxes(0, 2),
                                 true[b, f].swapaxes(0, 2), multichannel=True)
        eval_res['ssim'] = ssim / (pred.shape[0] * pred.shape[1])

    if 'psnr' in metrics:
        psnr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                psnr += PSNR(pred[b, f], true[b, f])
        eval_res['psnr'] = psnr / (pred.shape[0] * pred.shape[1])

    if 'snr' in metrics:
        snr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                snr += SNR(pred[b, f], true[b, f])
        eval_res['snr'] = snr / (pred.shape[0] * pred.shape[1])

    if 'lpips' in metrics:
        lpips = 0
        cal_lpips = LPIPS(net='alex', use_gpu=False)
        pred = pred.transpose(0, 1, 3, 4, 2)
        true = true.transpose(0, 1, 3, 4, 2)
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                lpips += cal_lpips(pred[b, f], true[b, f])
        eval_res['lpips'] = lpips / (pred.shape[0] * pred.shape[1])

    if return_log:
        for k, v in eval_res.items():
            eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
            eval_log += eval_str

    return eval_res, eval_log
    


def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse'],
           clip_range=[0, 1], channel_names=None,
           spatial_norm=False, return_log=True, threshold=74.0):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    if mean is not None and isinstance(mean, torch.Tensor):
        mean = mean.detach().cpu().numpy()
    if std is not None and isinstance(std, torch.Tensor):
        std = std.detach().cpu().numpy()

    if mean is not None and std is not None:

        if pred.ndim == 5 and mean.ndim >= 1:
            # (1, C2, 1, 1)
            if mean.ndim == 1:
                mean = mean.reshape(1, mean.shape[0], 1, 1)
                std = std.reshape(1, std.shape[0], 1, 1)

            if mean.ndim == 4:
                C = pred.shape[2]       
                C2 = mean.shape[1]      
                if C != C2:
                    if C % C2 == 0:
                        repeat = C // C2  
                        mean = np.repeat(mean, repeat, axis=1)
                        std = np.repeat(std, repeat, axis=1)
                    else:
                        raise ValueError(
                            f"Cannot broadcast mean/std: pred C={C}, mean/std C2={C2}"
                        )
        pred = pred * std + mean
        true = true * std + mean
    # -------------------------------------------------------------------

    eval_res = {}
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'psnr', 'snr', 'lpips', 'pod', 'sucr', 'csi']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')
    if isinstance(channel_names, list):
        assert pred.shape[2] % len(channel_names) == 0 and len(channel_names) > 1
        c_group = len(channel_names)
        c_width = pred.shape[2] // c_group
    else:
        channel_names, c_group, c_width = None, None, None

    if 'mse' in metrics:
        if channel_names is None:
            eval_res['mse'] = MSE(pred, true, spatial_norm)
        else:
            mse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mse_{str(c_name)}'] = MSE(
                    pred[:, :, i*c_width: (i+1)*c_width, ...],
                    true[:, :, i*c_width: (i+1)*c_width, ...],
                    spatial_norm
                )
                mse_sum += eval_res[f'mse_{str(c_name)}']
            eval_res['mse'] = mse_sum / c_group

    if 'mae' in metrics:
        if channel_names is None:
            eval_res['mae'] = MAE(pred, true, spatial_norm)
        else:
            mae_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mae_{str(c_name)}'] = MAE(
                    pred[:, :, i*c_width: (i+1)*c_width, ...],
                    true[:, :, i*c_width: (i+1)*c_width, ...],
                    spatial_norm
                )
                mae_sum += eval_res[f'mae_{str(c_name)}']
            eval_res['mae'] = mae_sum / c_group

    if 'rmse' in metrics:
        if channel_names is None:
            eval_res['rmse'] = RMSE(pred, true, spatial_norm)
        else:
            rmse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'rmse_{str(c_name)}'] = RMSE(
                    pred[:, :, i*c_width: (i+1)*c_width, ...],
                    true[:, :, i*c_width: (i+1)*c_width, ...],
                    spatial_norm
                )
                rmse_sum += eval_res[f'rmse_{str(c_name)}']
            eval_res['rmse'] = rmse_sum / c_group

    if 'pod' in metrics:
        hits, fas, misses = sevir_metrics(pred, true, threshold)
        eval_res['pod'] = POD(hits, misses)
        eval_res['sucr'] = SUCR(hits, fas)
        eval_res['csi'] = CSI(hits, fas, misses)

    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])

    if 'ssim' in metrics:
        ssim = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                ssim += cal_ssim(pred[b, f].swapaxes(0, 2),
                                 true[b, f].swapaxes(0, 2), multichannel=True)
        eval_res['ssim'] = ssim / (pred.shape[0] * pred.shape[1])

    if 'psnr' in metrics:
        psnr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                psnr += PSNR(pred[b, f], true[b, f])
        eval_res['psnr'] = psnr / (pred.shape[0] * pred.shape[1])

    if 'snr' in metrics:
        snr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                snr += SNR(pred[b, f], true[b, f])
        eval_res['snr'] = snr / (pred.shape[0] * pred.shape[1])

    if 'lpips' in metrics:
        lpips_v = 0
        cal_lpips = LPIPS(net='alex', use_gpu=False)
        pred_chw = pred.transpose(0, 1, 3, 4, 2)
        true_chw = true.transpose(0, 1, 3, 4, 2)
        for b in range(pred_chw.shape[0]):
            for f in range(pred_chw.shape[1]):
                lpips_v += cal_lpips(pred_chw[b, f], true_chw[b, f])
        eval_res['lpips'] = lpips_v / (pred_chw.shape[0] * pred_chw.shape[1])

    if return_log:
        for k, v in eval_res.items():
            eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
            eval_log += eval_str

    return eval_res, eval_log"""