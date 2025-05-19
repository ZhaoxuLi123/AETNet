import scipy.io as scio
import numpy as np
import argparse
import torch
from utils.RX import RX
from models.aetnet import AETNet
from sklearn.metrics import roc_auc_score
import math
from scipy.stats import entropy


def calculate_entropy(band):
    hist, _ = np.histogram(band, bins=256, range=(0, 1))  # 假设数据归一化到 [0,1]
    prob = hist / hist.sum()
    return entropy(prob)


def data_preprocess(channel, data, band_choose='order'):
    m, n, b = data.shape
    data = (data - np.min(data)) / (np.max(data) - np.min(data)+1e-10)
    if channel > b:
        # x = np.repeat(data, math.ceil(channel/b), axis=2)
        x = np.tile(data, (1, 1, math.ceil(channel/b)))
        x = x[:, :, :channel]
    elif channel < b:
        if band_choose == 'order':
            x = data[:, :, :channel]
        elif band_choose == 'std':
            band_variances = np.var(data, axis=(0, 1))  # 沿空间维度计算方差
            # 按方差从大到小排序，选择前N个波段
            selected_band_indices = np.argsort(band_variances)[:channel]  # 获取索引
            x = data[:, :, selected_band_indices]
        elif band_choose == 'entropy':
            band_entropies = np.array([calculate_entropy(data[:, :, i]) for i in range(b)])
            selected_band_indices = np.argsort(band_entropies)[:channel]
            x = data[:, :,selected_band_indices]
        elif band_choose == 'low_correlation':
            data_2d = data.reshape(-1, b)  # 形状 (n_samples, n_bands)
            correlation_matrix = np.corrcoef(data_2d, rowvar=False)
            band_correlation_scores = np.sum(np.abs(correlation_matrix), axis=1)
            selected_band_indices = np.argsort(band_correlation_scores)[:channel]
            x = data[:, :, selected_band_indices]
        else:
            raise ValueError(
                f"Invalid band_choose value: {band_choose}. Expected one of: 'order', 'std', 'entropy', 'low_correlation'")
    else:
        x= data
    x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1
    x = x * 0.1
    x = torch.from_numpy(x)
    x = x.type(torch.FloatTensor)
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x

def main(img_data, gt_map=None, band_choose = 'order'):
    """
    :param img_data: numpy array,[h,w,channel]
    :param gt_map: numpy array,[h,w]
    :param gt_map: str, 'order','std','entropy','low_correlation'
    :return:
    """

    parser = argparse.ArgumentParser(description='hyperspectral anomaly detection')
    parser.add_argument('--input_channel', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--device_ids', type=list, default=[0])
    parser.add_argument('--detect', type=str, default='RX', )
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./result/50bands_aviris_ng_1/seed60_zeromask_msgms4loss_dim32/best_model.pt')
    args = parser.parse_args()
    model = AETNet(in_dim=args.input_channel, dim=32, image_size=args.img_size, stage=2)
    model = model.cuda(device=args.device_ids[0])
    checkpoint = torch.load(args.checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    img = data_preprocess(args.input_channel, img_data, band_choose=band_choose)
    img = img.cuda(device=args.device_ids[0])
    output = model(img)
    RX_score_map = RX(img)
    detect_score_map = RX(output)
    if gt_map is not None:
        RX_AUC = roc_auc_score(gt_map.flatten(), RX_score_map.flatten())
        AETNet_AUC = roc_auc_score(gt_map.flatten(), detect_score_map.flatten())
        print('RX_AUC', RX_AUC)
        print('AETNet_AUC', AETNet_AUC)
    return detect_score_map


if __name__ == '__main__':
    data_path = 'data/Sandiego.mat'
    img_data = scio.loadmat(data_path)['data']
    gt_map = scio.loadmat(data_path)['map']
    detect_score_map = main(img_data, gt_map, band_choose='order')

    data_path = 'data/abu-urban-1.mat'
    img_data = scio.loadmat(data_path)['data']
    gt_map = scio.loadmat(data_path)['map']
    detect_score_map = main(img_data, gt_map, band_choose='order')

    # Inference on Any Image
    # img_data = your data
    # detect_score_map = main(img_data, None, band_choose='order')