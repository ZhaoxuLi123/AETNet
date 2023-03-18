import os
import random
import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
from utils.utils import  print_log, seed_torch,write_eval_result,write_name
from datasets.HADDatasets import HADDataset, HADTestDataset
from utils.RX import RX
from models.aetnet import AETNet
from sklearn.metrics import roc_auc_score,precision_recall_curve
import scipy.io as scio
from matplotlib import rcParams
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='hyperspectral anomaly detection')
    parser.add_argument('--data_path', type=str, default='./data/HAD100Dataset/') #aviris_ng hyper
    parser.add_argument('--input_channel', type=int, default=50)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=60, help='manual seed')
    parser.add_argument('--sensor', type=str, default='aviris_ng',help=' sensor used in training,  aviris_ng or aviris')
    parser.add_argument('--device_ids', type=list, default=[0])
    parser.add_argument('--detect', type=str, default='RX',help='RX' )
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./result/50bands_aviris_ng_1/seed60_zeromask_msgms4loss_dim32/best_model.pt')
    args = parser.parse_args()
    args.save_dir = args.checkpoint_dir.replace('./result/', './test/').replace('/best_model.pt', '/')
    if not os.path.exists( args.save_dir):
        os.makedirs(args.save_dir)
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    seed_torch(seed=args.seed)
    log = open(os.path.join(args.save_dir, 'log.txt'), 'w')
    model =AETNet(in_dim=args.input_channel, dim=32, image_size=args.img_size, stage=2)
    model = model.cuda(device=args.device_ids[0])
    checkpoint = torch.load(args.checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'])
    # load dataset
    kwargs = {'num_workers':4, 'pin_memory': True}
    test_dataset = HADTestDataset(dataset_path=args.data_path, resize=args.img_size, channel=args.input_channel)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1*len(args.device_ids), shuffle=False, **kwargs)
    test_imgs, recon_imgs, scores, gt_imgs, total_time = test(args, model, test_loader)
    print_log('total_time: %.5f' % (total_time), log)
    print_log('mean_time: %.8f' % (total_time/len(test_imgs)), log)
    scores = np.asarray(scores)
    gt_imgs = np.asarray(gt_imgs)
    # fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_imgs.flatten(), scores.flatten())
    # print_log('all pixel ROCAUC: %.5f' % (per_pixel_rocauc), log)
    AU_ROC_per_img = np.zeros(len(test_imgs))
    threshold = np.zeros(len(test_imgs))
    for i in range(len(test_imgs)):
        AU_ROC_per_img[i] = roc_auc_score(gt_imgs[i, :].flatten() == 1,
                                          scores[i, :].flatten())
        precision, recall, thresholds = precision_recall_curve(gt_imgs[i, :].flatten() == 1,
                                                          scores[i, :].flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold[i] = thresholds[np.argmax(f1)]
    mean_AU_ROC = np.mean(AU_ROC_per_img)
    print_log('mean pixel ROCAUC: %.5f' % (mean_AU_ROC), log)
    write_eval_result(os.path.join(args.save_dir,'each_auc.txt'), test_dataset.test_img,
                      AU_ROC_per_img, list(range(len(test_dataset.test_img))),write_mode='a')
    write_name(os.path.join(args.save_dir, 'test_list.txt'), test_dataset.test_img)
    scio.savemat(os.path.join(args.save_dir, 'AET.mat'), {'result':scores})
    scio.savemat(os.path.join(args.save_dir, 'gt.mat'), {'gt': gt_imgs})


def test(args, model,test_loader):
    model.eval()
    scores = []
    test_imgs = []
    gt_imgs = []
    recon_imgs = []
    total_time = 0
    for (data, gt) in tqdm(test_loader):
        test_imgs.extend(data.cpu().numpy())
        gt_imgs.extend(gt.cpu().numpy())
        t1 = time.time()
        with torch.no_grad():
            data = data.cuda(device=args.device_ids[0])
            output =model(data)
            score = np.zeros([data.shape[0],data.shape[-1],data.shape[-1]])
            for i in range(data.shape[0]):
                if args.detect == 'RX':
                    score[i, :] = RX(output)
        if len(score.shape) == 2:
            score=np.expand_dims(score,axis=0)
        total_time = total_time + time.time() - t1
        scores.extend(score)
        recon_imgs.extend(output.cpu().numpy())
    return test_imgs, recon_imgs, scores, gt_imgs, total_time

if __name__ == '__main__':

    main()
