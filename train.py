import os
import random
import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets.HADDatasets import HADDataset, HADTestDataset
from models.aetnet import AETNet
from losses.msgms_loss import MSGMS_Loss
from utils.utils import time_string, convert_secs2time, AverageMeter, print_log, seed_torch, save_checkpoint, write_eval_result
from utils.RX import RX
from kornia.losses import SSIMLoss
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='hyperspectral anomaly detection')
    parser.add_argument('--device_ids', type=list, default=[0])
    parser.add_argument('--data_path', type=str, default='./data/HAD100Dataset/')
    parser.add_argument('--start_channel_id', type=int, default=0, help='the start id of spectral channel')
    parser.add_argument('--input_channel', type=int, default=50, help='the spectral channel number of input HSI')
    parser.add_argument('--latent_channel', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200, help='the maximum of training epochs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of Adam')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='decay of Adam')
    parser.add_argument('--seed', type=int, default=60, help='manual seed')
    parser.add_argument('--mask', type=str, default='zero', help='spectrum pasted on masks, no, zero,image, sin, other_sensor')
    parser.add_argument('--train_ratio', type=float, default=1, help='data ratio used for training')
    parser.add_argument('--sensor', type=str, default='aviris_ng',help='sensor used in training,  aviris_ng or aviris  test')
    parser.add_argument('--loss', type=str, default='msgms', help = 'l1, l2, ssim, gms, msgms')
    parser.add_argument('--gms_pool_num', type=int, default=4)
    parser.add_argument('--use_unet', type=bool, default=True)
    parser.add_argument('--use_swin', type=bool, default=True)
    parser.add_argument('--save_txt', type=bool, default=True)
    parser.add_argument('--save_model', type=bool, default=True)
    args = parser.parse_args()
    # build save path
    if args.loss == 'msgms':
        args.save_dir = os.path.join('./result/', str(args.input_channel)+'bands_'+str(args.sensor)+'_'+str(args.train_ratio),
                                 'seed' + str(args.seed) + '_'+ args.mask+'mask_'+ args.loss + str(args.gms_pool_num) + 'loss_dim' + str(args.latent_channel))
    else:
        args.save_dir = os.path.join('./result/', str(args.input_channel)+'bands_'+str(args.sensor)+'_'+str(args.train_ratio),
                                 'seed' + str(args.seed) + '_'+ args.mask+'mask_'+ args.loss +'loss_dim' + str(args.latent_channel))

    epoch_write_dir = os.path.join(args.save_dir, 'epoch')
    if not os.path.exists(epoch_write_dir):
        os.makedirs(epoch_write_dir)
    # set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    seed_torch(seed=args.seed)

    log = open(os.path.join(args.save_dir, 'training_log.txt'), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    # load model and dataset
    model = AETNet(in_dim=args.input_channel,dim=32,image_size=args.img_size,stage=2,use_unet=args.use_unet,use_siwn=args.use_swin)
    if len(args.device_ids)>1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)
    model = model.cuda(device=args.device_ids[0])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # load dataset
    kwargs = {'num_workers':4, 'pin_memory': True}
    train_dataset = HADDataset(dataset_path=args.data_path,sensor= args.sensor, mask_class = args.mask, resize=args.img_size,
                                start_channel=args.start_channel_id, channel=args.input_channel, train_ratio = args.train_ratio)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size*len(args.device_ids), shuffle=True, **kwargs)
    test_dataset = HADTestDataset(dataset_path=args.data_path, resize=args.img_size, channel=args.input_channel)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1*len(args.device_ids), shuffle=False, **kwargs)

    # start training
    max_score =0
    start_time = time.time()
    epoch_time = AverageMeter()
    stop_counter = 0
    for epoch in range(1, args.epochs + 1):
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)
        train(args, model, epoch, train_loader, optimizer, log)
        stop_counter += 1
        test_imgs, recon_imgs,scores, gt_imgs = test(args, model, test_loader)
        scores = np.asarray(scores)
        gt_imgs = np.asarray(gt_imgs)
        RX_max = np.max(scores[0,:])
        # calculate ROCAUC
        print_log('max score: %.5f ' % (RX_max), log)
        per_pixel_rocauc = roc_auc_score(gt_imgs.flatten(), scores.flatten())
        print_log('all pixel ROCAUC: %.5f' % (per_pixel_rocauc), log)
        AU_ROC_per_img = np.zeros(len(test_imgs))
        for i in range(len(test_imgs)):
            AU_ROC_per_img[i] = roc_auc_score(gt_imgs[i, :].flatten() == 1,
                                                              scores[i, :].flatten())
        mean_AU_ROC = np.mean(AU_ROC_per_img)
        print_log('mean pixel ROCAUC: %.5f' % (mean_AU_ROC), log)
        if args.save_txt:
            write_eval_result(os.path.join(epoch_write_dir, 'epoch{:d}.txt'.format(epoch)),test_dataset.test_img,AU_ROC_per_img,
                              list(range(len(test_dataset.test_img))))
        if RX_max > max_score:
            if args.save_model:
                model_save_dir = os.path.join(args.save_dir, 'best_model.pt')
                save_checkpoint(model_save_dir, model)
                print_log('************************save model************************', log)
            if args.save_txt:
                write_eval_result(os.path.join(args.save_dir, 'best.txt'), test_dataset.test_img,
                                  AU_ROC_per_img,list(range(len(test_dataset.test_img))))
            max_score = RX_max
            stop_counter = 0
        if stop_counter == 30:
            print_log('--------------------------early stop----------------------------', log)
            break
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    log.close()


def train(args, model, epoch, train_loader, optimizer, log):
    model.train()
    losses = AverageMeter()
    if args.loss == 'msgms':
        Loss = MSGMS_Loss(args.device_ids, pool_num=args.gms_pool_num)
    elif args.loss == 'gms':
        Loss = MSGMS_Loss(args.device_ids, pool_num=0)
    elif args.loss == 'ssim':
        Loss = SSIMLoss(5)
    elif args.loss == 'l1':
        Loss = nn.L1Loss(reduction='mean')
    elif args.loss == 'l2':
        Loss = nn.MSELoss(reduction='mean')
    else:
        raise Exception("the loss is not defined")
    for (data,data_m) in tqdm(train_loader):
        optimizer.zero_grad()
        data = data.cuda(device=args.device_ids[0])
        data_m = data_m.cuda(device=args.device_ids[0])
        output = model(data_m)
        loss = Loss(data, output)
        losses.update(loss.item(), data.size(0))
        loss.backward()
        optimizer.step()
    print_log(('Train Epoch: {}  Loss: {:.8f} '.format(epoch,  losses.avg)), log)

def test(args, model,test_loader):
    model.eval()
    scores = []
    test_imgs = []
    gt_imgs = []
    recon_imgs = []
    for (data, gt) in tqdm(test_loader):
        test_imgs.extend(data.cpu().numpy())
        gt_imgs.extend(gt.cpu().numpy())
        with torch.no_grad():
            data = data.cuda(device=args.device_ids[0])
            output =model(data)
            score = np.zeros([data.shape[0],data.shape[-1],data.shape[-1]])
            for i in range(data.shape[0]):
                score1 = RX(output)
                score[i, :] = score1
        if len(score.shape) == 2:
            score=np.expand_dims(score,axis=0)
        scores.extend(score)
        recon_imgs.extend(output.cpu().numpy())
    return test_imgs, recon_imgs, scores, gt_imgs



if __name__ == '__main__':

    main()
