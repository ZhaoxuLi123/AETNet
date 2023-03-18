import time
import random
import os
import numpy as np
import torch
import re

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs



def time_file_str():
    ISOTIMEFORMAT = '%Y-%m-%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string + '-{}'.format(random.randint(1, 10000))


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()

def write_eval_result(txt, name_list, AU_ROCs, anomaly_ids, write_mode = 'w'):
    f = open(txt, write_mode)
    for i in range(len(anomaly_ids)):
        id = anomaly_ids[i]
        data_path = name_list[id]
        name = data_path.split('/')[-1].split('.')[0]
        auc =AU_ROCs[i]
        f.write('{:}:{:} \n'.format(name,auc))
    f.close()

def write_name(txt, name_list):
    f = open(txt, 'w')
    for i in range(len(name_list)):
        data_path = name_list[i]
        name =re.split(r'[\\/.]', data_path)[-2]
        f.write('scene{:d}:{:}\n'.format(i+1,name))
    f.close()



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



def seed_torch(seed=50):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


def save_checkpoint(model_path, model):
    state = {
        'state_dict': model.state_dict(),
    }
    torch.save(state, model_path)
