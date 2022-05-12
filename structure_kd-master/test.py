from torch.utils import data
from networks.pspnet_combine import Res_pspnet, BasicBlock, Bottleneck
from networks.evaluate import evaluate_main
from dataset.datasets import CSDataTestSet,CSDataSet
from utils.train_options import TrainOptionsForTest
import torch
import numpy as np
from utils.utils import *
import logging

if __name__ == '__main__':
    args = TrainOptionsForTest().initialize()
    testloader = data.DataLoader(CSDataTestSet(args.data_dir, './dataset/list/cityscapes/test.lst', crop_size=(1024, 2048)),
                                    batch_size=1, shuffle=False, pin_memory=True)
    IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) #add
    valloader = data.DataLoader(CSDataSet(args.data_dir, './dataset/list/cityscapes/val.lst', crop_size=(1024, 2048), mean=IMG_MEAN, scale=False, mirror=False),
                                batch_size=1, shuffle=False, pin_memory=True) #add
    student = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes = 19)
    student.load_state_dict(torch.load(args.resume_from, map_location='cpu'))
    # mean_IU,IU_array = evaluate_main(student, testloader, '0', '512,512', 19, True, type = 'test') #change
    mean_IU,IU_array = evaluate_main(student, valloader, '0', '512,512', 19, False, type = 'val') #change
    print('[val 512,512] mean_IU:{:.6f}  IU_array:{}'.format(mean_IU, IU_array)) #add

    # teacher = Res_pspnet(Bottleneck, [3, 4, 23, 3], num_classes=args.classes_num)
    # load_T_model(teacher, args.T_ckpt_path)
    # mean_IU, IU_array = evaluate_main(teacher, valloader, '0', '512,512', 19, True, type='val')  # change
    # print('[val 512,512] mean_IU:{:.6f}  IU_array:{}'.format(mean_IU, IU_array))  # add