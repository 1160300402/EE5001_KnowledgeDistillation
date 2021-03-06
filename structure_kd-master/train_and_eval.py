import sys
from utils.train_options import TrainOptions
from networks.kd_model import NetModel
import logging
import warnings
warnings.filterwarnings("ignore")
from torch.utils import data
from dataset.datasets import CSDataSet
import numpy as np

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
#args = TrainOptions().initialize()
args = TrainOptions().initialize_yao() # 3.22.2022  yaoyuan修改
h, w = map(int, args.input_size.split(','))
print(args.data_dir)
trainloader = data.DataLoader(CSDataSet(args.data_dir, './dataset/list/cityscapes/train.lst', max_iters=args.num_steps*args.batch_size, crop_size=(h, w),
                scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
                batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
valloader = data.DataLoader(CSDataSet(args.data_dir, './dataset/list/cityscapes/val.lst', crop_size=(1024, 2048), mean=IMG_MEAN, scale=False, mirror=False),
                                batch_size=1, shuffle=False, pin_memory=True)
#trainloader_val = data.DataLoader(CSDataSet(args.data_dir, './dataset/list/cityscapes/train.lst', max_iters=2975, crop_size=(h, w),
#                scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
#                batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
#save_steps = int(2975/args.batch_size)
model = NetModel(args)
# for epoch in range(args.start_epoch, args.epoch_nums):
#     for step, data in enumerate(trainloader, args.last_step+1):
#         model.adjust_learning_rate(args.lr_g, model.G_solver, step)
#         model.adjust_learning_rate(args.lr_d, model.D_solver, step)
#         model.set_input(data)
#         model.optimize_parameters()
#         model.print_info(epoch, step)
#         if (step > 1) and ((step % save_steps == 0) and (step > args.num_steps - 1000)) or (step == args.num_steps - 1):
#             mean_IU, IU_array = model.evalute_model(model.student, valloader, '0', '512,512', 19, True)
#             model.save_ckpt(epoch, step, mean_IU, IU_array)
#             logging.info('[val 512,512] mean_IU:{:.6f}  IU_array:{}'.format(mean_IU, IU_array))

def main():
    best_IU = 0.0

    is_best = False
    for epoch in range(args.start_epoch, args.epoch_nums):
        for step, data in enumerate(trainloader, args.last_step + 1):
            model.adjust_learning_rate(args.lr_g, model.G_solver, step)
            model.adjust_learning_rate(args.lr_d, model.D_solver, step)
            model.set_input(data)
            model.optimize_parameters()
            #model.print_info(epoch, step)
            is_best = False
            if (step + 1) % args.print_freq == 0:
                model.print_info(epoch, step)

            if ((step+1) % args.save_freq == 0) or ((step > args.num_steps - 1000) and (step % args.save_steps == 0)):

                mean_IU, IU_array = model.evalute_model(model.student, valloader, '0', '512,512', 19, True)
                #mean_IU_train, IU_array_train = model.evalute_model(model.student, trainloader_val, '0', '512,512', 19,True)
                if(mean_IU > best_IU):
                    best_IU = mean_IU
                    is_best = True
                model.save_ckpt(epoch, step, mean_IU, IU_array)
                model.save_checkpoint(title='ohem_', epoch=epoch, step=step, is_best=is_best, mean_IU=mean_IU,
                                      best_IU=best_IU, IU_array=IU_array)
                logging.info(
                    '[val 512,512] mean_IU:{:.6f} best_IU:{:.6f} IU_array:{}'.format(mean_IU, best_IU, IU_array))
                #logging.info('[train 512,512] mean_IU:{:.6f} IU_array:{}'.format(mean_IU_train, IU_array_train))


if __name__ == '__main__':
    main()

