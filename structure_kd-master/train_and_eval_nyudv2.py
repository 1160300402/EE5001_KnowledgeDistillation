from utils.train_options import TrainOptions
from networks.kd_model import NetModel
import logging
import warnings
warnings.filterwarnings("ignore")
from torch.utils import data
from dataset.nyuv2_loader import NYUv2Loader
from networks.evaluate import evaluate_nyudv2
from utils import transform

args = TrainOptions().init_nyudv2_kd() # 3.22.2022  yaoyuan修改

mean = [104.00699, 116.66877, 122.67892]
train_transform = transform.Compose([
    transform.RandScale([0.5, 2.1]),
    transform.RandRotate([-20, 20], padding=mean, ignore_label=255),
    transform.RandomGaussianBlur(),
    transform.RandomHorizontalFlip(),
    transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=255)])

val_transform = transform.Compose([
    transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean,ignore_label=255)])

# nyu_dv2 dataset
# trainnum=799
trainloader = data.DataLoader(NYUv2Loader(root="./dataset/nyu_dv2", is_transform=True, is_augment=True, augmentations=train_transform,img_size=(256, 256)),
                  batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
# valnum=655
valloader = data.DataLoader(NYUv2Loader(root="./dataset/nyu_dv2", split='val', is_transform=True),
                batch_size=1, shuffle=False, pin_memory=True)

save_steps = int(799/args.batch_size)
model = NetModel(args)

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



def main():
    best_IU = 0.0

    is_best = False
    step = args.start_epoch * len(trainloader)
    for epoch in range(args.start_epoch, args.epoch_nums):
        for i, datas in enumerate(trainloader):
            model.adjust_learning_rate(args.lr_g, model.G_solver, step)
            model.adjust_learning_rate(args.lr_d, model.D_solver, step)
            model.set_input(datas)
            model.optimize_parameters()
            #model.print_info(epoch, step)
            step=step+1;
            is_best = False
            if (step + 1) % args.print_freq == 0:
                model.print_info(epoch, step)

            if ((step+1) % args.save_freq == 0) or ((step > args.num_steps - 1000) and (step % args.save_steps == 0)):

                mean_IU, IU_array = evaluate_nyudv2(model.student, valloader, '0', '480,640', 14, False)
                mean_IU_train, IU_array_train = evaluate_nyudv2(model.student, trainloader, '0', '256,256', 14, False)
                if(mean_IU > best_IU):
                    best_IU = mean_IU
                    is_best = True
                model.save_ckpt(epoch, step, mean_IU, IU_array)
                logging.info(
                    '[val 512,512] mean_IU:{:.6f} best_IU:{:.6f} IU_array:{}'.format(mean_IU, best_IU, IU_array))
                logging.info(
                    '[nyudv2 train 256,256] mean_IU:{:.6f} IU_array:{}'.format(mean_IU_train, IU_array_train))

if __name__ == '__main__':
    main()

