import logging
import os.path as osp
import torch.backends.cudnn as cudnn
from utils.criterion import CriterionDSN, CriterionOhemDSN, CriterionPixelWise, \
    CriterionAdv, CriterionAdvForG, CriterionAdditionalGP, CriterionPairWiseforWholeFeatAfterPool
from networks.pspnet_combine import Res_pspnet, BasicBlock, Bottleneck
from utils.train_options import TrainOptions
import logging
import warnings
from torch.utils import data
from dataset.nyuv2_loader import NYUv2Loader
from networks.evaluate import evaluate_nyudv2
from utils.utils import *
from utils.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, RandScale, RandomGaussianBlur, CenterCrop
warnings.filterwarnings("ignore")

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

class nyudv2_trainmodel():

    def __init__(self, args):
        cudnn.enabled = True

        augmentations_train = Compose([
            RandScale(),
            RandomRotate(10),
            RandomGaussianBlur(),
            RandomHorizontallyFlip(),
            CenterCrop(size=(args.train_h, args.train_w))])



        # nyu_dv2 dataset
        # trainnum=799
        trainloader = data.DataLoader(NYUv2Loader(root="./dataset/nyu_dv2", is_transform=True, is_augment=True, augmentations=augmentations_train),
                                      batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        # valnum=655
        valloader = data.DataLoader(NYUv2Loader(root="./dataset/nyu_dv2", split='val', is_transform=True,),
                                    batch_size=1, shuffle=False, pin_memory=True)
        self.args = args
        self.trainloader = trainloader
        self.valloader = valloader
        device = args.device
        if args.train_type == 'T':
            model = Res_pspnet(Bottleneck, [3, 4, 23, 3], num_classes=args.classes_num)
        else:
            model = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes=args.classes_num)
        #load_S_model(args, model, False)
        model = torch.nn.DataParallel(model.to(device))
        if(args.is_student_load_imgnet):
            model.load_state_dict(torch.load(args.student_pretrain_model_imgnet))
            logging.info("model => load" + str(args.student_pretrain_model_imgnet))
        print_model_parm_nums(model, 'train_model')
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
        self.criterion = torch.nn.DataParallel(CriterionDSN().cuda())  # CriterionCrossEntropy()
        cudnn.benchmark = True
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)


    def adjust_learning_rate(self, base_lr, optimizer, i_iter):
        args = self.args
        lr = base_lr * ((1 - float(i_iter) / args.max_iter) ** (args.power))
        optimizer.param_groups[0]['lr'] = lr
        return lr

    def print_info(self, epoch, iter, remain_time):
        logging.info(
            'epoch:{:4d} step:{:5d} lr:{:.6f} loss:{:.5f} Remain:{remain_time}'.format(
                epoch, iter, self.optimizer.param_groups[-1]['lr'],self.loss,remain_time=remain_time))

    def train(self, model):
        args = self.args
        max_iter = args.epochs * len(self.trainloader)
        args.max_iter = max_iter
        current_iter =  args.start_epoch * len(self.trainloader)
        model.train()
        for epoch in range(args.start_epoch, args.epochs):
            epoch_log = epoch + 1
            batch_time = AverageMeter()
            data_time = AverageMeter()
            end = time.time()
            for i, datas in enumerate(self.trainloader):
                data_time.update(time.time() - end)

                self.adjust_learning_rate(args.lr, self.optimizer, current_iter)
                input, target = datas
                input = input.cuda()
                target = target.cuda()
                pred = model.train()(input)
                self.optimizer.zero_grad()
                loss = self.criterion(pred, target)
                loss.backward()
                self.loss = loss.item()
                self.optimizer.step()

                batch_time.update(time.time() - end)
                end = time.time()
                current_iter = current_iter + 1
                if (i + 1) % args.print_freq == 0:
                    remain_iter = max_iter - current_iter
                    remain_time = remain_iter * batch_time.avg
                    t_m, t_s = divmod(remain_time, 60)
                    t_h, t_m = divmod(t_m, 60)
                    remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
                    self.print_info(epoch, current_iter, remain_time)

            if (epoch_log % args.save_freq == 0) or (epoch == args.epochs-1):
                mean_IU, IU_array = evaluate_nyudv2(model, self.valloader, '0', '256,256', 14, False)
                self.save_model(model=model, epoch=epoch_log, mean_IU=mean_IU)
                logging.info(
                    '[nyudv2 val 480,640] mean_IU:{:.6f} IU_array:{}'.format(mean_IU, IU_array))


    def save_model(self, model, epoch, mean_IU):#只存放model
        torch.save(model.state_dict(),
            osp.join(self.args.snapshot_dir, 'nyudv2_1' + str(self.args.train_type) + '_' + str(epoch) + '_' + str(mean_IU) + '.pth'))

if __name__ == '__main__':
    args = TrainOptions().initialize_nyudv2train()
    net = nyudv2_trainmodel(args)
    net.train(net.model)