import os
import collections
import torch
import numpy as np
import scipy.misc as m
import cv2
import sys
from torch.utils import data
from networks.pspnet_combine import Res_pspnet, BasicBlock, Bottleneck
from networks.evaluate import *
from utils.utils import recursive_glob
from utils.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, RandScale, RandomGaussianBlur, CenterCrop, Scale
from utils import transform

class NYUv2Loader(data.Dataset):
    """
    NYUv2 loader
    Download From (only 13 classes):
    test source: http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz
    train source: http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz
    test_labels source:
      https://github.com/ankurhanda/nyuv2-meta-data/raw/master/test_labels_13/nyuv2_test_class13.tgz
    train_labels source:
      https://github.com/ankurhanda/nyuv2-meta-data/raw/master/train_labels_13/nyuv2_train_class13.tgz
    """

    def __init__(
            self,
            root,
            split="training",
            is_transform=False,
            is_augment=False,
            img_size=(480, 640),
            augmentations=Compose([Scale(512), RandomRotate(10)]),
            img_norm=True,
            test_mode=False,
    ):
        self.root = root
        self.is_transform = is_transform
        self.is_augment = is_augment
        self.n_classes = 14
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.cmap = self.color_map(normalized=False)

        split_map = {"training": "train", "val": "test"}
        self.split = split_map[split]

        for split in ["train", "test"]:
            file_list = recursive_glob(rootdir=self.root + "/" + split + "/", suffix="png")
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        img_number = img_path.split("_")[-1][:4]
        lbl_path = os.path.join(
            self.root, self.split + "_label", "new_nyu_class13_" + img_number + ".png"
        )

        # img = m.imread(img_path)
        # img = np.array(img, dtype=np.uint8)
        #
        # lbl = m.imread(lbl_path)
        # lbl = np.array(lbl, dtype=np.uint8)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        img = np.float32(img)
        # img = np.array(img, dtype=np.uint8) #4.4 新增

        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        lbl = np.array(lbl, dtype=np.uint8) #4.4 新增

        if not (len(img.shape) == 3 and len(lbl.shape) == 2):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        if self.is_augment:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = cv2.resize(img, dsize=(self.img_size[1], self.img_size[0]))  # uint8 with RGB mode,前width，后height
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, dsize=(self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        lbl = lbl.astype(int)
        assert np.all(classes == np.unique(lbl))

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def color_map(self, N=256, normalized=False):
        """
        Return Color Map in PASCAL VOC format
        """

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = "float32" if normalized else "uint8"
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255.0 if normalized else cmap
        return cmap

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.cmap[l, 0]
            g[temp == l] = self.cmap[l, 1]
            b[temp == l] = self.cmap[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    restore_from = '../snapshots/nyudv2_kd_4aug200_0.001_399_0.384887.pth'
    student = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes=14)
    student = torch.nn.DataParallel(student)
    student.load_state_dict(torch.load(restore_from,map_location='cpu'))
    student.eval()

    restore_from1 = '../snapshots/nyudv2_2S_400_0.37867.pth'
    student1 = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes=14)
    student1 = torch.nn.DataParallel(student1)
    student1.load_state_dict(torch.load(restore_from, map_location='cpu'))
    student1.eval()

    augmentations = Compose([
        RandScale(),
        RandomRotate(10),
        RandomGaussianBlur(),
        RandomHorizontallyFlip(),
        CenterCrop(size=(256, 256))])

    local_path = "../dataset/nyu_dv2"
    # dst = NYUv2Loader(local_path, is_transform=True, augmentations=augmentations)
    dst = NYUv2Loader(root=local_path, split='val', is_transform=True, augmentations=augmentations)
    bs = 2
    trainloader = data.DataLoader(NYUv2Loader(root="../dataset/nyu_dv2", is_transform=True, img_size=(256, 256), is_augment=True, augmentations=augmentations),
                                  batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    valloader = data.DataLoader(NYUv2Loader(root="../dataset/nyu_dv2", split='val', is_transform=True),
                                batch_size=2, shuffle=True, pin_memory=True)

    for i, datas in enumerate(valloader):
        imgs, labels = datas  # (batchsize,3,480,640) (batchsize,480,640)

        output = predict_sliding_test(student, imgs.numpy(), (480,640), 14, False, 1)
        seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output1 = predict_sliding_test(student1, imgs.numpy(), (480, 640), 14, False, 1)
        seg_pred1 = np.asarray(np.argmax(output1, axis=2), dtype=np.uint8)

        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 4)
        for j in range(bs):
            #axarr[j][0].imshow((imgs[j]*255).astype(np.uint8))
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
            axarr[j][2].imshow(dst.decode_segmap(seg_pred))
            axarr[j][3].imshow(dst.decode_segmap(seg_pred1))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()