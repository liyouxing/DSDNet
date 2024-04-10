from PIL import Image
from torch.utils.data import Dataset
from random import randrange
from torchvision.transforms import Compose, ToTensor


def read_txt(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    file_list = []
    for line in lines:
        file_list.append(line.strip())
    f.close()
    return file_list  # [1:10000]


'Dataset for pre-training TLNet'


class DerainDataset1(Dataset):
    def __init__(self, crop_size=None, data_dir=None, txt_files=None, isTraining=True):
        super().__init__()

        self.crop_size = crop_size
        self.data_dir = data_dir
        self.txt_files = txt_files
        self.isTraining = isTraining

        self.in_ids = read_txt(self.data_dir + self.txt_files[0])  # get the ID list of input
        self.gt_ids = read_txt(self.data_dir + self.txt_files[1])  # get the ID list of gt

    def __getitem__(self, index):

        # open .jpg .png image
        inp_img = Image.open(self.data_dir + self.in_ids[index])
        gt_img = Image.open(self.data_dir + self.gt_ids[index])

        img_name = self.in_ids[index]

        # resize to 384x384 image
        inp_img = inp_img.resize((384, 384), Image.ANTIALIAS)
        gt_img = gt_img.resize((384, 384), Image.ANTIALIAS)

        # crop size for training
        if self.isTraining:
            width, height = inp_img.size
            # random select x,y coordinate of left-top corner
            x, y = randrange(0, width - self.crop_size[0] + 1), randrange(0, height - self.crop_size[1] + 1)
            inp_img = inp_img.crop((x, y, x + self.crop_size[0], y + self.crop_size[1]))
            gt_img = gt_img.crop((x, y, x + self.crop_size[0], y + self.crop_size[1]))

        # transform image to tensor, default: PIL int8 [0,255] --> float [0.0,1.0]
        transform = Compose([ToTensor()])
        inp_ts = transform(inp_img)  # (c,h,w)
        gt_ts = transform(gt_img)

        # checking RGB channels
        if list(inp_ts.shape)[0] is not 3 or list(gt_ts.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(inp_ts))

        if self.isTraining:
            return (inp_ts, gt_ts)
        else:
            return (inp_ts, gt_ts, img_name)

    def __len__(self):
        return len(self.in_ids)


'Dataset for joint training TA/ TAH'


class DerainDataset2(Dataset):
    def __init__(self, data_dir=None, txt_files=None, isTraining=True):
        super().__init__()

        self.data_dir = data_dir
        self.txt_files = txt_files
        self.isTraining = isTraining

        self.in_ids = read_txt(self.data_dir + self.txt_files[0])  # get the ID list of input
        self.gt_ids = read_txt(self.data_dir + self.txt_files[1])  # get the ID list of gt

    def __getitem__(self, index):

        # open .jpg .png image
        inp_img = Image.open(self.data_dir + self.in_ids[index])
        gt_img = Image.open(self.data_dir + self.gt_ids[index])

        img_name = self.in_ids[index]

        # resize to 384x384 image
        inp_img = inp_img.resize((384, 384), Image.ANTIALIAS)
        gt_img = gt_img.resize((384, 384), Image.ANTIALIAS)

        # crop size for training
        """if self.isTraining:
            width, height = inp_img.size
            # random select x,y coordinate of left-top corner
            x, y = randrange(0, width - self.crop_size[0] + 1), randrange(0, height - self.crop_size[1] + 1)
            inp_img = inp_img.crop((x, y, x + self.crop_size[0], y + self.crop_size[1]))
            gt_img = gt_img.crop((x, y, x + self.crop_size[0], y + self.crop_size[1]))"""

        # transform image to tensor, default: PIL int8 [0,255] --> float [0.0,1.0]
        transform = Compose([ToTensor()])
        inp_ts = transform(inp_img)  # (c,h,w)
        gt_ts = transform(gt_img)

        # checking RGB channels
        if list(inp_ts.shape)[0] is not 3 or list(gt_ts.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(inp_ts))

        if self.isTraining:
            return (inp_ts, gt_ts)
        else:
            return (inp_ts, gt_ts, img_name)

    def __len__(self):
        return len(self.in_ids)


'Dataset for testing unlabeled real'


class DerainDataset_real(Dataset):
    def __init__(self, data_dir=None, txt_files=None, isTraining=True):
        super().__init__()

        self.data_dir = data_dir
        self.txt_files = txt_files
        self.isTraining = isTraining

        self.in_ids = read_txt(self.data_dir + self.txt_files[0])  # get the ID list of input

    def __getitem__(self, index):

        # open .jpg .png image
        inp_img = Image.open(self.data_dir + self.in_ids[index])

        img_name = self.in_ids[index]

        # resize to 16x image
        width, height = inp_img.size
        width = int(width // 16) * 16
        height = int(height // 16) * 16
        inp_img = inp_img.resize((width, height), Image.ANTIALIAS)

        # crop size for training
        """if self.isTraining:
            width, height = inp_img.size
            # random select x,y coordinate of left-top corner
            x, y = randrange(0, width - self.crop_size[0] + 1), randrange(0, height - self.crop_size[1] + 1)
            inp_img = inp_img.crop((x, y, x + self.crop_size[0], y + self.crop_size[1]))
            gt_img = gt_img.crop((x, y, x + self.crop_size[0], y + self.crop_size[1]))"""

        # transform image to tensor, default: PIL int8 [0,255] --> float [0.0,1.0]
        transform = Compose([ToTensor()])
        inp_ts = transform(inp_img)  # (c,h,w)

        # checking RGB channels
        if list(inp_ts.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(inp_ts))

        if self.isTraining:
            return (inp_ts)
        else:
            return (inp_ts, img_name)

    def __len__(self):
        return len(self.in_ids)
