import os
import torch

from turtle import Turtle
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
from cutout import Cutout

# several data augumentation strategies
def cv_random_flip_rgb(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label

def cv_random_flip_rgb_edge(img, label, edge):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, edge

def cv_random_flip_rgb_contour_body(img, label, contour, body):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        contour = contour.transpose(Image.FLIP_LEFT_RIGHT)
        body = body.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, contour, body


def cv_random_flip_trip(img, label, pseudo_label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        pseudo_label = pseudo_label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, pseudo_label


def randomCrop_rgb(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomCrop_rgb_edge(image, label, edge):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), edge.crop(random_region)


def randomCrop_trip(image, label, pseudo_label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), pseudo_label.crop(random_region)


def randomRotation_rgb(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def randomRotation_rgb_edge(image, label, edge):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return image, label, edge


def randomRotation_rgb_contour_body(image, label, contour, body):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        contour = contour.rotate(random_angle, mode)
        body = body.rotate(random_angle, mode)
    return image, label, contour, body


def randomRotation_trip(image, label, pseudo_label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        pseudo_label = pseudo_label.rotate(random_angle, mode)
    return image, label, pseudo_label

def cv_random_flip_rgbd(img, label, depth):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, depth


def randomCrop_rgbd(image, label, depth):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region)


def randomCrop_rgb_edge(image, label, edge):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), edge.crop(random_region)


def randomCrop_rgb_edge_body(image, label, edge, body):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), edge.crop(random_region), body.crop(random_region)


def randomRotation_rgbd(image, label, depth):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
    return image, label, depth

# several data augumentation strategies
def cv_random_flip_weak(img, label, mask, gray):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        gray = gray.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, mask, gray


def randomCrop_weak(image, label, mask, gray):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), mask.crop(random_region), gray.crop(random_region)


def randomRotation_weak(image, label, mask, gray):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        mask = mask.rotate(random_angle, mode)
        gray = gray.rotate(random_angle, mode)
    return image, label, mask, gray


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


class SalObjDatasetRGB(data.Dataset):
    def __init__(self, image_root, gt_root, edge_root, trainsize, patchsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edges = sorted(self.edges)
        
        self.filter_files()
        self.size = len(self.images)
        self.patchsize = patchsize
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        edge = self.binary_loader(self.edges[index])
        
        image, gt, edge = cv_random_flip_rgb_edge(image, gt, edge)
        image, gt, edge = randomCrop_rgb_edge(image, gt, edge)
        image, gt, edge = randomRotation_rgb_edge(image, gt, edge)
        
        
        image_pure = self.img_transform(image)
        gt_pure = self.gt_transform(gt)
        edge = self.edge_transform(edge)
        
        
        return image_pure, gt_pure, edge, index
        
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        edges = []
        grays = []

        for img_path, gt_path, edge_path in zip(self.images, self.gts, self.edges):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            edge = Image.open(edge_path)
            
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                edges.append(edge_path)
            
        self.images = images
        self.gts = gts
        self.edges = edges
        
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size



class SemiSalObjDatasetRGB(data.Dataset):
    def __init__(self, image_root, gt_root, psd_root, edge_root, psd_contour_root, psd_body_root, trainsize, patchsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.psds = [psd_root + f for f in os.listdir(psd_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.psds_contour = [psd_contour_root + f for f in os.listdir(psd_contour_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.psds_body = [psd_body_root + f for f in os.listdir(psd_body_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.psds = sorted(self.psds)
        self.edges = sorted(self.edges)
        self.psds_contour = sorted(self.psds_contour)
        self.psds_body = sorted(self.psds_body)
        
        self.filter_files()
        self.size = len(self.images)
        self.patchsize = patchsize
        
        self.img_color_jitter_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
        self.psd_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
        self.psd_contour_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
        self.psd_body_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
        
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        psd = self.binary_loader(self.psds[index])
        edge = self.binary_loader(self.edges[index])
        psd_contour = self.binary_loader(self.psds_contour[index])
        psd_body = self.binary_loader(self.psds_body[index])
        
        # image_wda, gt_wda, edge_wda = randomCrop_rgb_edge(image, gt, edge)
        image_wda, gt_wda, edge_wda = cv_random_flip_rgb_edge(image, gt, edge)
        image_wda, gt_wda, edge_wda = randomCrop_rgb_edge(image_wda, gt_wda, edge_wda)
        image_wda, gt_wda, edge_wda = randomRotation_rgb_edge(image_wda, gt_wda, edge_wda)
        
        image_sda, psd_sda, contour_sda, body_sda = cv_random_flip_rgb_contour_body(image, psd, psd_contour, psd_body)
        image_sda, psd_sda, contour_sda, body_sda = randomCrop_rgb_edge_body(image_sda, psd_sda, contour_sda, body_sda)
        image_sda, psd_sda, contour_sda, body_sda = randomRotation_rgb_contour_body(image_sda, psd_sda, contour_sda, body_sda)
        

        image_wda = self.img_transform(image_wda)
        gt_wda = self.gt_transform(gt_wda)
        edge_wda = self.edge_transform(edge_wda)

        image_sda = self.img_color_jitter_transform(image_sda)
        psd_sda = self.psd_transform(psd_sda)
        contour_sda = self.psd_contour_transform(contour_sda)
        body_sda = self.psd_body_transform(body_sda)
        
        return image_wda, gt_wda, edge_wda, image_sda, psd_sda, contour_sda, body_sda, index
        
        
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        psds = []
        edges = []
        psds_contour = []
        psds_body = []
        
        for img_path, gt_path, psd_path, edge_path, psd_contour_path, psd_body_path in zip(self.images, self.gts, self.psds, self.edges, self.psds_contour, self.psds_body):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            psd = Image.open(psd_path)
            edge = Image.open(edge_path)
            psd_contour = Image.open(psd_contour_path)
            psd_body = Image.open(psd_body_path)
            
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                psds.append(psd_path)
                edges.append(edge_path)
                psds_contour.append(psd_contour_path)
                psds_body.append(psd_body_path)
                
        self.images = images
        self.gts = gts
        self.psds = psds        
        self.edges = edges
        self.psds_contour = psds_contour
        self.psds_body = psds_body        
                
        
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size



class SalObjDatasetRGBD(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root=None, trainsize=352):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.rgb_loader(self.depths[index])
        image, gt, depth = cv_random_flip_rgbd(image, gt, depth)
        image, gt, depth = randomCrop_rgbd(image, gt, depth)
        image, gt, depth = randomRotation_rgbd(image, gt, depth)
        image = colorEnhance(image)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)

        return image, gt, depth, index

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        depths = []
        for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            if img.size == gt.size and gt.size == depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths = depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h), Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, edge_root, ratio, batchsize, trainsize, patchsize, shuffle=True, num_workers=12, pin_memory=True):
    dataset = SalObjDatasetRGB(image_root, gt_root, edge_root, trainsize=trainsize, patchsize=patchsize)
    dataset_size = len(dataset)
    label_set_size = int(ratio * dataset_size)
    print("label_set_size:", label_set_size)

    unlabel_set_size = dataset_size - label_set_size
    
    if ratio == 1:
        # data_loader = data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        print('Supervised learning')
    else:
        # partial_size = int(option['partial_data'] * dataset.size)
        print("Semi_supervised/Unsupervised learning........")
    
    # set_size = dataset.size
    # train_img_idx = range(set_size)
    
    train_sampler, train_remain_sampler = data.random_split(dataset, [label_set_size, unlabel_set_size], generator=torch.Generator().manual_seed(0))
    
    
    data_loader = data.DataLoader(dataset=train_sampler, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    data_remain_loader = data.DataLoader(dataset=train_remain_sampler, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)      
    
    
    return data_loader, data_remain_loader



def get_loader_semi(image_root, gt_root, psd_root, edge_root, psd_contour_root, psd_body_root, ratio, batchsize, trainsize, patchsize, shuffle=True, num_workers=12, pin_memory=True):
    dataset = SemiSalObjDatasetRGB(image_root, gt_root, psd_root, edge_root, psd_contour_root, psd_body_root, trainsize=trainsize, patchsize=patchsize)
    dataset_size = len(dataset)
    label_set_size = int(ratio * dataset_size)
    print("label_set_size:", label_set_size)

    unlabel_set_size = dataset_size - label_set_size
    
    if ratio == 1:
        # data_loader = data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        print('Supervised learning')
    else:
        # partial_size = int(option['partial_data'] * dataset.size)
        print("Semi_supervised/Unsupervised learning........")
    
    
    train_sampler, train_remain_sampler = data.random_split(dataset, [label_set_size, unlabel_set_size], generator=torch.Generator().manual_seed(0))
    
    data_loader = data.DataLoader(dataset=train_sampler, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    data_remain_loader = data.DataLoader(dataset=train_remain_sampler, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)      
    
    
    return data_loader, data_remain_loader


def get_loader_rgbd(image_root, gt_root, depth_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):
    dataset = SalObjDatasetRGBD(image_root, gt_root, depth_root, trainsize=trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader, dataset.size

class SalObjDatasetWeak(data.Dataset):
    def __init__(self, image_root, gt_root, mask_root, gray_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.masks = [mask_root + f for f in os.listdir(mask_root) if f.endswith('.png')]
        self.grays = [gray_root + f for f in os.listdir(gray_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.masks = sorted(self.masks)
        self.grays = sorted(self.grays)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gray_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        # print(self.images[index])
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        mask = self.binary_loader(self.masks[index])
        gray = self.binary_loader(self.grays[index])
        # print('*****************************')
        image, gt, mask, gray = cv_random_flip_weak(image, gt, mask, gray)
        image, gt, mask, gray = randomCrop_weak(image, gt, mask, gray)
        # print('#############################')
        image, gt, mask, gray = randomRotation_weak(image, gt, mask, gray)
        # print('-------------------------------')
        image = colorEnhance(image)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        mask = self.mask_transform(mask)
        gray = self.gray_transform(gray)
        # depth = depth//(255*255)
        # depth = torch.tensor(depth, dtype=torch.float)
        # depth = torch.FloatTensor(depth)

        return image, gt, mask, gray


    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        masks = []
        grays = []
        for img_path, gt_path, mask_path, gray_path in zip(self.images, self.gts, self.masks, self.grays):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            mask = Image.open(mask_path)
            gray = Image.open(gray_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                masks.append(mask_path)
                grays.append(gray_path)
        self.images = images
        self.gts = gts
        self.masks = masks
        self.grays = grays


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def depth_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('I')

    def resize(self, img, gt, mask, gray):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), mask.resize((w, h), Image.NEAREST), gray.resize((w, h), Image.NEAREST)
        else:
            return img, gt, mask, gray

    def __len__(self):
        return self.size


# dataloader for training
def get_loader_weak(image_root, gt_root, mask_root, gray_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):
    dataset = SalObjDatasetWeak(image_root, gt_root, mask_root, gray_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class test_dataset_rgbd:
    def __init__(self, image_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        depth = self.rgb_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)

        name = self.images[self.index].split('/')[-1]
        # image_for_post=self.rgb_loader(self.images[self.index])
        # image_for_post=image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, depth, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# class eval_Dataset(data.Dataset):
#     def __init__(self, img_root, label_root):
#         lst_label = sorted(os.listdir(label_root))
#         # print(label_root)
#         lst_pred = sorted(os.listdir(img_root))
#         # print(img_root)
#         lst = []
#         for name in lst_label:
#             if name in lst_pred:
#                 lst.append(name)

#         self.image_path = list(map(lambda x: os.path.join(img_root, x), lst))
#         self.label_path = list(map(lambda x: os.path.join(label_root, x), lst))

#     def __getitem__(self, item):
#         pred = Image.open(self.image_path[item]).convert('L')
#         gt = Image.open(self.label_path[item]).convert('L')
#         if pred.size != gt.size:
#             pred = pred.resize(gt.size, Image.BILINEAR)
#         return pred, gt

#     def __len__(self):
#         return len(self.image_path)

        

class eval_Dataset(data.Dataset):
    def __init__(self, img_root, label_root):
        lst_label = sorted(os.listdir(label_root))
        # print(label_root)
        lst_pred = sorted(os.listdir(img_root))
        # print(img_root)
        self.label_abbr, self.pred_abbr = lst_label[0].split('.')[-1], lst_pred[0].split('.')[-1]
        label_list, pred_list = [], []
        for name in lst_label:
            label_name = name.split('.')[0]
            if label_name+'.'+self.label_abbr in lst_label:
                label_list.append(name)
    
        for name in lst_pred:
            label_name = name.split('.')[0]
            if label_name+'.'+self.pred_abbr in lst_pred:
                pred_list.append(name)

        self.image_path = list(map(lambda x: os.path.join(img_root, x), pred_list))
        self.label_path = list(map(lambda x: os.path.join(label_root, x), label_list))

    def __getitem__(self, item):
        img_path = self.image_path[item]
        label_path = self.label_path[item]
        pred = Image.open(img_path).convert('L')
        gt = Image.open(label_path).convert('L')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)