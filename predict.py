import sys
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util import util
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image


def predict_image(rgb, depth):
    input = {'rgb_image': rgb, 'depth_image': depth}
    model.set_input(input)
    model.forward()
    _, pred = torch.max(model.output.data.cpu(), 1)
    #pred = pred.float().detach().int().numpy()
    palet_file = 'datasets/palette.txt'
    impalette = list(np.genfromtxt(palet_file,dtype=np.uint8).reshape(3*256))
    im = util.tensor2labelim(pred, impalette)
    return im

if __name__ == '__main__':
    sys.argv.extend(['--dataroot', 'datasets/nyuv2','--dataset', 'nyuv2','--name', 'nyuv2', '--epoch', '400'])


    opt = TestOptions()
    # hard-code some parameters for test
    # opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    opt.dataroot = 'datasets/nyuv2 '
    opt.dataset = 'nyuv2 '
    opt.name = 'nyuv2 '
    opt = opt.parse()

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset.ignore_label = 1
    model = create_model(opt, dataset.dataset)
    model.setup(opt)
    model.eval()

    rgb_image = Image.open('datasets/test/33_rgb_image.png')
    depth_image = Image.open('datasets/test/33_depth_image.png')

    test_transforms_rgb = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor()])
    test_transforms_depth = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.Grayscale(1),
                                        transforms.ToTensor()])
    rgb_image = test_transforms_rgb(rgb_image).float().unsqueeze(0)
    depth_image = test_transforms_depth(depth_image).float().unsqueeze(0)

    pred = predict_image(rgb_image, depth_image)
    #print('############## Index: ')
    #print(index)
    cv2.imwrite('results/res_33.png', pred)

