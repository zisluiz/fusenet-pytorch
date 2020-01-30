import os
import sys
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util.visualizer import save_images,save_scannet_prediction
from util import util
from util import html
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import scipy.misc as misc

def predict_image(rgb, depth):
    #image_tensor = test_transforms(image).float()
    #image_tensor = image_tensor.unsqueeze_(0)
    #input = Variable(rgb, depth)
    #input = input.to(device)
    #self.rgb_image = input['rgb_image'].to(self.device)
    #self.depth_image = input['depth_image'].to(self.device)
    #self.mask = input['mask'].to(self.device)
    #input['rgb_image'] = rgb
    #input['depth_image'] = depth
    input = {'rgb_image': rgb, 'depth_image': depth}
    model.set_input(input)
    model.forward()
    #index = output.data.cpu().numpy().argmax()
    _, pred = torch.max(model.output.data.cpu(), 1)
    #pred = pred.float().detach().int().numpy()
    palet_file = 'datasets/palette.txt'
    impalette = list(np.genfromtxt(palet_file,dtype=np.uint8).reshape(3*256))
    im = util.tensor2labelim(pred, impalette)
    return im

def _transform(filename, __channels):
    #image = misc.imread(filename, flatten=False if __channels else True, mode='RGB' if __channels else 'P')
    image = cv2.imread(filename)
    resize_size = 224
    image = cv2.resize(image, (resize_size,resize_size))
    #resize_image = misc.imresize(image,
    #                                [resize_size, resize_size], interp='nearest')

    if __channels == 3:
        return image.reshape(1, resize_size,resize_size,3 if __channels else 1)    
    else:
        return image

if __name__ == '__main__':
    sys.argv.extend(['--dataroot', 'datasets/nyuv2','--dataset', 'nyuv2','--name', 'nyuv2', '--epoch', '400'])


    opt = TestOptions()
    # hard-code some parameters for test
    # opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    opt.dataroot = 'datasets/nyuv2'
    opt.dataset = 'nyuv2'
    opt.name = 'nyuv2'    
    opt = opt.parse()    

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset.ignore_label = 1
    model = create_model(opt, dataset.dataset)
    model.setup(opt)
    #model.netFuseNet.load_state_dict(torch.load('checkpoints/nyuv2/400_net_FuseNet.pth'))
    #model.load_networks('400')
    model.eval()

    #rgb_image = np.array(Image.open('datasets/test/rgb_00000.png'))
    #depth_image = np.array(Image.open('datasets/test/depth_00000.png'))

    rgb_image = Image.open('datasets/test/33_rgb_image.png')
    depth_image = Image.open('datasets/test/33_depth_image.png')

    #rgb_image = np.array(Image.open('datasets/test/rgb_00000.png'))
    #depth_image = np.array(Image.open('datasets/test/depth_00000.png'))

    test_transforms_rgb = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        #transforms.Normalize([0.485, 0.456, 0.406],
                                        #                     [0.229, 0.224, 0.225])
                                        ])
    test_transforms_depth = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.Grayscale(1),
                                        transforms.ToTensor(),
                                        #transforms.Normalize([0.485, 0.456, 0.406],
                                        #                     [0.229, 0.224, 0.225])
                                        ])
    rgb_image = test_transforms_rgb(rgb_image).float().unsqueeze(0)
    depth_image = test_transforms_depth(depth_image).float().unsqueeze(0)

    #rgb_image = test_transforms(rgb_image)
    #depth_image = test_transforms(depth_image)

    ##rgb_image = _transform('datasets/test/1.png', 3)
    ##depth_image = _transform('datasets/test/1_1.png', 1)

    #rgb_image = np.array(rgb_image).reshape(1, 224,224,3)
    #depth_image = np.array(depth_image).reshape(1, 224,224,1)

    ##rgb_image = transforms.ToTensor()(np.array(rgb_image,dtype=np.uint8))
    ##depth_image = transforms.ToTensor()(np.array(depth_image,dtype=np.uint8))

    #rgb_image = transforms.Resize(224,224).ToTensor()(rgb_image)
    #depth_image = transforms.Resize(224,224).ToTensor()(depth_image[:, :, np.newaxis])
    #rgb_image = test_transforms(rgb_image)
    #depth_image = test_transforms(depth_image[:, :, np.newaxis])



    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = create_model(opt, dataset.dataset)

    #model.setup(opt)
    #model.eval()

    #to_pil = transforms.ToPILImage()
    pred = predict_image(rgb_image, depth_image)
    #print('############## Index: ')
    #print(index)
    cv2.imwrite('results/res_33.png', pred)

