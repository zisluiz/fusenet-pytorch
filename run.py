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
import glob
import os
import os.path as osp
from pathlib import Path
import psutil
from datetime import datetime
import time
import nvidia_smi

def predict_image(rgb, depth):
    input = {'rgb_image': rgb, 'depth_image': depth}
    model.set_input(input)
    model.forward()
    _, pred = torch.max(model.output.data.cpu(), 1)
    #pred = pred.float().detach().int().numpy()
    #palet_file = 'datasets/palette.txt'
    #impalette = list(np.genfromtxt(palet_file,dtype=np.uint8).reshape(3*256))
    #im = util.tensor2labelim(pred, impalette)
    im = util.tensor2label(pred)
    return im

if __name__ == '__main__':
    f = open("results/run_"+str(int(round(time.time() * 1000)))+".txt", "w+")
    f.write('=== Start time: '+str(datetime.now())+'\n')

    p = psutil.Process(os.getpid())
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    sys.argv.extend(['--dataroot', 'datasets/sunrgbd','--dataset', 'sunrgbd','--name', 'sunrgbd', '--epoch', '400'])


    opt = TestOptions()
    # hard-code some parameters for test
    # opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    opt.dataroot = 'datasets/sunrgbd'
    opt.dataset = 'sunrgbd'
    opt.name = 'sunrgbd'
    opt = opt.parse()

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset.ignore_label = 1
    model = create_model(opt, dataset.dataset)
    model.setup(opt)
    model.eval()

    print('Starting list image files')
    filesCount = 0

    files = glob.glob("datasets/mestrado/**/rgb/*.png", recursive = True)
    files.extend(glob.glob("datasets/mestrado/**/rgb/*.jpg", recursive = True))
    cpuTimes = [0.0, 0.0, 0.0, 0.0]

    gpuTimes = 0.0
    gpuMemTimes = 0.0
    maxNumThreads = 0
    memUsageTimes = 0

    for imagePath in files:
        print('imagePath: ' + imagePath)
        pathRgb = Path(imagePath)
        datasetName = osp.basename(str(pathRgb.parent.parent))
        #print('datasetName: ' + datasetName)
        parentDatasetDir = str(pathRgb.parent.parent)
        #print('parentDatasetDir: ' + parentDatasetDir)
        depthImageName = os.path.basename(imagePath).replace('jpg','png')
        rgb_image = Image.open(imagePath)
        depth_image = Image.open(parentDatasetDir + '/depth/' + depthImageName)

        os.makedirs('results/' + datasetName, exist_ok=True)

        if datasetName == "active_vision" or datasetName == "putkk":
            rgb_image = rgb_image.crop((420, 0, 1500, 1080))
            depth_image = depth_image.crop((420, 0, 1500, 1080))

        test_transforms_rgb = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor()])
        test_transforms_depth = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.Grayscale(1),
                                            transforms.ToTensor()])
        rgb_image = test_transforms_rgb(rgb_image).float().unsqueeze(0)
        depth_image = test_transforms_depth(depth_image).float().unsqueeze(0)

        pred = predict_image(rgb_image, depth_image)

        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        curGpuTime = res.gpu
        #curGpuMemTime = res.memory #(in percent)
        curGpuMemTime = mem_res.used / 1e+6
        gpuTimes += curGpuTime
        gpuMemTimes += curGpuMemTime
        f.write('GPU Usage Percent: ' + str(curGpuTime) + '\n')
        f.write('GPU Mem Usage (MB)): ' + str(curGpuMemTime) + '\n')

        curProcessCpuPerU = p.cpu_percent()
        curCpusPerU = psutil.cpu_percent(interval=None, percpu=True)

        # gives a single float value
        for i in range(len(cpuTimes)):
            curProcessCpu = curProcessCpuPerU
            curCpu = curCpusPerU[i]
            cpuTimes[i] += curCpu
            f.write('Process CPU Percent: ' + str(curProcessCpu) + ' --- CPU Percent: ' + str(curCpu) + '\n')

        # you can convert that object to a dictionary
        memInfo = dict(p.memory_full_info()._asdict())
        curMemUsage = memInfo['uss']
        memUsageTimes += curMemUsage

        f.write('Process memory usage: ' + str(curMemUsage / 1e+6) + '\n')
        f.write('Memory information: ' + str(memInfo) + '\n')

        if maxNumThreads < p.num_threads():
            maxNumThreads = p.num_threads()

        #print('############## Index: ')
        #print(index)
        cv2.imwrite('results/'+datasetName+'/'+depthImageName, pred)
        filesCount = filesCount + 1
    nvidia_smi.nvmlShutdown()

    start = time.time()
    for imagePath in files:
        pathRgb = Path(imagePath)
        datasetName = osp.basename(str(pathRgb.parent.parent))
        parentDatasetDir = str(pathRgb.parent.parent)
        depthImageName = os.path.basename(imagePath).replace('jpg', 'png')
        rgb_image = Image.open(imagePath)
        depth_image = Image.open(parentDatasetDir + '/depth/' + depthImageName)

        if datasetName == "active_vision" or datasetName == "putkk":
            rgb_image = rgb_image.crop((420, 0, 1500, 1080))
            depth_image = depth_image.crop((420, 0, 1500, 1080))

        test_transforms_rgb = transforms.Compose([transforms.Resize((224, 224)),
                                                  transforms.ToTensor()])
        test_transforms_depth = transforms.Compose([transforms.Resize((224, 224)),
                                                    transforms.Grayscale(1),
                                                    transforms.ToTensor()])
        rgb_image = test_transforms_rgb(rgb_image).float().unsqueeze(0)
        depth_image = test_transforms_depth(depth_image).float().unsqueeze(0)

        pred = predict_image(rgb_image, depth_image)
    end = time.time()

    f.write('=== Mean GPU Usage Percent: ' + str(gpuTimes / filesCount) + '\n')
    f.write('=== Mean GPU Mem Usage (MB): ' + str(gpuMemTimes / filesCount) + '\n')
    for i in range(len(cpuTimes)):
        f.write("=== Mean cpu"+str(i)+" usage: " + str(cpuTimes[i] / filesCount) + '\n')
    f.write("=== Mean memory usage (MB): " + str((memUsageTimes / filesCount) / 1e+6) + '\n')

    f.write("=== Total image predicted: " + str(filesCount)+'\n')
    f.write("=== Seconds per image: " + str( ((end-start)/filesCount) )+'\n')
    f.write("=== Max num threads: " + str(maxNumThreads) + '\n')

    f.write('=== End time: ' + str(datetime.now())+'\n')
    f.close()