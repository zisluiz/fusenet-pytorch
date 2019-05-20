import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable

if __name__ == '__main__':
    data_dir = '/data/train'

    test_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        #transforms.Normalize([0.485, 0.456, 0.406],
                                        #                     [0.229, 0.224, 0.225])
                                        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=torch.load('checkpoints/nyuv2/400_net_FuseNet.pth')
    model.eval()

    to_pil = transforms.ToPILImage()
    index = predict_image(image)
    print('############## Index: ')
    print(index)

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels