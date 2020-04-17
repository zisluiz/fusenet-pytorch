import sys
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
import torch
import cv2
from torchvision import transforms
from PIL import Image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['FuseNet']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

def predict_image(model, device, rgb, depth):
    #input = {'rgb_image': rgb, 'depth_image': depth}
    rgb_image = rgb.to(device)
    depth_image = depth.to(device)
    output = model(rgb_image, depth_image)

    _, pred = torch.max(output.data.cpu(), 1)
    pred = pred.float().detach().int().numpy()    
    return pred


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_path = 'checkpoints/sunrgbd/220_net_FuseNet.pth'

    model = load_checkpoint(load_path)    

    rgb_image = Image.open('datasets/test/rgb_00000.png')
    depth_image = Image.open('datasets/test/depth_00000.png')

    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor()])
    rgb_image = test_transforms(rgb_image).float().unsqueeze(0)
    depth_image = test_transforms(depth_image).float().unsqueeze(0)

    index = predict_image(model, device, rgb_image, depth_image)
    print('############## Index: ')
    print(index)
    cv2.imwrite('results/res_00000.png', index.reshape(224,224))

