import os
import torch
import math
import sys 
import time
import argparse
import numpy as np
from imageio import imread
from skimage.transform import  resize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models.swin import SwinTransformerSys
from models.common import post_process_output
from scipy import ndimage
import cv2 as cv

from utils.dataset_processing.grasp import detect_grasps
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Visualise Grasp Transformer Output')

    # Network
    parser.add_argument('--network', type=str,default="/media/geoffroy/T7/grasp-transformer/output/models/230608_1345_/epoch_173_iou_0.97", help='Path to saved network to evaluate')
    # Dataset & Data 
    parser.add_argument('--dataset', type=str, default="cornell",help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,default="/media/geoffroy/T7/cornell" ,help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for training (0/1)')

    #Visualization parameters
    parser.add_argument('--width', type=int, default=224, help="Width of the input image")
    parser.add_argument('--height', type=int, default=224, help="Height of the input image")

    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')

    args = parser.parse_args()
    return args

def resize1(img, shape):
    """
    Resize image to shape.
    :param shape: New shape.
    """
    if img.shape == shape:
        return
    img = resize(img, shape, preserve_range=True).astype(img.dtype)
    return img

def normalise(img):
    """
    Normalise the image by converting to float [0,1] and zero-centering
    """
    img = img.astype(np.float32)/255.0
    img -= img.mean()
    return img

def depth_normalize(img):
    img = np.clip((img - img.mean()), -1, 1)
    return img

def zoom_image_center(image, scale=0.5):
    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the new dimensions after zooming
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Calculate the center coordinates
    center_x = original_width // 2
    center_y = original_height // 2

    # Calculate the new top-left corner coordinates
    left = center_x - (new_width // 2)
    top = center_y - (new_height // 2)

    # Calculate the new bottom-right corner coordinates
    right = left + new_width
    bottom = top + new_height

    # Crop the image to the new dimensions
    zoomed_image = image[top:bottom, left:right]

    # Resize the image to the new dimensions
    zoomed_image = cv.resize(zoomed_image, (new_width, new_height), interpolation=cv.INTER_LANCZOS4)

    return zoomed_image
#Dataloader class to create standardized inputs tensor shape (1, 4, 224, 224) ie. 1 Tensor, 4 Channels, dimx, dimy
class ReachyDataset(Dataset):
    def __init__(self, dataset_path):        
        self.rgb_img = []
        self.save_rgb = [] #Save of the RGB image before normalisation for visualization
        self.depth_img = []
        self.filename = []
        #Create a custom dataset with rescaled and normalized images
        for filename in os.scandir(dataset_path):
            if "depth" in filename.name and args.use_depth:
                curr_depth = imread(filename.path)
                curr_depth = zoom_image_center(curr_depth)
                curr_depth = depth_normalize(curr_depth)
                curr_depth = resize1(curr_depth, (args.width,args.height))
                self.depth_img.append(curr_depth)
                self.filename.append(filename)

            elif "rgb" in filename.name and args.use_rgb: #RGB
                curr_rgb = imread(filename.path)
                curr_rgb = zoom_image_center(curr_rgb)
                curr_rgb = resize1(curr_rgb, (args.width,args.height))
                self.save_rgb.append(curr_rgb)
                curr_rgb = normalise(curr_rgb)
                curr_rgb = curr_rgb.transpose((2, 0, 1))
                self.rgb_img.append(curr_rgb)
                self.filename.append(filename)
        if args.use_depth and args.use_rgb:
            assert len(self.depth_img) == len(self.rgb_img)

    def __len__(self):
        if args.use_depth:
            return len(self.depth_img)q
        else:
            return len(self.rgb_img)
    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
        
    def __getitem__(self, index):
        if args.use_rgb and args.use_depth:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(self.depth_img[index], 0),
                        self.rgb_img[index]),
                    0
                )
            )
        elif args.use_rgb:
            x = self.numpy_to_torch(
                np.expand_dims(self.depth_img[index], 0)
            )
        else:
            x = self.numpy_to_torch(
                np.expand_dims(self.rgb_img[index], 0)
            )
        return x

    def get_save_rgb(self, index):
        return self.save_rgb[index]
    
    def get_filename(self, index):
        return self.filename[index]

args = parse_args()    
if __name__ == "__main__":
    dataset_path = args.dataset_path
    network_path = args.network
    # Load the network
    device = torch.device('cpu') #Change to cuda if torch.cuda.is_available() == True
    net = torch.load(args.network, map_location=device)
    dataset = ReachyDataset(dataset_path=dataset_path)
    print(f"{dataset.__len__()} image(s) detected")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        shuffle=True,
        num_workers=1
    )

    t_infer = []
    with torch.no_grad():
        for index, img in enumerate(dataloader):
            t0 = time.time()
            output = net(img)
            q_img, ang_img, width_img = post_process_output(output[0], output[1], output[2], output[3])
            gs_l = detect_grasps(q_img, ang_img, width_img, no_grasps=3)
            print(f"Inference time = {time.time() - t0:.2f}s")
            t_infer.append(time.time() - t0)

            #Visualisation
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 4, 1)
            #Show Grasp
            for g in gs_l:
                g.plot(ax)
            ax.imshow(dataset.get_save_rgb(index))

            ax = fig.add_subplot(1, 4, 2)
            plot = ax.imshow(q_img, cmap="jet", vmin=0, vmax=1)  # ？terrain
            plt.colorbar(plot)
            ax.axis('off')
            ax.set_title('q image')

            ax = fig.add_subplot(1, 4, 3)  # flag  prism jet
            plot = ax.imshow(ang_img, cmap="hsv", vmin=-np.pi / 2, vmax=np.pi / 2)
            plt.colorbar(plot)
            ax.axis('off')
            ax.set_title('angle')

            ax = fig.add_subplot(1, 4, 4)
            plot = ax.imshow(width_img, cmap='jet', vmin=-0, vmax=150)
            plt.colorbar(plot)
            ax.set_title('width')
            ax.axis('off')

            plt.show()
    t_infer = np.asarray(t_infer)
    print(f"Mean inference time on {device} = {t_infer.mean():.2f}s")

