import torch
import math
import time
import numpy as np
from imageio.v2 import imread
from skimage.transform import  resize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models.swin import SwinTransformerSys
from models.common import post_process_output
from utils.dataset_processing.grasp import detect_grasps
from ros_utils.real_realsense import RealsenseD401Infos, RealsenseD401Subscriber
from ros_utils.luxonis_listener import LuxonisSubscriber
from ros_utils.reachy_depth_subscriber import RosDepthCameraSubscriber
import matplotlib.pyplot as plt
import rclpy
from reachy_sdk import ReachySDK
import cv2
from reachy_sdk.trajectory import goto
from ros_utils.target_publisher import MarkerPublisher
from ros_utils.tf_listener import FrameListener

''' ============ ROS Init Subscribers  ==============='''
#Init ROS wrapping
rclpy.init()

#Init Reachy's depth camera subscriber

''' === LUXONIS ==='''

image_getter = LuxonisSubscriber(node_name="depth_viewer")
image_getter.update_image()

''' === REALSENSE === '''
'''image_getter = RealsenseD401Subscriber(node_name="depth_viewer")
image_getter.update_image()'''

''' === GAZEBO ===='''
'''
image_getter = RosDepthCameraSubscriber(node_name="depth_viewer")
image_getter.update_image()
'''

'''
Get depth camera intrinsic values from a ROS subsriber
The depth camera is located at the same spot as head_tip 
Intel Realsense D400
'''
'''
depth_cam_info = RealsenseD401Infos(node_name="cam_inf")
rclpy.spin_once(depth_cam_info)
FOCAL_LENGTHX = depth_cam_info.fx
FOCAL_LENGTHY = depth_cam_info.fy
OPTICAL_CENTERX = depth_cam_info.cx
OPTICAL_CENTERY = depth_cam_info.cy
depth_cam_info.destroy_node()
'''


''' ============ Neural Network Initialization ============='''
#network_path = "/media/geoffroy/T7/grasp-transformer/output/models/230608_1345_/epoch_206_iou_1.00_statedict.pt"
#network_path = "/home/geoffroy/epoch_206_iou_1.00_statedict.pt"
#network_path = "/media/geoffroy/T7/grasp-transformer/output/graspnet/models/230629_1546__d=1_scale=3/epoch_290_iou_0.24"
#network_path = "/media/geoffroy/T7/grasp-transformer/output/cornell_2/230704_1633_Learning_on_the_cornell_dataset_with_10%_of_the_dataset_removed_for_cross_validation/epoch_97_iou_1.00_statedict.pt"
network_path = "/home/geoffroy/epoch_51_iou_1.00"

#Creating the model
model = SwinTransformerSys(in_chans=4, embed_dim=48, num_heads=[1, 2, 4, 8])
#Loading the model

#Cornell
#model.load_state_dict(torch.load(network_path, map_location=torch.device('cpu')))

model = torch.load(network_path, map_location=torch.device("cpu"))

#Graspnet
#model = torch.load(network_path, map_location=torch.device('cpu'))
device = torch.device("cpu")
print("Running on CPU...")
model = model.to(device=device)

#Dataloader class to create standardized inputs tensor shape (1, 4, 224, 224) ie. 1 Tensor, 4 Channels, dimx, dimy
class ReachyDataset(Dataset):
    def __init__(self, rgb_img, depth_img):
        self.rgb_img = rgb_img
        self.depth_img = depth_img

    def __len__(self):
        return 1

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
        
    def __getitem__(self, index):
        x = self.numpy_to_torch(
            np.concatenate(
                (np.expand_dims(self.depth_img, 0),
                    self.rgb_img),
                0
            )
        )
        #x = self.numpy_to_torch(np.expand_dims(self.depth_img, 0))
        return x

def process_depth_image(
    depth,
    crop_size,
    out_size=224,
    return_mask=False,
    crop_y_offset=0,
    crop_x_offset=0,
):
    imh, imw = depth.shape

    # Crop.
    # crop square window of size crop_size x crop_size in the middle of the screen, with an offset of crop_y_offset to the top and crop_x_offset to the left

    y0 = max((imh - crop_size) // 2 - crop_y_offset, 0)
    y1 = min((imh - crop_size) // 2 + crop_size - crop_y_offset, imh)
    x0 = max((imw - crop_size) // 2 - crop_x_offset, 0)
    x1 = min((imw - crop_size) // 2 + crop_size - crop_x_offset, imw)

    depth_crop = depth[y0:y1, x0:x1]
    # depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    # Inpaint
    # OpenCV inpainting does weird things at the border.
    depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)
    # depth_nan_mask |= np.isinf(depth_crop).astype(np.uint8)
    # depth_nan_mask |= np.where(
    #     depth_crop >= 2.0, 1.0, 0.0).astype(np.uint8)

    depth_crop[depth_nan_mask == 1] = 0
    # humm we cut everything more than 1m
    depth_crop[depth_crop >= 1.0] = 1.0
    depth_crop[depth_crop == np.inf] = 1.0  # humm
    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    depth_scale = np.abs(depth_crop).max()
    depth_scale = depth_scale
    # print(f'scale: {depth_scale}')

    # Has to be float32, 64 not supported.
    depth_crop = depth_crop.astype(np.float32) / depth_scale

    depth_crop = cv2.inpaint(
        depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS
    )  # to avoid "holes" but mostly usefull for a top-down image

    # Back to original size and value range.
    depth_crop = depth_crop[1:-1, 1:-1]
    # should keep normalised?
    # depth_crop = depth_crop * depth_scale

    # Resize
    depth_crop = cv2.resize(depth_crop, (out_size, out_size), cv2.INTER_AREA)

    # self.depth_pub.publish(self.cv_bridge.cv2_to_imgmsg(
    #     depth_crop, encoding=self.current_depth_msg.encoding))

    if return_mask:
        depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
        depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), cv2.INTER_NEAREST)
        return depth_crop, depth_nan_mask
    else:
        return depth_crop


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

last_frame = None
#fig = plt.figure(figsize=(10, 10))
fig = plt.figure()

save_index = 0

while True:
    t0 = time.time()
    #Ensure that the output will not be null

    while image_getter.current_color_img is None or image_getter.current_depth is None or np.array_equal(image_getter.current_color_img,last_frame):
        image_getter.update_image()

    rgb_img = image_getter.current_color_img
    
    
    last_frame = rgb_img
    
    height, width, _ = rgb_img.shape
    print("AVANT",np.shape(rgb_img))
    start_x = int((width - 480) / 2)
    start_y = int((height - 480) / 2)
    end_x = start_x + 480
    end_y = start_y + 480
    
    rgb_img = rgb_img[start_y:end_y, start_x:end_x]
    
    
    print("APRES",np.shape(rgb_img))
    rgb_img = resize1(rgb_img, (224, 224))
    save_rgb = rgb_img
    save_rgb2 = save_rgb
    rgb_img = normalise(rgb_img)
    rgb_img = rgb_img.transpose((2, 0, 1))

    depth_img = image_getter.current_depth

    depth_img = depth_img[start_y:end_y, start_x:end_x]

    #Pre-process Depth camera output, replace inf by max, remove NaN, fill holes
    #== Remove Inf==
    '''max_depth = np.max(depth_img[np.isfinite(depth_img)])
    depth_img[np.isinf(depth_img)] = max_depth
    '''
    save_depth = resize1(depth_img, (224,224))
    depth_img = depth_normalize(depth_img)
    depth_img = resize1(depth_img, (224,224))
    depth_img = process_depth_image(depth=depth_img, crop_size=48)

    
    mask = np.zeros((depth_img.shape[0], depth_img.shape[1]), np.uint8)
    mask[100:mask.shape[0]-100,100:mask.shape[1]-100] = 255

    save_rgb = cv2.bitwise_and(save_rgb, save_rgb, mask=mask)
    print(mask.shape)
    print(save_rgb.shape)

    depth_img = cv2.bitwise_and(depth_img, depth_img, mask=mask)

    rgb_img = cv2.bitwise_and(rgb_img.transpose(), rgb_img.transpose(), mask=mask)
    rgb_img = rgb_img.transpose()
    #Init dataloader
    reachyDB = ReachyDataset(rgb_img=rgb_img, depth_img=depth_img)

    val_data = torch.utils.data.DataLoader(
        reachyDB,
        batch_size=1,
        shuffle=True,
        num_workers=1
    )
    
    for index, img in enumerate(val_data):
        #Inference
        output = model(img)
        q_img, ang_img, width_img = post_process_output(output[0], output[1], output[2], output[3])
        gs_l = detect_grasps(q_img, ang_img, width_img, no_grasps=3)
    
    ax = fig.add_subplot(1, 1, index+1)
    ax.imshow(save_rgb2, alpha=0.5)
    ax.imshow(q_img, alpha=0.5)
    ax.axis('off')
    ax.set_title("Rgb Image")
    for g in gs_l:
        g.plot(ax)
    fig.canvas.draw()
  
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    if image_getter.current_color_img is not None:
            '''for g in gs_l:
                cv2.circle(save_rgb, g.center, radius=0, color=(0,0,255), thickness=-1)'''
            cv2.imshow('D401-RGB', img)

    if cv2.waitKey(32) == ord(' '):
        print("Saved")
        cv2.imwrite("./output/"+str(save_index)+".png", img)
        save_index += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(f'FPS : {(1/(time.time() - t0)):.2f}.')

rclpy.shutdown()