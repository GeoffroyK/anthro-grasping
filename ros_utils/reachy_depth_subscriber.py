from reachy_sdk import ReachySDK
import rclpy
import time
from rclpy.node import Node
from sensor_msgs.msg._compressed_image import CompressedImage
from sensor_msgs.msg import Image, CameraInfo
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge

class RosDepthCameraSubscriber(
                Node,
                ):
    """ROS node subscribing to the image topics."""

    def __init__(self, node_name: str) -> None:
        super().__init__(node_name=node_name)
        
        self.cv_bridge = CvBridge()

        self.current_depth = None
        self.processed_depth = None
        self.depth_img_sub = self.create_subscription(
            Image,
            '/depth_camera/depth/image_rect_raw',
            self.on_image_depth,
            10,
        )

        self.current_color_img = None
        self.color_img_sub = self.create_subscription(
            Image,
            '/depth_camera/color/image_raw',
            self.on_update_color,
            10,
        )

        self.cam_img = None

    def on_image_depth(self, msg):
        self.current_depth = msg
        self.current_depth = self.cv_bridge.imgmsg_to_cv2(msg)

    def on_update_color(self, msg):
        self.current_color_img = msg
        self.current_color_img = self.cv_bridge.imgmsg_to_cv2(msg)

    def update_image(self):
        """Get the last image by spinning the node."""
        rclpy.spin_once(self)

class DepthCameraInfo(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name=node_name)

        self.depth_cam_info = self.create_subscription(
            CameraInfo,
            '/depth_camera/depth/camera_info',
            self.cam_info_cb,
            10,
        )

    def cam_info_cb(self, msg):
        self.K = msg.k
        self.fx = self.K[0]
        self.cx = self.K[2]
        self.fy = self.K[4]
        self.cy = self.K[5]
        self.cam_info_done = True
        self.camera_frame = msg.header.frame_id

        self.crop_size = msg.height
        self.out_size = msg.height  # THE IMAGE SHOULD BE SQUARE!!!

        self.destroy_subscription(self.depth_cam_info)


if __name__ == '__main__':
    reachy = ReachySDK(host='localhost')

    rclpy.init()
    time.sleep(1)
    image_getter = RosDepthCameraSubscriber(node_name="image_viewer")
    
    tf1 = DepthCameraInfo(node_name="Touchepoamonposte")
    rclpy.spin_once(tf1)
    print(tf1.fx)
    '''
    while True:
        image_getter.update_image()
        if image_getter.current_depth is not None:
            cv.imshow(' camera uno', image_getter.current_depth)
        if image_getter.current_color_img is not None:
            cv.imshow(' camera dos', image_getter.current_color_img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    '''