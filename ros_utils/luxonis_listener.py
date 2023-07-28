import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
'''
Realsense D401 subscriber test
'''
class LuxonisSubscriber(
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
            '/oak/stereo/image_raw',
            self.on_image_depth,
            10,
        )

        self.current_color_img = None
        self.color_img_sub = self.create_subscription(
            Image,
            '/oak/rgb/image_raw',
            self.on_update_color,
            10,
        )

        self.cam_img = None

    def on_image_depth(self, msg):
        bridge = CvBridge()
        self.current_depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
    def on_update_color(self, msg):
        '''self.current_color_img = msg
        #self.current_color_img = self.cv_bridge.imgmsg_to_cv2(msg)
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.current_color_img = cv.imdecode(np_arr, cv.IMREAD_COLOR)'''
        bridge = CvBridge()
        self.current_color_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        #self.current_color_img = cv.cvtColor(self.current_color_img, cv.COLOR_BGR2RGB) #Convert BGR to RGB...


    def update_image(self):
        """Get the last image by spinning the node."""
        rclpy.spin_once(self)

if __name__ == "__main__":
    rclpy.init()

    rs_image = LuxonisSubscriber(node_name="d401_images")
    
    while True:
        rs_image.update_image()
        if rs_image.current_color_img is not None:
            cv.imshow('LUXONIS-RGB', rs_image.current_color_img)
        if rs_image.current_depth is not None:
            cv.imshow('LUXONIS-DEPTH', rs_image.current_depth)
        if cv.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()
            break