import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
'''
Realsense D401 subscriber test
'''
class RealsenseD401Subscriber(
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
            '/depth/image_rect_raw',
            self.on_image_depth,
            10,
        )

        self.current_color_img = None
        self.color_img_sub = self.create_subscription(
            Image,
            '/color/image_rect_raw',
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
        self.current_color_img = cv.cvtColor(self.current_color_img, cv.COLOR_BGR2RGB) #Convert BGR to RGB...


    def update_image(self):
        """Get the last image by spinning the node."""
        rclpy.spin_once(self)

class RealsenseD401Infos(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name=node_name)

        self.depth_cam_info = self.create_subscription(
            CameraInfo,
            '/color/camera_info',
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

if __name__ == "__main__":
    rclpy.init()
    rs = RealsenseD401Infos(node_name="d401_infos")
    rclpy.spin_once(rs)
    print(rs.fx)

    rs_image = RealsenseD401Subscriber(node_name="d401_images")
    
    while True:
        rs_image.update_image()
        if rs_image.current_color_img is not None:
            cv.imshow('D401-RGB', rs_image.current_color_img)
        if rs_image.current_depth is not None:
            cv.imshow('D401-DEPTH', rs_image.current_depth)
        if cv.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()
            break
