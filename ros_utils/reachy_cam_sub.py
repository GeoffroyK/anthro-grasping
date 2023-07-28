import rclpy
from rclpy.node import Node
from sensor_msgs.msg._compressed_image import CompressedImage
import cv2 as cv

class RosCameraSubscriber(
                Node,
                ):
    """ROS node subscribing to the image topics."""

    def __init__(self, node_name: str, side: str) -> None:
        """Set up the node.

        Subscribe to the requested image topic (either /left_image/image_raw/compressed or /right_image/image_raw/compressed).
        """
        super().__init__(node_name=node_name)

        self.camera_sub = self.create_subscription(
            CompressedImage,
            side+'_image/image_raw/compressed',
            self.on_image_update,
            1,
        )

        self.cam_img = None

    def on_image_update(self, msg):
        """Get data from image. Callback for "/'side'_image "subscriber."""
        data = np.frombuffer(msg.data.tobytes(), dtype=np.uint8)
        self.cam_img = cv.imdecode(data, cv.IMREAD_COLOR)

    def update_image(self):
        """Get the last image by spinning the node."""
        rclpy.spin_once(self)