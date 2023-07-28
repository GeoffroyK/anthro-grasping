import asyncio
import logging
import os
import numpy as np
import importlib
import threading

from google.protobuf.wrappers_pb2 import FloatValue

from reachy_sdk_api.realsense_pb2 import PointCloud, Point, PointCloudMessage

"""Listen to transform published in ROS2 by robot_state_publisher."""
import rclpy
from rclpy.node import Node

from tf2_ros import TransformException, TransformStamped
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from scipy.spatial.transform import Rotation as R


class FrameListener(Node):
    """Frame listener class.

    Listen to the tf published by ROS2 between source_frame and target_frame.
    """
    def __init__(self, target_frame: str, source_frame: str):
        super().__init__('tf2_listener')

        self.target_frame = target_frame
        self.source_frame = source_frame

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.tf_stamped = TransformStamped()

        self.timer = self.create_timer(0.1, self._on_timer)

    def poseMatrixFromTransform(self, transform):

            x = transform.translation.x
            y = transform.translation.y
            z = transform.translation.z

            qx = transform.rotation.x
            qy = transform.rotation.y
            qz = transform.rotation.z
            qw = transform.rotation.w

            rot_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()

            T = np.eye(4)
            T[:3, :3] = rot_mat
            T[:3, 3] = [x, y, z]

            return T

    def _on_timer(self):
        try:
            now = rclpy.time.Time()
            self.tf_stamped = self.tf_buffer.lookup_transform(
                target_frame=self.target_frame,
                source_frame=self.source_frame,
                time=now)

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.target_frame} to {self.source_frame}: {ex}')
            return

   

if __name__ == "__main__":
    rclpy.init()

    tf_head = FrameListener("torso", "head_tip")
    '''
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(tf_head)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    
    executor_thread.start()
    '''
    rclpy.spin(tf_head)

    tf = tf_head.tf_stamped
    
    transform = poseMatrixFromTransform(tf._transform)
    print(transform)