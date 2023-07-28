import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from visualization_msgs.msg import Marker

''' Publisher class to create a single marker on the simulation with a fixed id (in this case id=1)'''
class MarkerPublisher(Node):
    def __init__(self, posx, posy, posz):
        super().__init__('marker_publisher')
        self.publisher_ = self.create_publisher(Marker, 'visualization_marker', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.msg_count = 0
        self.posx = posx
        self.posy = posy
        self.posz = posz
        self.achieved = False

    def timer_callback(self):
        marker = Marker()
        marker.header.frame_id = "depth_camera_optical"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = self.posx
        marker.pose.position.y = self.posy
        marker.pose.position.z = self.posz

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0

        self.publisher_.publish(marker)

        self.msg_count += 1
        #self.get_logger().info("Published message" +  str(self.msg_count)) #De-comment if you want the info

    def remove_marker(self):
        # Création du marqueur de suppression
        marker = Marker()
        marker.header.frame_id = 'world'
        marker.id = 1  # ID du marqueur à supprimer
        marker.action = Marker.DELETE

        # Publication du marqueur de suppression
        self.publisher_.publish(marker)
        self.get_logger().info('Marqueur supprimé')

''' Launch this method to initialize your own marker in the needed coordinates '''
def launchNode(posx, posy, posz):
    if(not rclpy.ok()):
        rclpy.init()
    try:
        marker_publisher = MarkerPublisher(posx, posy, posz)
        executor = SingleThreadedExecutor()
        executor.add_node(marker_publisher)
        try:
            executor.spin()
        finally:
            executor.shutdown()
            marker_publisher.destroy_node()
    finally:
        rclpy.shutdown()

'''  Test function to set a marker in the simulation '''
def main(args=None):
    rclpy.init(args=args)

    marker_publisher = MarkerPublisher(1.0,1.0,1.0)
    marker_publisher.achieved = True

    rclpy.spin(marker_publisher)
    marker_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
