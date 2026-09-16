import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion


class WheelOdometry(Node):

    def __init__(self):
        super().__init__('wheel_odometry')

        # ============================================================
        # ROBOT PARAMETERS
        # ============================================================

        self.wheel_radius = 0.033
        self.wheel_separation = 0.40

        self.left_wheel_name = 'left_wheel_joint'
        self.right_wheel_name = 'right_wheel_joint'

        # ============================================================
        # ODOMETRY STATE
        # ============================================================

        self.x = -2.5
        self.y = -2.5
        self.yaw = math.pi / 2.0

        self.previous_left_position = None
        self.previous_right_position = None

        self.last_time = None

        self.joint_state_received = False

        # ============================================================
        # /joint_states
        # ============================================================

        joint_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            joint_qos
        )

        # ============================================================
        # /wheel_odom
        # ============================================================

        self.odom_pub = self.create_publisher(
            Odometry,
            '/wheel_odom',
            10
        )

        # ============================================================
        # UPDATE LOOP
        # ============================================================

        self.timer = self.create_timer(
            0.02,       # 50 Hz
            self.update_odometry
        )

        self.get_logger().info(
            'Wheel odometry node started.'
        )

    # ================================================================
    # JOINT STATE CALLBACK
    # ================================================================

    def joint_state_callback(self, msg):

        left_position = None
        right_position = None

        for i, name in enumerate(msg.name):

            if name == self.left_wheel_name:
                left_position = msg.position[i]

            elif name == self.right_wheel_name:
                right_position = msg.position[i]

        if left_position is None or right_position is None:
            return

        # Convert explicitly to float.
        self.current_left_position = float(left_position)
        self.current_right_position = float(right_position)

        self.joint_state_received = True

    # ================================================================
    # ODOMETRY UPDATE
    # ================================================================

    def update_odometry(self):

        if not self.joint_state_received:
            return

        current_time = self.get_clock().now()

        # ------------------------------------------------------------
        # First measurement
        # ------------------------------------------------------------

        if self.previous_left_position is None:

            self.previous_left_position = self.current_left_position
            self.previous_right_position = self.current_right_position
            self.last_time = current_time

            self.publish_odometry(
                current_time,
                0.0,
                0.0
            )

            return

        # ------------------------------------------------------------
        # Wheel angular displacement
        # ------------------------------------------------------------

        left_delta_theta = (
            self.current_left_position
            - self.previous_left_position
        )

        right_delta_theta = (
            self.current_right_position
            - self.previous_right_position
        )

        # ------------------------------------------------------------
        # Wheel linear displacement
        # ------------------------------------------------------------

        left_distance = (
            self.wheel_radius * left_delta_theta
        )

        right_distance = (
            self.wheel_radius * right_delta_theta
        )

        # ------------------------------------------------------------
        # Differential-drive motion
        # ------------------------------------------------------------

        center_distance = (
            left_distance + right_distance
        ) / 2.0

        delta_yaw = (
            right_distance - left_distance
        ) / self.wheel_separation

        # ------------------------------------------------------------
        # Integrate robot pose
        #
        # Use the midpoint heading for better integration.
        # ------------------------------------------------------------

        midpoint_yaw = self.yaw + (delta_yaw / 2.0)

        self.x += center_distance * math.cos(midpoint_yaw)
        self.y += center_distance * math.sin(midpoint_yaw)

        self.yaw += delta_yaw

        self.yaw = self.normalize_angle(self.yaw)

        # ------------------------------------------------------------
        # Velocity
        # ------------------------------------------------------------

        now_seconds = current_time.nanoseconds / 1e9
        previous_seconds = self.last_time.nanoseconds / 1e9

        dt = now_seconds - previous_seconds

        if dt > 0.0:

            linear_velocity = (
                center_distance / dt
            )

            angular_velocity = (
                delta_yaw / dt
            )

        else:

            linear_velocity = 0.0
            angular_velocity = 0.0

        # ------------------------------------------------------------
        # Save current wheel positions
        # ------------------------------------------------------------

        self.previous_left_position = (
            self.current_left_position
        )

        self.previous_right_position = (
            self.current_right_position
        )

        self.last_time = current_time

        # ------------------------------------------------------------
        # Publish
        # ------------------------------------------------------------

        self.publish_odometry(
            current_time,
            linear_velocity,
            angular_velocity
        )

    # ================================================================
    # PUBLISH ODOMETRY
    # ================================================================

    def publish_odometry(
        self,
        current_time,
        linear_velocity,
        angular_velocity
    ):

        msg = Odometry()

        # ------------------------------------------------------------
        # Header
        # ------------------------------------------------------------

        msg.header.stamp = current_time.to_msg()
        msg.header.frame_id = 'odom'

        # ------------------------------------------------------------
        # Child frame
        # ------------------------------------------------------------

        msg.child_frame_id = 'base_link'

        # ------------------------------------------------------------
        # Position
        # ------------------------------------------------------------

        msg.pose.pose.position.x = float(self.x)
        msg.pose.pose.position.y = float(self.y)
        msg.pose.pose.position.z = 0.0

        # ------------------------------------------------------------
        # Orientation
        # ------------------------------------------------------------

        q = self.yaw_to_quaternion(self.yaw)

        msg.pose.pose.orientation = q

        # ------------------------------------------------------------
        # Linear velocity
        # ------------------------------------------------------------

        msg.twist.twist.linear.x = float(linear_velocity)
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.linear.z = 0.0

        # ------------------------------------------------------------
        # Angular velocity
        # ------------------------------------------------------------

        msg.twist.twist.angular.x = 0.0
        msg.twist.twist.angular.y = 0.0
        msg.twist.twist.angular.z = float(angular_velocity)

        # ------------------------------------------------------------
        # Covariance
        #
        # These are wheel-odometry estimates, so uncertainty exists.
        # ------------------------------------------------------------

        msg.pose.covariance[0] = 0.001
        msg.pose.covariance[7] = 0.001
        msg.pose.covariance[35] = 0.01

        msg.twist.covariance[0] = 0.001
        msg.twist.covariance[7] = 0.001
        msg.twist.covariance[35] = 0.01

        self.odom_pub.publish(msg)

    # ================================================================
    # ANGLE HELPERS
    # ================================================================

    @staticmethod
    def normalize_angle(angle):

        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    @staticmethod
    def yaw_to_quaternion(yaw):

        q = Quaternion()

        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)

        return q


def main(args=None):

    rclpy.init(args=args)

    node = WheelOdometry()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()