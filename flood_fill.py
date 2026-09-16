from collections import deque
import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped

from .grid_localizer import (
    pose_to_cell,
    cell_to_pose,
    distance_to_cell_center,
)


# ============================================================
# MAZE CONFIGURATION
# ============================================================

ROWS = 6
COLS = 6

START = (0, 0)
GOAL = (5, 5)

CELL_SIZE = 1.0

directions = {
    'N': (1, 0),
    'E': (0, 1),
    'S': (-1, 0),
    'W': (0, -1),
}

OPPOSITE = {
    'N': 'S',
    'E': 'W',
    'S': 'N',
    'W': 'E',
}


# ============================================================
# FLOOD FILL
# ============================================================

def flood_fill(wall_map):
    """
    Calculate the Flood Fill distance map.

    wall_map format:

        {
            (row, col): {
                'N': True/False/None,
                'E': True/False/None,
                'S': True/False/None,
                'W': True/False/None
            }
        }

    True  = wall
    False = known open
    None  = unknown

    Unknown directions are treated as OPEN for planning.
    This allows exploration of an initially unknown maze.
    """

    distance = [
        [999 for _ in range(COLS)]
        for _ in range(ROWS)
    ]

    distance[GOAL[0]][GOAL[1]] = 0

    queue = deque()
    queue.append(GOAL)

    while queue:

        row, col = queue.popleft()

        for direction, (dr, dc) in directions.items():

            new_row = row + dr
            new_col = col + dc

            if new_row < 0 or new_row >= ROWS:
                continue

            if new_col < 0 or new_col >= COLS:
                continue

            # The edge between the current cell and neighbor
            # must not be a known wall.
            cell_walls = wall_map.get(
                (row, col),
                {'N': None, 'E': None, 'S': None, 'W': None}
            )

            if cell_walls.get(direction) is True:
                continue

            new_distance = distance[row][col] + 1

            if new_distance < distance[new_row][new_col]:

                distance[new_row][new_col] = new_distance
                queue.append((new_row, new_col))

    return distance


def choose_next_direction(row, col, distance, wall_map):
    """
    Select the neighboring cell with the smallest Flood Fill value.

    Unknown walls are treated as potentially open.
    """

    best_direction = None
    best_value = 999

    # Deterministic tie-breaking.
    direction_order = ['N', 'E', 'S', 'W']

    cell_walls = wall_map.get(
        (row, col),
        {'N': None, 'E': None, 'S': None, 'W': None}
    )

    for direction in direction_order:

        dr, dc = directions[direction]

        new_row = row + dr
        new_col = col + dc

        if new_row < 0 or new_row >= ROWS:
            continue

        if new_col < 0 or new_col >= COLS:
            continue

        if cell_walls.get(direction) is True:
            continue

        neighbor_value = distance[new_row][new_col]

        if neighbor_value < best_value:

            best_value = neighbor_value
            best_direction = direction

    return best_direction


# ============================================================
# ROBOT
# ============================================================

class MazeRobot(Node):

    def __init__(self):

        super().__init__('maze_robot')

        # ========================================================
        # ODOMETRY
        # ========================================================

        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        self.odom_sub = self.create_subscription(
            Odometry,
            '/wheel_odom',
            self.odom_callback,
            10
        )

        # ========================================================
        # LIDAR
        # ========================================================

        self.front_distance = float('inf')
        self.right_distance = float('inf')
        self.rear_distance = float('inf')
        self.left_distance = float('inf')

        self.lidar_ready = False

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # ========================================================
        # CMD VEL
        # ========================================================

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/diff_drive_base_controller/cmd_vel',
            10
        )

        # ========================================================
        # GRID STATE
        # ========================================================

        # Last confirmed maze cell.
        self.current_row = START[0]
        self.current_col = START[1]

        # Continuous physical localization.
        self.actual_row = START[0]
        self.actual_col = START[1]

        self.heading = 'N'

        # ========================================================
        # TARGET
        # ========================================================

        self.target_direction = None

        self.target_row = None
        self.target_col = None

        self.target_x = None
        self.target_y = None
        self.target_yaw = None

        # ========================================================
        # EXIT
        # ========================================================

        self.exit_target_x = None
        self.exit_target_y = None
        self.exit_target_yaw = None

        # ========================================================
        # DISCOVERED MAZE MAP
        # ========================================================

        # None  = unknown
        # True  = wall
        # False = open

        self.discovered_walls = {}

        for row in range(ROWS):
            for col in range(COLS):

                self.discovered_walls[(row, col)] = {
                    'N': None,
                    'E': None,
                    'S': None,
                    'W': None,
                }

        # Add the permanent outer boundaries.
        self.initialize_boundaries()

        # ========================================================
        # FLOOD FILL
        # ========================================================

        self.distance = flood_fill(
            self.discovered_walls
        )

        # ========================================================
        # MOVEMENT PARAMETERS
        # ========================================================

        self.linear_speed = 0.30
        self.angular_speed = 0.90

        self.max_steering = 0.25

        self.position_tolerance = 0.08
        self.heading_tolerance = math.radians(3.0)

        # ========================================================
        # LIDAR PARAMETERS
        # ========================================================

        # A wall immediately adjacent to the current cell
        # is approximately 0.5 m from the robot center.
        #
        # 0.60 m gives some tolerance while still allowing
        # an open neighboring cell to be distinguished.
        self.wall_threshold = 0.60

        # If the robot is already driving toward a direction
        # and LiDAR suddenly sees an obstacle closer than this,
        # treat it as a newly discovered wall.
        self.obstacle_threshold = 0.35

        # ========================================================
        # STATE MACHINE
        # ========================================================

        self.state = 'WAIT_FOR_ODOM'

        # ========================================================
        # CONTROL LOOP
        # ========================================================

        self.timer = self.create_timer(
            0.05,
            self.control_loop
        )

        self.get_logger().info(
            'Maze robot started. Waiting for wheel odometry...'
        )


    # ============================================================
    # INITIALIZE OUTER BOUNDARIES
    # ============================================================

    def initialize_boundaries(self):

        for row in range(ROWS):

            # West boundary
            self.discovered_walls[(row, 0)]['W'] = True

            # East boundary
            self.discovered_walls[(row, COLS - 1)]['E'] = True

        for col in range(COLS):

            # South boundary.
            #
            # Keep the starting entrance open at R0C0.
            if col != 0:
                self.discovered_walls[(0, col)]['S'] = True

            # North boundary.
            #
            # Keep the maze exit open at R5C5.
            if col != COLS - 1:
                self.discovered_walls[(ROWS - 1, col)]['N'] = True


    # ============================================================
    # ODOMETRY CALLBACK
    # ============================================================

    def odom_callback(self, msg):

        self.robot_x = float(
            msg.pose.pose.position.x
        )

        self.robot_y = float(
            msg.pose.pose.position.y
        )

        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cosy_cosp = 1.0 - 2.0 * (
            q.y * q.y +
            q.z * q.z
        )

        self.robot_yaw = math.atan2(
            siny_cosp,
            cosy_cosp
        )


    # ============================================================
    # LIDAR CALLBACK
    # ============================================================

    def scan_callback(self, msg):

        if not msg.ranges:
            return

        def get_sector_min(start_deg, end_deg):

            start_rad = math.radians(start_deg)
            end_rad = math.radians(end_deg)

            minimum = float('inf')

            for i, distance in enumerate(msg.ranges):

                angle = (
                    msg.angle_min +
                    i * msg.angle_increment
                )

                # Normalize to [-pi, pi]
                angle = math.atan2(
                    math.sin(angle),
                    math.cos(angle)
                )

                if start_rad <= angle <= end_rad:

                    if (
                        not math.isnan(distance)
                        and not math.isinf(distance)
                        and msg.range_min <= distance <= msg.range_max
                    ):

                        minimum = min(
                            minimum,
                            distance
                        )

            return minimum

        # Robot frame:
        #
        #             FRONT
        #               0°
        #                ↑
        #
        # LEFT  +90° ← ROBOT → -90° RIGHT
        #
        #               ↓
        #             REAR ±180°

        self.front_distance = get_sector_min(
            -15,
            15
        )

        self.right_distance = get_sector_min(
            -105,
            -75
        )

        self.rear_distance = min(
            get_sector_min(165, 180),
            get_sector_min(-180, -165)
        )

        self.left_distance = get_sector_min(
            75,
            105
        )

        self.lidar_ready = True


    # ============================================================
    # MAP HELPERS
    # ============================================================

    def set_wall(self, row, col, direction, is_wall):

        if row < 0 or row >= ROWS:
            return

        if col < 0 or col >= COLS:
            return

        self.discovered_walls[(row, col)][direction] = is_wall

        dr, dc = directions[direction]

        neighbor_row = row + dr
        neighbor_col = col + dc

        # Keep the neighboring cell consistent.
        if (
            0 <= neighbor_row < ROWS
            and 0 <= neighbor_col < COLS
        ):

            opposite = OPPOSITE[direction]

            self.discovered_walls[
                (neighbor_row, neighbor_col)
            ][opposite] = is_wall


    def relative_to_world_direction(
        self,
        relative_direction
    ):
        """
        Convert LiDAR robot-frame direction into
        a world/grid direction.

        Robot heading:
            N -> front=N
            E -> front=E
            S -> front=S
            W -> front=W
        """

        heading_order = ['N', 'E', 'S', 'W']

        heading_index = heading_order.index(
            self.heading
        )

        relative_offsets = {
            'FRONT': 0,
            'RIGHT': 1,
            'REAR': 2,
            'LEFT': 3,
        }

        index = (
            heading_index +
            relative_offsets[relative_direction]
        ) % 4

        return heading_order[index]


    def update_map_from_lidar(self):

        if not self.lidar_ready:
            return False

        row = self.current_row
        col = self.current_col

        measurements = {
            'FRONT': self.front_distance,
            'RIGHT': self.right_distance,
            'REAR': self.rear_distance,
            'LEFT': self.left_distance,
        }

        changed = False

        for relative_direction, distance in measurements.items():

            world_direction = (
                self.relative_to_world_direction(
                    relative_direction
                )
            )

            if math.isinf(distance):

                # No return.
                # Do not automatically classify as open,
                # because the LiDAR may simply have no valid
                # measurement in that sector.
                continue

            if distance <= self.wall_threshold:

                is_wall = True

            else:

                is_wall = False

            previous = self.discovered_walls[
                (row, col)
            ][world_direction]

            if previous != is_wall:

                self.set_wall(
                    row,
                    col,
                    world_direction,
                    is_wall
                )

                changed = True

        return changed


    # ============================================================
    # PRINT CURRENT CELL MAP
    # ============================================================

    def log_current_cell_map(self):

        cell = self.discovered_walls[
            (self.current_row, self.current_col)
        ]

        self.get_logger().info(
            f'Cell R{self.current_row}C{self.current_col} '
            f'| N={cell["N"]} '
            f'E={cell["E"]} '
            f'S={cell["S"]} '
            f'W={cell["W"]}'
        )


    # ============================================================
    # PRINT FLOOD FILL MAP
    # ============================================================

    def log_distance_map(self):

        self.get_logger().info(
            'Flood Fill distance map:'
        )

        for row in reversed(range(ROWS)):

            values = ' '.join(
                f'{self.distance[row][col]:2d}'
                if self.distance[row][col] < 999
                else 'XX'
                for col in range(COLS)
            )

            self.get_logger().info(
                values
            )


    # ============================================================
    # PUBLISH COMMAND
    # ============================================================

    def publish_cmd(self, linear_x, angular_z):

        msg = TwistStamped()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.twist.linear.x = float(linear_x)
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0

        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = float(angular_z)

        self.cmd_pub.publish(msg)


    # ============================================================
    # STOP
    # ============================================================

    def stop_robot(self):

        self.publish_cmd(
            0.0,
            0.0
        )


    # ============================================================
    # MAIN CONTROL LOOP
    # ============================================================

    def control_loop(self):

        # ========================================================
        # WAIT FOR ODOMETRY
        # ========================================================

        if (
            self.robot_x is None
            or self.robot_y is None
            or self.robot_yaw is None
        ):

            if self.state != 'WAIT_FOR_ODOM':
                self.stop_robot()

            return


        # ========================================================
        # CONTINUOUS PHYSICAL LOCALIZATION
        # ========================================================

        self.actual_row, self.actual_col = pose_to_cell(
            self.robot_x,
            self.robot_y
        )


        # ========================================================
        # WAIT FOR FIRST ODOMETRY
        # ========================================================

        if self.state == 'WAIT_FOR_ODOM':

            self.heading = self.yaw_to_direction(
                self.robot_yaw
            )

            self.get_logger().info(
                f'Odometry received: '
                f'x={self.robot_x:.3f}, '
                f'y={self.robot_y:.3f}, '
                f'yaw={math.degrees(self.robot_yaw):.1f}°'
            )

            self.get_logger().info(
                f'Current cell: '
                f'R{self.current_row}C{self.current_col}, '
                f'heading={self.heading}'
            )

            self.state = 'SENSE'

            return


        # ========================================================
        # SENSE / MAP
        # ========================================================

        elif self.state == 'SENSE':

            if not self.lidar_ready:

                self.stop_robot()
                return

            map_changed = (
                self.update_map_from_lidar()
            )

            self.get_logger().info(
                f'Sensing cell '
                f'R{self.current_row}C{self.current_col}'
            )

            self.log_current_cell_map()

            if map_changed:

                self.get_logger().info(
                    'Maze map updated from LiDAR.'
                )

            self.state = 'PLAN'

            return


        # ========================================================
        # PLAN
        # ========================================================

        elif self.state == 'PLAN':

            # ----------------------------------------------------
            # FINAL CELL INSIDE MAZE
            # ----------------------------------------------------

            if (
                self.current_row == GOAL[0]
                and self.current_col == GOAL[1]
            ):

                self.stop_robot()

                self.get_logger().info(
                    'Reached final maze cell R5C5.'
                )

                # One cell north of R5C5.
                self.exit_target_x = 2.5
                self.exit_target_y = 3.5
                self.exit_target_yaw = (
                    self.direction_to_yaw('N')
                )

                self.target_yaw = (
                    self.exit_target_yaw
                )

                self.target_direction = 'N'

                self.state = 'EXIT_TURN'

                return


            # ----------------------------------------------------
            # RUN FLOOD FILL
            # ----------------------------------------------------

            self.distance = flood_fill(
                self.discovered_walls
            )

            self.log_distance_map()


            # ----------------------------------------------------
            # CHOOSE NEXT DIRECTION
            # ----------------------------------------------------

            self.target_direction = (
                choose_next_direction(
                    self.current_row,
                    self.current_col,
                    self.distance,
                    self.discovered_walls
                )
            )


            if self.target_direction is None:

                self.get_logger().error(
                    'Flood Fill found no valid direction!'
                )

                self.stop_robot()

                self.state = 'DONE'

                return


            # ----------------------------------------------------
            # TARGET CELL
            # ----------------------------------------------------

            dr, dc = directions[
                self.target_direction
            ]

            self.target_row = (
                self.current_row + dr
            )

            self.target_col = (
                self.current_col + dc
            )


            # ----------------------------------------------------
            # TARGET PHYSICAL POSITION
            # ----------------------------------------------------

            self.target_x, self.target_y = (
                cell_to_pose(
                    self.target_row,
                    self.target_col
                )
            )

            self.target_yaw = (
                self.direction_to_yaw(
                    self.target_direction
                )
            )


            self.get_logger().info(
                f'Flood Fill: '
                f'R{self.current_row}C{self.current_col} '
                f'-> '
                f'R{self.target_row}C{self.target_col} '
                f'({self.target_direction})'
            )

            self.get_logger().info(
                f'Target center: '
                f'x={self.target_x:.2f}, '
                f'y={self.target_y:.2f}'
            )


            # ----------------------------------------------------
            # DETERMINE WHETHER TURN IS REQUIRED
            # ----------------------------------------------------

            heading_error = (
                self.angle_difference(
                    self.target_yaw,
                    self.robot_yaw
                )
            )

            if abs(heading_error) <= self.heading_tolerance:

                self.heading = (
                    self.target_direction
                )

                self.state = 'DRIVE'

            else:

                self.state = 'TURN'

            return


        # ========================================================
        # TURN
        # ========================================================

        elif self.state == 'TURN':

            error = self.angle_difference(
                self.target_yaw,
                self.robot_yaw
            )

            if abs(error) <= self.heading_tolerance:

                self.stop_robot()

                self.heading = (
                    self.target_direction
                )

                self.get_logger().info(
                    f'Turn complete. '
                    f'Heading={self.heading} | '
                    f'Yaw={math.degrees(self.robot_yaw):.1f}°'
                )

                self.state = 'DRIVE'

                return


            turn_speed = 2.0 * error

            turn_speed = max(
                -self.angular_speed,
                min(
                    self.angular_speed,
                    turn_speed
                )
            )

            self.publish_cmd(
                0.0,
                turn_speed
            )

            return


        # ========================================================
        # DRIVE
        # ========================================================

        elif self.state == 'DRIVE':

            # ----------------------------------------------------
            # CHECK TARGET DISTANCE
            # ----------------------------------------------------

            distance_to_target = (
                distance_to_cell_center(
                    self.robot_x,
                    self.robot_y,
                    self.target_row,
                    self.target_col
                )
            )


            # ----------------------------------------------------
            # TARGET REACHED
            # ----------------------------------------------------

            if (
                distance_to_target
                <= self.position_tolerance
            ):

                self.stop_robot()

                self.current_row = (
                    self.target_row
                )

                self.current_col = (
                    self.target_col
                )

                self.heading = (
                    self.target_direction
                )

                self.get_logger().info(
                    f'ARRIVED at '
                    f'R{self.current_row}C{self.current_col}'
                )

                self.get_logger().info(
                    f'Actual pose: '
                    f'x={self.robot_x:.3f}, '
                    f'y={self.robot_y:.3f}'
                )

                self.state = 'SENSE'

                return


            # ----------------------------------------------------
            # PHYSICAL OBSTACLE DETECTION
            # ----------------------------------------------------

            if (
                self.lidar_ready
                and self.front_distance
                < self.obstacle_threshold
            ):

                self.stop_robot()

                self.get_logger().warn(
                    f'New obstacle detected while driving '
                    f'toward {self.target_direction}. '
                    f'Front={self.front_distance:.2f} m'
                )

                # The robot is trying to move from the current
                # confirmed cell toward target_direction.
                #
                # Therefore a close object directly ahead is
                # recorded as a wall in that direction.

                self.set_wall(
                    self.current_row,
                    self.current_col,
                    self.target_direction,
                    True
                )

                self.get_logger().warn(
                    f'Updated map: '
                    f'R{self.current_row}C{self.current_col} '
                    f'{self.target_direction}=WALL'
                )

                self.state = 'SENSE'

                return


            # ----------------------------------------------------
            # HEADING CORRECTION
            # ----------------------------------------------------

            heading_error = (
                self.angle_difference(
                    self.target_yaw,
                    self.robot_yaw
                )
            )

            correction = (
                1.5 * heading_error
            )

            correction = max(
                -self.max_steering,
                min(
                    self.max_steering,
                    correction
                )
            )


            # ----------------------------------------------------
            # DRIVE
            # ----------------------------------------------------

            self.publish_cmd(
                self.linear_speed,
                correction
            )

            return


        # ========================================================
        # EXIT TURN
        # ========================================================

        elif self.state == 'EXIT_TURN':

            error = self.angle_difference(
                self.exit_target_yaw,
                self.robot_yaw
            )

            if abs(error) <= self.heading_tolerance:

                self.stop_robot()

                self.get_logger().info(
                    'Aligned north at R5C5. '
                    'Driving through maze exit toward R6C5.'
                )

                self.state = 'EXIT_DRIVE'

                return


            turn_speed = 2.0 * error

            turn_speed = max(
                -self.angular_speed,
                min(
                    self.angular_speed,
                    turn_speed
                )
            )

            self.publish_cmd(
                0.0,
                turn_speed
            )

            return


        # ========================================================
        # EXIT DRIVE
        # ========================================================

        elif self.state == 'EXIT_DRIVE':

            distance_to_exit = math.sqrt(
                (
                    self.robot_x -
                    self.exit_target_x
                ) ** 2
                +
                (
                    self.robot_y -
                    self.exit_target_y
                ) ** 2
            )

            if (
                distance_to_exit
                <= self.position_tolerance
            ):

                self.stop_robot()

                self.get_logger().info(
                    '========================================'
                )

                self.get_logger().info(
                    'EXIT REACHED: R6C5'
                )

                self.get_logger().info(
                    'ROBOT IS OUTSIDE THE MAZE'
                )

                self.get_logger().info(
                    '========================================'
                )

                self.state = 'DONE'

                return


            heading_error = (
                self.angle_difference(
                    self.exit_target_yaw,
                    self.robot_yaw
                )
            )

            correction = (
                1.5 * heading_error
            )

            correction = max(
                -self.max_steering,
                min(
                    self.max_steering,
                    correction
                )
            )

            self.publish_cmd(
                self.linear_speed,
                correction
            )

            return


        # ========================================================
        # DONE
        # ========================================================

        elif self.state == 'DONE':

            self.stop_robot()

            return


    # ============================================================
    # DIRECTION → YAW
    # ============================================================

    def direction_to_yaw(self, direction):

        if direction == 'E':
            return 0.0

        if direction == 'N':
            return math.pi / 2.0

        if direction == 'W':
            return math.pi

        if direction == 'S':
            return -math.pi / 2.0

        return 0.0


    # ============================================================
    # YAW → DIRECTION
    # ============================================================

    def yaw_to_direction(self, yaw):

        directions_yaw = {
            'E': 0.0,
            'N': math.pi / 2.0,
            'W': math.pi,
            'S': -math.pi / 2.0,
        }

        best_direction = 'E'
        smallest_error = float('inf')

        for direction, target_yaw in directions_yaw.items():

            error = self.angle_difference(
                target_yaw,
                yaw
            )

            if abs(error) < smallest_error:

                smallest_error = abs(error)
                best_direction = direction

        return best_direction


    # ============================================================
    # ANGLE DIFFERENCE
    # ============================================================

    def angle_difference(self, target, current):

        error = target - current

        while error > math.pi:
            error -= 2.0 * math.pi

        while error < -math.pi:
            error += 2.0 * math.pi

        return error


# ================================================================
# MAIN
# ================================================================

def main(args=None):

    rclpy.init(args=args)

    node = MazeRobot()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.stop_robot()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()