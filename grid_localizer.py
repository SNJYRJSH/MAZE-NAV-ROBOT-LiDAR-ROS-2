import math


ROWS = 6
COLS = 6

CELL_SIZE = 1.0

# Cell centers
X_CENTERS = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
Y_CENTERS = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]


def pose_to_cell(x, y):
    """
    Convert world coordinates into the nearest maze cell.

    Returns:
        (row, col)
    """

    col = min(
        range(COLS),
        key=lambda c: abs(x - X_CENTERS[c])
    )

    row = min(
        range(ROWS),
        key=lambda r: abs(y - Y_CENTERS[r])
    )

    return row, col


def cell_to_pose(row, col):
    """
    Return the world-coordinate center of a maze cell.

    Returns:
        (x, y)
    """

    if not (0 <= row < ROWS):
        raise ValueError(f'Invalid row: {row}')

    if not (0 <= col < COLS):
        raise ValueError(f'Invalid column: {col}')

    x = X_CENTERS[col]
    y = Y_CENTERS[row]

    return x, y


def distance_to_cell_center(x, y, row, col):
    """
    Euclidean distance from robot pose to a cell center.
    """

    target_x, target_y = cell_to_pose(row, col)

    return math.sqrt(
        (target_x - x) ** 2 +
        (target_y - y) ** 2
    )


def yaw_to_direction(yaw):
    """
    Convert robot yaw into nearest cardinal direction.
    """

    directions = {
        'E': 0.0,
        'N': math.pi / 2.0,
        'W': math.pi,
        'S': -math.pi / 2.0
    }

    best_direction = None
    best_error = float('inf')

    for direction, target_yaw in directions.items():

        error = target_yaw - yaw

        while error > math.pi:
            error -= 2.0 * math.pi

        while error < -math.pi:
            error += 2.0 * math.pi

        if abs(error) < best_error:
            best_error = abs(error)
            best_direction = direction

    return best_direction