import numpy as np
from shapely.geometry import Point, Polygon

###### ROOM GEOMETRY DEFINITION ####################################################################

def get_room_polygon():
    """
    Returns the outer boundary polygon of the room.
    This includes a corridor cut-in on the left and a step on the top-right.

    Returns
    -------
    Polygon
        Shapely polygon defining the room boundary.
    """
    return Polygon([
        (1, 0), (15, 0), (15, 1), (14.5, 2), (14.5, 3),
        (15, 4), (15, 5), (0, 5), (0, 3), (1, 3)
    ])

def get_pillars():
    """
    Returns a list of cylindrical obstacles inside the room.

    Returns
    -------
    list of tuples
        Each tuple is (x, y, radius) for one circular pillar.
    """
    return [
        (3, 2.5, 0.3),
        (8, 2.5, 0.3),
        (12, 1.0, 0.3)
    ]

###### ROOM MASK GENERATION ########################################################################

def generate_room_mask(X, Y):
    """
    Generates a boolean mask indicating which points are inside the room and not inside pillars.

    Parameters
    ----------
    X, Y : 2D ndarray
        Meshgrid arrays of x and y coordinates.

    Returns
    -------
    mask : 2D boolean ndarray
        True where the point is inside the room and outside all obstacles.
    """
    room = get_room_polygon()
    pillars = get_pillars()

    mask = np.zeros_like(X, dtype=bool)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pt = Point(X[i, j], Y[i, j])
            inside_room = room.contains(pt)
            outside_pillars = all(pt.distance(Point(xc, yc)) > r for xc, yc, r in pillars)
            mask[i, j] = inside_room and outside_pillars

    return mask
