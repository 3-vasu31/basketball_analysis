import cv2
import sys
sys.path.append("../")  
import numpy as np

from utils import get_centre_of_bbox, get_bbox_width

def draw_triangle(frame, bbox, color):
    """
    Draws a triangle on the frame based on the bounding box coordinates.

    Args:
        frame (numpy.ndarray): The video frame to draw on.
        bbox (list): The bounding box coordinates [x1, y1, x2, y2].
        color (tuple): The color of the triangle in BGR format.

    Returns:
        numpy.ndarray: The frame with the drawn triangle.
    """
    y = int(bbox[1])
    x, _ = get_centre_of_bbox(bbox)
    
    traingle_points = np.array([
        [int(x), int(y)],
        [int(x- 10), int(y - 20)],
        [int(x + 10), int(y - 20)],
    ])
    

    cv2.drawContours(frame, [traingle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [traingle_points], 0, (0, 0, 0), 2)
    
    return frame

def draw_ellipse(frame, bbox, color, track_id=None):
    """
    Draws an ellipse on the frame based on the bounding box coordinates.

    Args:
        frame (numpy.ndarray): The video frame to draw on.
        bbox (list): The bounding box coordinates [x1, y1, x2, y2].
        color (tuple): The color of the ellipse in BGR format.
        track_id (int, optional): The ID of the track. If provided, it will be drawn on the frame.

    Returns:
        numpy.ndarray: The frame with the drawn ellipse.
    """
    x1, y1, x2, y2 = bbox

    x_center,_= get_centre_of_bbox(bbox)
    width = get_bbox_width(bbox)
    center = (int(x_center),int( y2))
    axes = (int(width/2), int(0.35*width))
    angle = 0.0
    startAngle = -45.0
    endAngle = 235.0
    thickness = 2
    cv2.ellipse(frame, 
                center,
                axes ,
                angle,
                startAngle,
                endAngle,
                color,
                thickness,
                lineType=cv2.LINE_4)
    
    rectangle_width =40
    rectangle_height = 20
    x1_rect = x_center -rectangle_width//2
    x2_rect = x_center + rectangle_width//2
    y1_rect= (y2 - rectangle_height//2) + 15
    y2_rect = (y2 + rectangle_height//2) + 15

    if track_id is not None:
        cv2.rectangle(
            frame,
            (int(x1_rect),int(y1_rect)),
            (int(x2_rect),int(y2_rect)),
            color,
            cv2.FILLED
        )

        x1_text = x1_rect + 12
        if track_id >99: #if the track id is more than 99, then we need to adjust the x1_text
            x1_text -= 10

        cv2.putText(
            frame,
            str(track_id),
            (int(x1_text), int(y2_rect-3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.6,
            color=(0,0,0),
            thickness=2
        )

    return frame