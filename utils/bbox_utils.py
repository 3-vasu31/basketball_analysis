def get_centre_of_bbox(bbox):
    '''Get the centre of a bounding box.'''
    x1,y1,x2,y2=bbox
    
    return int((x1+x2)/2),int((y1+y2)/2)


def get_bbox_width(bbox):
    '''Get the width of a bounding box.'''
    x1,y1,x2,y2=bbox
    
    return abs(x2-x1)

def measure_distance(p1, p2):
    '''Measure the Euclidean distance between two points.'''
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


