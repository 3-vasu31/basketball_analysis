def get_centre_of_bbox(bbox):
    '''Get the centre of a bounding box.'''
    x1,y1,x2,y2=bbox
    
    return int((x1+x2)/2),int((y1+y2)/2)


def get_bbox_width(bbox):
    '''Get the width of a bounding box.'''
    x1,y1,x2,y2=bbox
    
    return abs(x2-x1)