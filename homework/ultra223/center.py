def center(left, right):
    lx = (left[0][0][0]+left[0][0][2])/2
    rx = (right[0][0][0]+right[0][0][2])/2
    return int((lx+rx)/2)
