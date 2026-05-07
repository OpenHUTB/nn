def separate(lines):
    left = []
    right = []
    for l in lines:
        x1,y1,x2,y2 = l.reshape(4)
        k = (y2-y1)/(x2-x1)
        left.append(l) if k<0 else right.append(l)
    return left, right
