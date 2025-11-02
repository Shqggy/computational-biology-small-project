def euclidianDist(x, y):
    if len(x) != len(y):
        return -1
    
    ss = 0 # sum of squared distance
    for i in range(len(x)):
        ss += (y[i] - x[i])**2

    return ss**.5


