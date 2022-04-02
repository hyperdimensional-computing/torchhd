def min_max_scaler(tensor, min_max=(0.0, 1.0)):
    y_min, y_max = min_max
    x_min = tensor.min().float()
    x_max = tensor.max().float()

    dist = x_max - x_min
    dist[dist == 0.0] = 1.0
    scale = 1.0 / dist

    scaled = tensor.float()
    scaled = scaled - x_min
    scaled *= scale
    scaled *= y_max - y_min
    scaled += y_min

    return scaled
