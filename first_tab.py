# Sample coordinates (as if processed by a model)
sample_coords = [(835, 996), (1473, 1029), (1289, 1279), (604, 1228), (1375, 1654), (1386, 2019), 
                 (1333, 2200), (1263, 2272), (1305, 2252), (694, 1805), (1460, 1870), (1450, 1864), 
                 (1588, 1753), (1569, 2013), (1514, 1620), (1382, 2310), (944, 1506), 
                 (1436, 1569), (664, 1340)]

def process_image(image):
    if image is None:
        return None
    return sample_coords

def first_tab(image):
    coords = process_image(image)
    if coords is None:
        return None, None
    return coords, image
