import cv2
from measurement import calculate_angles_and_distances, define_vectors, thresholds

def plot_landmarks(image, coords):
    if image is None or coords is None:
        return None
    
    img_copy = image.copy()
    for i, (x, y) in enumerate(coords):
        img_copy = cv2.circle(img_copy, (x, y), 10, (0, 255, 0), -1)
        img_copy = cv2.putText(img_copy, f"{i+1}", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
    
    img_copy_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    return img_copy_rgb

def plot_vectors(image, vectors):
    img_copy = image.copy()
    for vector in vectors.values():
        img_copy = cv2.line(img_copy, tuple(vector.start), tuple(vector.end), (0, 0, 255), 2)
    img_copy_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    return img_copy_rgb

def second_tab(coords, image, group):
    if coords is None or image is None or group is None:
        return None, None
    
    points, vectors = define_vectors(coords)
    eastman = calculate_angles_and_distances(coords)
    image_with_landmarks = plot_landmarks(image, coords)
    image_with_vectors = plot_vectors(image_with_landmarks, vectors)

    selected_thresholds = thresholds[group]
    measurements_html = "<ul>"
    measurement_names = ["SNA", "SNB", "ANB", "MMPA", "LFH", "U1A", "L1A"]
    
    for name, measurement in zip(measurement_names, eastman):
        lower_bound, upper_bound = selected_thresholds[name]
        color = "red" if not (lower_bound <= measurement <= upper_bound) else "green"
        measurements_html += f"<li style='color: {color};'>{name}: {measurement:.2f}</li>"
    
    measurements_html += "</ul>"
    return image_with_vectors, measurements_html
