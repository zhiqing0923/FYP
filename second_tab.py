import cv2
import matplotlib.pyplot as plt
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

def plot_vectors(image, vectors, highlight_measurement=None):
    img_copy = image.copy()
    
    # Define colors for normal and highlighted vectors
    normal_color = (0, 0, 255)  # Red
    highlight_color = (0, 255, 0)  # Green

    measurement_to_vectors = {
        "SNA": ["NS", "NA"],  
        "SNB": ["NS", "NB"],
        "ANB": ["NS", "NA", "NB"],
        "MMPA": ["GoMe", "PNS_ANS"],  
        "LFH": ["PNS_ANS"],
        "U1A": ["ANS_U1", "PNS_ANS"],
        "L1A": ["MeL1", "GoMe"],
    }
    highlight_vectors = measurement_to_vectors.get(highlight_measurement, [])
    
    # Iterate through vectors and plot them
    for vector_name, vector in vectors.items():
        color = highlight_color if vector_name in highlight_vectors else normal_color
        img_copy = cv2.line(img_copy, tuple(vector.start), tuple(vector.end), color, 2)
    
    img_copy_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    return img_copy_rgb

def plot_line_analysis(value, mean, sd, lower_threshold, upper_threshold):
    """Plot a line analysis for a specific measurement."""
    fig, ax = plt.subplots(figsize=(6, 1.5))

    ax.axvspan(lower_threshold, upper_threshold, color='red', alpha=0.3, label='Threshold Range')
    ax.axvline(mean, color='black', linestyle='--', label='Mean')
    ax.axvline(sd, color= 'none', label=f'SD: {sd}')
    ax.plot(value, 0, 'g^', markersize=12, label=f'Value: {value:.2f}')

    # Add labels and legend
    ax.set_xlim(lower_threshold - 10, upper_threshold + 10)
    ax.set_yticks([])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize='large')
    plt.tight_layout()

    return fig


def second_tab(coords, image, group, clicked_measurement=None):
    if coords is None or image is None or group is None:
        return None, None
    
    # Calculate angles and distances using the coordinates
    points, vectors = define_vectors(coords)  # Get both points and vectors
    eastman = calculate_angles_and_distances(coords)  # Get eastman measurements
    
    # Plot the landmarks on the image
    image_with_landmarks = plot_landmarks(image, coords)

    # Plot the vectors on the image with potential highlight for clicked measurement
    image_with_vectors = plot_vectors(image_with_landmarks, vectors, highlight_measurement=clicked_measurement)

    # Fetch the thresholds for the selected group
    selected_thresholds = thresholds[group]

    # Prepare HTML output for measurements
    measurements_html = "<ul>"
    
    measurement_names = ["SNA", "SNB", "ANB", "MMPA", "LFH", "U1A", "L1A"]
    line_plot = None  # Initialize line plot as None
    for name, measurement in zip(measurement_names, eastman):
        lower_bound, upper_bound = selected_thresholds[name]
        color = "red" if not (lower_bound <= measurement <= upper_bound) else "green"
        measurements_html += f"<li style='color: {color}; font-size: 14px;'>{name}: {measurement:.2f}</li>"

        # Generate a line plot for the clicked measurement
        if clicked_measurement == name:
            mean = (lower_bound + upper_bound) / 2
            sd = (upper_bound - lower_bound) / 2  
            line_plot = plot_line_analysis(measurement, mean, sd, lower_bound, upper_bound)

    measurements_html += "</ul>"
    
    return image_with_vectors, measurements_html, line_plot


