import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from first_tab import AI_landmarks
from second_tab import second_tab
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2




def matplotlib_to_html(fig):
    """Convert a Matplotlib figure to an HTML string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return f"<img src='data:image/png;base64,{img_base64}'/>"


import gradio as gr

def update_analysis_tab(coords, image, group, measurement):
    if coords is None:
        return gr.Warning("Missing coordinates. Please upload a valid cephalogram image and detect landmarks.")
    if image is None:
        return gr.Warning("No image provided. Please upload a valid cephalogram image.")
    
    try:
        image_with_vectors, measurements_html, line_plot = second_tab(coords, image, group, measurement)
        
        if line_plot:
            line_plot_html = matplotlib_to_html(line_plot)
        else:
            line_plot_html = "<p>No plot available</p>"

        return image_with_vectors, measurements_html, line_plot_html
    except Exception as e:
        return gr.Warning(f"An error occurred: {str(e)}")


def detect_landmarks(ceph_image_input):
    annotated_image = AI_landmarks()
    keypoints = annotated_image.process_image(ceph_image_input)

    img_copy = ceph_image_input.copy()
    for i, (x, y) in enumerate(keypoints):
        img_copy = cv2.circle(img_copy, (x, y), 10, (0, 255, 0), -1)
        img_copy = cv2.putText(img_copy, f"{i+1}", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
    
    img_copy_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    return keypoints, img_copy_rgb

with gr.Blocks() as demo:
    
    with gr.Tab("Upload and Detect"):
        with gr.Row():
            ceph_image_input = gr.Image(type="numpy", label="Upload Cephalogram Image")
            with gr.Column(): 
                coords_output = gr.JSON(label="Landmark Coordinates")
                detect_button = gr.Button("Detect Landmarks")
                
                detect_button.click(
                    fn=detect_landmarks, 
                    inputs=ceph_image_input, 
                    outputs=[coords_output, ceph_image_input]

            )
    
    with gr.Tab("Analysis") as analysis:
        with gr.Row():
            # Left side: Cephalogram with Landmarks
            ceph_image_plot = gr.Image(type="numpy", label="Cephalogram with Landmarks")

            # Right side: Measurement controls and output
            with gr.Column():
                group_selector = gr.Dropdown(label="Select Demographic Group", choices=["Malay", "Chinese", "Indian"], value="Malay")
                measurement_selector = gr.Radio(
                    choices=["SNA", "SNB", "ANB", "MMPA", "LFH", "U1A", "L1A"], 
                    label="Select Measurement", 
                    value="SNA"  # Default to the first measurement
                )
                line_analysis_plot = gr.HTML(label="Line Analysis Plot")
                measurements_output = gr.HTML(label="Measurements")
                

        # Automatically initialize outputs when the tab is loaded
        analysis.select(
            fn=update_analysis_tab, 
            inputs=[coords_output, ceph_image_input, group_selector, measurement_selector], 
            outputs=[ceph_image_plot, measurements_output, line_analysis_plot]
        )

        # Update outputs when landmark coordinates or demographic group changes
        coords_output.change(
            fn=update_analysis_tab, 
            inputs=[coords_output, ceph_image_input, group_selector, measurement_selector], 
            outputs=[ceph_image_plot, measurements_output, line_analysis_plot]
        )
        group_selector.change(
            fn=update_analysis_tab,
            inputs=[coords_output, ceph_image_input, group_selector, measurement_selector],
            outputs=[ceph_image_plot, measurements_output, line_analysis_plot]
        )
        measurement_selector.change(
            fn=update_analysis_tab,
            inputs=[coords_output, ceph_image_input, group_selector, measurement_selector],
            outputs=[ceph_image_plot, measurements_output, line_analysis_plot]
        )

demo.launch()
