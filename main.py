import gradio as gr
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from first_tab import first_tab
from second_tab import second_tab


def matplotlib_to_html(fig):
    """Convert a Matplotlib figure to an HTML string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return f"<img src='data:image/png;base64,{img_base64}'/>"


def update_analysis_tab(coords, image, group, measurement):
    """Helper function to handle updates for the analysis tab."""
    image_with_vectors, measurements_html, line_plot = second_tab(coords, image, group, measurement)
    if line_plot:
        line_plot_html = matplotlib_to_html(line_plot)
    else:
        line_plot_html = "<p>No plot available</p>"
    return image_with_vectors, measurements_html, line_plot_html


with gr.Blocks() as demo:
    with gr.Tab("Upload and Detect"):
        ceph_image_input = gr.Image(type="numpy", label="Upload Cephalogram Image", width=500, height=500)
        coords_output = gr.JSON(label="Landmark Coordinates")
        detect_button = gr.Button("Detect Landmarks")
        
        detect_button.click(
            fn=first_tab, 
            inputs=ceph_image_input, 
            outputs=[coords_output, ceph_image_input]
        )
    
    with gr.Tab("Analysis"):
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
        demo.load(
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
