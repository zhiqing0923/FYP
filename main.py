import gradio as gr
import cv2
import numpy as np
from first_tab import first_tab
from second_tab import second_tab

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
            ceph_image_plot = gr.Image(type="numpy", label="Cephalogram with Landmarks")
            with gr.Column():
                group_selector = gr.Dropdown(label="Select Demographic Group", choices=["Malay", "Chinese", "Indian"], value="Malay")
                measurements_output = gr.HTML(label="Measurements")

        coords_output.change(
            fn=second_tab, 
            inputs=[coords_output, ceph_image_input, group_selector], 
            outputs=[ceph_image_plot, measurements_output]
        )
        
        group_selector.change(
            fn=second_tab, 
            inputs=[coords_output, ceph_image_input, group_selector], 
            outputs=[ceph_image_plot, measurements_output]
        )

demo.launch(share=True)
