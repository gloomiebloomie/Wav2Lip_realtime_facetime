import time
import threading
import gradio as gr

class OutputUpdater:
    def __init__(self, gr_interface):
        self.gr_interface = gr_interface
        self.output = None
        self.lock = threading.Lock()

    def update_output(self, new_output):
        with self.lock:
            self.output = new_output
            self.gr_interface.outputs[0].textbox.value = self.output

def read_text_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    return content

def video_response():
    passit_content = read_text_file("passit_output.txt")
    outfile_path = "results/result_voice.mp4"
    return passit_content, outfile_path

def launch_gr_interface():
    demoo = gr.Interface(fn=video_response, inputs=[], outputs=[gr.Textbox(), gr.Video()])
    demoo.launch()

    # Create an instance of the OutputUpdater class
    output_updater = OutputUpdater(demoo)

    while True:
        # Update the output
        new_output = video_response()
        output_updater.update_output(new_output[0])  # Pass only the text content

        # Wait for the next update
        time.sleep(5)

if __name__ == '__main__':
    launch_gr_interface()



