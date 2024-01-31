import cv2
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the multimodal-maestro directory to the system path
path_root = Path(__file__).parents[1]
maestro_path = str(path_root)
sys.path.append(maestro_path)

print(maestro_path)

# Importing maestro components
from maestro import SegmentAnythingMarkGenerator, MarkVisualizer, refine_marks, extract_relevant_masks, prompt_image_local

# Load the image
image_path = "/path/to/img.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Generate and refine marks
generator = SegmentAnythingMarkGenerator(device='cpu')
marks = generator.generate(image=image)
refined_marks = refine_marks(marks=marks)

# Visualize marks
mark_visualizer = MarkVisualizer()
marked_image = mark_visualizer.visualize(image=image, marks=refined_marks)

# Custom payload function for local server
def custom_payload_func(image_base64, prompt, system_prompt):
    return {
        "prompt": f"{system_prompt}. USER:[img-12]{prompt}\nASSISTANT:",
        "image_data": [{"data": image_base64, "id": 12}],
        "n_predict": 256,
        "top_p": 0.5,
        "temp": 0.2
    }

# Convert image to base64 and send request to local server
response = prompt_image_local(marked_image, "Find the crowbar", "http://localhost:8080/completion", custom_payload_func)
print(response)

# Extract relevant masks based on the response
masks = extract_relevant_masks(text=response, detections=refined_marks)

# Display using matplotlib
plt.imshow(marked_image)
plt.axis('off')
plt.show()
