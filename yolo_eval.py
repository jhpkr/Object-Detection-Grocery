import torch
import random
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
weights_path = "C:/Users/Joon Park/Desktop/School/UCHICAGO WORK/ML/final/yolov5/runs/train/exp11/weights/best.pt"
image_dir = Path("C:\\Users\\Joon Park\\Desktop\\School\\UCHICAGO WORK\\ML\\final\\yolov5\\test\\images")

# Load the YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path=weights_path, source='local')

# List all image files in the dataset directory
image_files = list(image_dir.glob("*.jpg"))  # Adjust the pattern if your images have different extensions

# Randomly sample 10 images
sample_images = random.sample(image_files, 10)

# Directory to save results
results_dir = Path("runs/detect/samples")
results_dir.mkdir(parents=True, exist_ok=True)

# Run detection and display results
for img_path in sample_images:
    # Read the image
    img = cv2.imread(str(img_path))

    # Run detection
    results = model(img)

    # Render the results
    rendered_img = results.render()[0]

    # Save the rendered image
    result_path = results_dir / img_path.name
    cv2.imwrite(str(result_path), rendered_img)

    # Display the image
    plt.imshow(cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB))
    plt.title(img_path.name)
    plt.axis('off')
    plt.show()

#python val.py --weights "C:/Users/Joon Park/Desktop/School/UCHICAGO WORK/ML/final/yolov5/runs/train/exp11/weights/best.pt" --data "C:\\Users\\Joon Park\\Desktop\\School\\UCHICAGO WORK\\ML\\final\\yolov5\\data.yaml" --task test --img 256
