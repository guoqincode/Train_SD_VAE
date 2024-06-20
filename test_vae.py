import os
import torch
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

image_processor = VaeImageProcessor(vae_scale_factor=0.1825)
model_path = "./sd-vae-ft-mse"
model = AutoencoderKL.from_pretrained(model_path, local_files_only=True)
model.eval()

output_dir = "./results_vae"
os.makedirs(output_dir, exist_ok=True)

def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def combine_images(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    
    combined = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return combined

def process_image(image_path):
    input_image = preprocess(image_path)
    original_image = Image.open(image_path).convert("RGB").resize((512, 512))
    
    decoded_images = [original_image] 
    
    with torch.no_grad():
        encoded = model.encode(input_image).latent_dist.sample()

    with torch.no_grad():
        decoded = model.decode(encoded)[0]

    decoded_image = image_processor.postprocess(decoded)[0]
    decoded_images.append(decoded_image)

    combined_image = combine_images(decoded_images)

    output_image_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_combined.jpg")
    combined_image.save(output_image_path)
    print(f"Saved combined image to {output_image_path}")

# 示例使用
image_path = "./test.jpg"
process_image(image_path)
