import torch
import torchvision
import os
import config
import cv2
import numpy as np
from model import Generator
from math import log2
import time


# Paths
gen_path = r'D:\omd_sa\thesis_sepehr\generator.pth'
save_path = r'D:\omd_sa\thesis_sepehr\synthetic_images'

# Create directory
os.makedirs(save_path, exist_ok=True)

def load_progan_generator(model_path):
    """
    Load the trained ProGAN generator model
    """
    # Initialize generator with same parameters as training
    generator = Generator(
        config.Z_DIM, 
        config.IN_CHANNELS, 
        img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)
    
    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the entire dict is the state dict
            generator.load_state_dict(checkpoint)
    else:
        # Direct state dict
        generator.load_state_dict(checkpoint)
    
    generator.eval()  # Set to evaluation mode
    return generator

def generate_single_images(gen, target_img_size, num_images=100):
    """
    Generate individual images using the exact same method as training code
    """
    step = int(log2(target_img_size / 4))
    alpha = 1.0  # Full resolution
    
    # Temporarily set generator to eval mode 
    gen.eval()
    
    print(f"Generating {num_images} individual images at {target_img_size}x{target_img_size}...")
    
    with torch.no_grad():
        for i in range(num_images):
            # Generate single image with random noise
            noise = torch.randn(1, config.Z_DIM, 1, 1).to(config.DEVICE)
            
            # Generate image
            fake_image = gen(noise, alpha, step)
            
            # Denormalize the images from [-1, 1] to [0, 1] 
            fake_image = (fake_image * 0.5) + 0.5
            
            # Convert tensor to numpy array (exact same as your code)
            img = fake_image[0].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype('uint8')
            
            # Convert from RGB to BGR for OpenCV (exact same as your code)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Save individual image
            filename = os.path.join(save_path, f'synthetic_img_{i:04d}_size{target_img_size}.png')
            cv2.imwrite(filename, img)
            
            if (i + 1) % 20 == 0:
                print(f"Generated {i + 1}/{num_images} images...")
    
    
    print(f"Completed! All {num_images} images saved to {save_path}")


    



# Main execution
if __name__ == "__main__":
    start_time = time.time()

    print("Loading ProGAN generator...")
    generator = load_progan_generator(gen_path)
    
    # Define target image size 
    TARGET_SIZE = 256  # Change this to your desired resolution
    
    print("="*64)
    print("PROGAN SYNTHETIC IMAGE GENERATION")
    print("="*64)
    
    # Generate individual images (100 single images)
    generate_single_images(generator, TARGET_SIZE, num_images=100)

    end_time = time.time()
    print(f'finished in {end_time - start_time} seconds')
    


