""" Training of ProGAN using WGAN-GP loss"""

import torch
import torch.optim as optim
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config
import cv2
import numpy as np
import os
import time
from augmentation_pipeline import DiffAugment



torch.backends.cudnn.benchmarks = True



policy = 'color,translation' # For Augmentation Pipeline





def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset




def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_step,
    writer,
    scaler_gen,
    scaler_critic,

):
    

    # Adaptive Augmenation Parameters
    ada_p = 0.0
    ada_target = 0.6
    ada_k = 0.001

    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):

        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]


        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)
        fake = gen(noise, alpha, step)  # Moved outside amp to reuse

        fake_aug = None  # Default so it's visible later


        with torch.cuda.amp.autocast():
            real_aug = DiffAugment(real, policy=policy)
            fake_aug = DiffAugment(fake, policy=policy)

            if ada_p!=0:
                critic_real = critic(real_aug, alpha, step)
                critic_fake = critic(fake_aug.detach(), alpha, step)
                gp = gradient_penalty(critic, real_aug, fake_aug, alpha, step, device=config.DEVICE)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + config.LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2))
                )

            else:
                critic_real = critic(real, alpha, step)
                critic_fake = critic(fake.detach(), alpha, step)
                gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + config.LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2))
                )


            with torch.no_grad():
                signs = torch.sign(critic_real)
                ada_stat = signs.mean().item()

            ada_p += ada_k * (ada_stat - ada_target)
            ada_p = min(max(ada_p, 0.0), 1.0)  # Clamp to [0, 1]


        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()


#        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
#        with torch.cuda.amp.autocast():
#            gen_input = fake_aug if fake_aug is not None else fake
#            gen_fake = critic(gen_input, alpha, step)
#            loss_gen = -torch.mean(gen_fake)

        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = torch.mean(gen_fake)

            
        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
            (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
            loss_generator=loss_gen.item(),
            ada_p = ada_p
        )

    return tensorboard_step, alpha


def main():
    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    # but really who cares..
    gen = Generator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)
    critic = Discriminator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

    # initialize optimizers and scalers for FP16 training
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99)
    )
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    # for tensorboard plotting
    writer = SummaryWriter(f"logs/gan1")

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,
        )

    gen.train()
    critic.train()

    tensorboard_step = 0
    # start at step that corresponds to img size that we set in config
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE) - 2)
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5  # start with very low alpha
        loader, dataset = get_loader(4 * 2 ** step)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
        print(f"Current image size: {4 * 2 ** step}")

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            tensorboard_step, alpha = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_critic,
            )

            # Generate and save images every 10 epochs
            if (epoch + 1) % 10 == 0:
                generate_and_save_images(gen, step, alpha, epoch, num_epochs)

            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC)

        step += 1  # progress to the next img size

    # Final image generation at the end of training
    print("Training complete. Generating example images...")
    
    # Switch to evaluation mode
    gen.eval()
    
    # Set to the final step/resolution
    final_step = step - 1
    alpha = 1.0  # Fully trained model uses alpha=1
    
    # Generate final images
    generate_and_save_images(gen, final_step, alpha, "final", "final")
    
    print("Images generated and saved.")


def generate_and_save_images(gen, step, alpha, epoch, total_epochs):
    """
    Generate and save 3 images in a row
    """
    # Temporarily set generator to eval mode
    gen_status = gen.training
    gen.eval()
    
    with torch.no_grad():
        # Create fixed noise for consistent visualization
        fixed_noise = torch.randn(3, config.Z_DIM, 1, 1).to(config.DEVICE)
        
        # Generate images
        fake_images = gen(fixed_noise, alpha, step)
        
        # Denormalize the images from [-1, 1] to [0, 1]
        fake_images = (fake_images * 0.5) + 0.5
        
        # Create a single row image with 3 images side by side
        # First convert tensors to numpy arrays
        images_np = []
        for i in range(3):
            img = fake_images[i].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype('uint8')
            # Convert from RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images_np.append(img)
        
        # Calculate the dimensions of the combined image
        img_height, img_width = images_np[0].shape[:2]
        combined_img = np.zeros((img_height, img_width * 3, 3), dtype=np.uint8)
        
        # Combine the images horizontally
        for i in range(3):
            combined_img[:, i * img_width:(i + 1) * img_width] = images_np[i]
        
        # Create directory if it doesn't exist
        os.makedirs("generated_images", exist_ok=True)
        
        # Save the combined image with epoch information
        epoch_info = f"{epoch}_of_{total_epochs}" if isinstance(epoch, int) else epoch
        img_size = 4 * (2 ** step)
        filename = f"generated_images/progress_size{img_size}_epoch{epoch_info}.png"
        cv2.imwrite(filename, combined_img)
        
        # Also write individual images if needed
        for i in range(3):
            individual_filename = f"generated_images/img{i}_size{img_size}_epoch{epoch_info}.png"
            cv2.imwrite(individual_filename, images_np[i])
    
    # Restore generator to its previous training state
    gen.train(gen_status)
    
    print(f"Generated images saved for size {img_size}, epoch {epoch}")




if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"finished in: {end_time - start_time:.2f} seconds")