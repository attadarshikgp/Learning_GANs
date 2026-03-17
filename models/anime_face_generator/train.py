import torchvision.utils as vutils
import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from models import Generator
from models import Discriminator

output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Root directory for dataset
dataroot = "C:\\acads\\sem4\\projects\\data\\anime faces"

# Number of workers for dataloader
workers = 0

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Mckinsey666 images are usually 64x64, but we'll resize just in case
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) # Mean and Std for [-1, 1] normalization


# Mckinsey666 images are usually 64x64, but we'll resize just in case
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) # Mean and Std for [-1, 1] normalization

device= torch.device("cuda:0")
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

dataset = dset.ImageFolder(root=dataroot, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
 
#weights initialization

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#create instances for both generator and discriminator

gen = Generator().to(device)

disc = Discriminator().to(device)

gen.apply(weights_init)
disc.apply(weights_init)

#Loss function

criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

optimizerD = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))

real_label = 1.
fake_label = 0.

#tarining loop

for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        #disc training
        disc.zero_grad()
        #all real batch 
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = disc(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        #all fakes

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = gen(noise)
        label.fill_(fake_label)
        output = disc(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        #gen training
        gen.zero_grad()
        label.fill_(real_label)
        output = disc(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            with torch.no_grad():
                fake_display = gen(fixed_noise).detach().cpu()
                
            # Create a filename like: epoch_0_batch_50.png
            img_filename = os.path.join(output_dir, f"epoch_{epoch}_batch_{i}.png")
            
            # Save the grid
            vutils.save_image(fake_display, img_filename, normalize=True)