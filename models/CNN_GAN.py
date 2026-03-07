import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import os

os.makedirs("generated_images", exist_ok=True)

#use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data loader 
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))
    ])

MINST_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(MINST_dataset, batch_size=128, shuffle=True)

#discriminator using cnn
class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            #output size = (n+2p-k_size)/stride + 1 
            nn.Conv2d( 1, 32, 4, stride=2, padding=1), # -> N, 32, 14, 14 
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # -> N, 64, 7, 7
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # -> N, 128, 3, 3
            nn.LeakyReLU(0.1),
            nn.Flatten(), # -> N, 128*3*3
            nn.Linear(128*3*3, 1), # -> N, 1
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(100, 128*7*7),
            nn.ReLU(),

            nn.Unflatten(1, (128,7,7)),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # N,64,14,14
            nn.ReLU(),

            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),   # N,1,28,28
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# testing discriminator 
'''disc = discriminator().to(device)
x = torch.randn(8,1,28,28).to(device)
out = disc(x).to(device)
print(out.shape)'''

#testing generator
'''gen = Generator().to(device)
z = torch.randn(8,100).to(device)
out = gen(z).to(device)
print(out.shape)'''

#tarining loop

disc = discriminator().to(device)
gen = Generator().to(device)

criterion = nn.BCELoss()

opt_disc = torch.optim.Adam(disc.parameters(), lr=0.0002)
optim_gen = torch.optim.Adam(gen.parameters(), lr=0.0002)

z_dim = 100
epochs = 60
#for each epoch --> for each batch --> train discriminator and generator

fixed_noise = torch.randn(16, z_dim).to(device)

for epoch in range(epochs):
    for batch_idx, (real_imgs, _) in enumerate(train_loader):

        #train discriminator: max log(D(real)) + log(1-D(fak  cv             e))
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)
        opt_disc.zero_grad()
        output_real = disc(real_imgs)
        loss_real = criterion(output_real, torch.ones(batch_size, 1).to(device))

        #generate fake images   (noise)
        z= torch.randn(batch_size, z_dim).to(device)
        fake_images = gen(z)

        # use detach to not update generator only train the discriminator 
        ouput_fake = disc(fake_images.detach())
        loss_fake = criterion(ouput_fake, torch.zeros(batch_size, 1).to(device))

        #DEFine the loss 
        loss_disc = loss_real + loss_fake

        loss_disc.backward()
        opt_disc.step()
        
        #train the generator: min log(1-D(fake)) <-> max log(D(fake))
        optim_gen.zero_grad()

        z = torch.randn(batch_size, z_dim).to(device)
        fake_image = gen(z)
        
        output = disc(fake_image)

        loss_gen = criterion(output, torch.ones(batch_size, 1).to(device))
        loss_gen.backward()
        optim_gen.step()
        
        
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}]  D_loss: {loss_disc:.4f}  G_loss: {loss_gen:.4f}")\
    
    with torch.no_grad():
        fake = gen(fixed_noise).detach().cpu()

        fake = (fake + 1) / 2 
        
    if epoch % 5 == 0:
        torchvision.utils.save_image(
            fake,
            f"generated_images/epoch_{epoch}.png",
            nrow=4,
            normalize=True
        )

# after training loop
torch.save(gen.state_dict(), "generator.pth")
torch.save(disc.state_dict(), "discriminator.pth")

print("Models saved")