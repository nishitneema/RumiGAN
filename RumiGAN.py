""" 
Simple GAN using fully connected layers

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-01: Initial coding
* 2022-12-20: Small revision of code, checked that it works with latest PyTorch version
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # Use 'TkAgg' backend or another suitable one for your system
import matplotlib.pyplot as plt


def save_checkpoint(state, filename = 'RumiGAN_disc.pth.tar'):
    print('saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print('loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 500
load_model = True

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)



## Creating Dataloader for odd and even class
class MNISTOddDataset(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.indices = [i for i, (_, label) in enumerate(self.mnist_dataset) if label % 2 == 1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.mnist_dataset[original_idx]


class MNISTEvenDataset(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.indices = [i for i, (_, label) in enumerate(self.mnist_dataset) if label % 2 == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.mnist_dataset[original_idx]
    


# Define the data transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download and load the MNIST dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_testset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Create custom datasets for odd and even labeled data
odd_dataset = MNISTOddDataset(mnist_trainset)
even_dataset = MNISTEvenDataset(mnist_trainset)

#Create data loaders
odd_dataloader = DataLoader(odd_dataset, batch_size=batch_size, shuffle=True)
even_dataloader = DataLoader(even_dataset, batch_size=batch_size, shuffle=True)


# dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
# loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"GAN/RumiGAN/logs/fake")
writer_real_pos = SummaryWriter(f"GAN/RumiGAN/logs/real_pos")
writer_real_neg = SummaryWriter(f"GAN/RumiGAN/logs/real_neg")
step = 0

if load_model:
    load_checkpoint(torch.load('GAN/RumiGAN/checkpoints/RumiGAN_disc.pt'), disc, opt_disc)
    load_checkpoint(torch.load('GAN/RumiGAN/checkpoints/RumiGAN_gen.pt'), gen, opt_gen)
else:
    for epoch in range(num_epochs):
        if epoch%5==0:
            checkpoint_disc = {'state_dict' : disc.state_dict(), 'optimizer' : opt_disc.state_dict()}
            save_checkpoint(checkpoint_disc, filename='GAN/RumiGAN/checkpoints/RumiGAN_disc.pt')
            checkpoint_gen = {'state_dict' : gen.state_dict(), 'optimizer' : opt_gen.state_dict()}
            save_checkpoint(checkpoint_gen, filename='GAN/RumiGAN/checkpoints/RumiGAN_gen.pt')
        for batch_idx, (batch1, batch2) in enumerate(zip(odd_dataloader, even_dataloader)):
            real_pos, label1 = batch1
            real_neg, label2 = batch2
            real_pos = real_pos.view(-1, 784).to(device)
            real_neg = real_neg.view(-1, 784).to(device)
            batch_size = real_pos.shape[0]

            ### Train Discriminator: max log(D(x)^+) + log(1 - D(G(z))) + log(1-D(x)^-)
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real_pos = disc(real_pos).view(-1)
            lossD_real_pos = criterion(disc_real_pos, torch.ones_like(disc_real_pos))
            disc_real_neg = disc(real_neg).view(-1)
            lossD_real_neg = criterion(disc_real_neg, torch.zeros_like(disc_real_neg))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real_pos + lossD_real_neg + lossD_fake) / 3
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # where the second option of maximizing doesn't suffer from
            # saturating gradients
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(odd_dataloader)} \
                        Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data_pos = real_pos.reshape(-1, 1, 28, 28)
                    data_neg = real_neg.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real_pos = torchvision.utils.make_grid(data_pos, normalize=True)
                    img_grid_real_neg = torchvision.utils.make_grid(data_neg, normalize=True)

                    writer_fake.add_image(
                        "Mnist Fake Images", img_grid_fake, global_step=step
                    )
                    writer_real_pos.add_image(
                        "Mnist Real Positive Images", img_grid_real_pos, global_step=step
                    )
                    writer_real_neg.add_image(
                        "Mnist Real Negative Images", img_grid_real_neg, global_step=step
                    )
                    step += 1
        
noi = torch.randn(batch_size, z_dim).to(device)
sam = gen(noi).reshape(batch_size, 1,28,28)
num_col = 4
grid_tensor = make_grid(sam, nrow=num_col)
save_image(sam,'GAN/RumiGAN/sample/img1.png')
sam_cpu = sam.detach().cpu()

# # Convert the tensor to a NumPy array
# sam_numpy = sam_cpu.permute(1, 2, 0).numpy()

# # Now you can use plt.imshow with the NumPy array
# sam_gray = sam_numpy.squeeze()  # Remove the single-channel dimension
# plt.imshow(sam_gray, cmap='gray')  # Display the grayscale image
# plt.show(block=True)

# img = Image.fromarray(sam_gray)  # Convert NumPy array to PIL Image if needed
# img = img.convert('L')
# save_directory = "GAN/RumiGAN_sample"
# os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist

# # Save the image as a PNG file in the specified directory
# img.save(os.path.join(save_directory, "image_500.png"))