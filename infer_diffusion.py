import torch
from denoising_diffusion_pytorch import GaussianDiffusion1D, Unet1D

model = Unet1D(
    dim = 20,
    dim_mults = (1, 2, 4, 8),
    channels = 1
)

print("Loading model...")



diffusion = GaussianDiffusion1D(
    model = model,
    seq_length = 400,
    timesteps = 1000,
    objective = 'pred_v'
)
diffusion.load_state_dict(torch.load("results/model-7.pt")["model"])


sampled_seq = diffusion.sample(batch_size = 1)
print(sampled_seq.shape, sampled_seq[0].dtype, sampled_seq[0].device)
for i in range(1,21):
    for j in range(1,21):
        if i*j < 400:
            print(1 if sampled_seq[0][0][i*j]>0.5 else 0 ,end=", ")
    print()
        