import torch
from denoising_diffusion_pytorch import GaussianDiffusion1D, Unet1D

model = Unet1D(
    dim = 20,
    dim_mults = (1, 2, 4, 8),
    channels = 1
)

model.load_state_dict(torch.load("results/model-7.pt"))
model.eval()
model = model.to("cuda")


diffusion = GaussianDiffusion1D(
    model = model,
    seq_length = 400,
    timesteps = 1000,
    objective = 'pred_v'
)


sampled_seq = diffusion.sample(batch_size = 4)
print(sampled_seq)