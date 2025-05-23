import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from make_graphs_no_triangle import load_graphs
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet1D(
    dim = 20,
    dim_mults = (1, 2, 4, 8),
    channels = 1
).to(device)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 400,
    timesteps = 1000,
    objective = 'pred_v'
)

if __name__ == "__main__":
    
    graphs = load_graphs("triangle_free_graphs/graphs_20.txt")
    training_seq = torch.Tensor(np.array(graphs))
    print(training_seq.shape, training_seq[0].dtype,training_seq[0].device)

    dataset = Dataset1D(training_seq)

    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 32,
        train_lr = 8e-4,
        train_num_steps = 7000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        split_batches= False,                # split batches into smaller chunks for memory efficiency
        num_workers= 0,                       # number of data loading workers
    )
    
    trainer.train()

    # save the model
    diffusion.save("./models/diffusion_model.pth")
    

    