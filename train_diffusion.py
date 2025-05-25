import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
from make_graphs_no_triangle import load_graphs
import numpy as np
from pprint import pprint

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

def count_edges(graph,N):
    count = 0
    for i in range(N):
        for j in range(i, N):
            if graph[i][j] == 1 or graph[j][i] == 1:
                count += 1
    return count

if __name__ == "__main__":
    
    graphs_ = load_graphs("triangle_free_graphs/graphs_20_num3000.txt")
    graphs = []
    #count of graphs by number of edges
    edge_count = {}
    for graph in graphs_:
        num_edges = count_edges(np.array(graph).reshape(20,20), N=20)
        if num_edges != 100:
            graphs.append(graph)
        if num_edges not in edge_count:
            edge_count[num_edges] = 0
        edge_count[num_edges] += 1

    print("Number of graphs by number of edges:")
    pprint(edge_count)
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

    

    