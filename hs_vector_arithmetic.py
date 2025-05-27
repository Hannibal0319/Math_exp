import torch
from denoising_diffusion_pytorch import Unet1D,GaussianDiffusion1D
from count_triangles import triangleInGraph
import numpy as np

if __name__ == "__main__":
    model = Unet1D(
        dim=20,
        dim_mults=(1, 2, 4, 8),
        channels=1
    )
    diffusion = GaussianDiffusion1D(
        model=model,
        seq_length=400,
        timesteps=1000,
        objective='pred_v'
    )
    diffusion.load_state_dict(torch.load("results/model-4.pt")["model"])

    device = next(model.parameters()).device
    seq_len = 400
    

    # Get bottleneck representations
    sample1,h1 = diffusion.sample_with_h(batch_size=1)
    
    count = triangleInGraph((sample1[0] > 0.5).reshape(20, 20).cpu().numpy(), V=20)
    print("Number of triangles in first sample graph:", count)
    num_edges = (sample1[0] > 0.5).sum().item()// 2
    print("Number of edges in first sample graph:", num_edges)
    
    sample2,h2 = diffusion.sample_with_h(batch_size=1)
    
    count = triangleInGraph((sample2[0] > 0.5).reshape(20, 20).cpu().numpy(), V=20)
    print("Number of triangles in second sample graph:", count)
    num_edges = (sample2[0] > 0.5).sum().item()// 2
    print("Number of edges in second sample graph:", num_edges)
    
    h_sum =  [(h1[i] + h2[i]) for i in range(len(h1)) ]

    # Decode
    with torch.no_grad():
        decoded = diffusion.sample_from_h(
            h=h_sum,
            batch_size=1,
        )

    print("Decoded graph shape:", decoded.shape)
    print("Decoded graph (first sample):", decoded[0])

    e = (decoded[0] > 0.5).reshape(20, 20).cpu().numpy()
    count = triangleInGraph(e, V=20)
    print("Number of triangles in summed bottleneck h-space graph:", count)