import torch
from denoising_diffusion_pytorch import Unet1D,GaussianDiffusion1D
from count_triangles import triangleInGraph
import numpy as np
from tqdm import tqdm

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Parse arguments for diffusion model.")
    parser.add_argument("--model_path","-m", type=str, default="results/model-4.pt", help="Path to the model checkpoint.")
    parser.add_argument("--batch_size","-b", type=int, default=1, help="Batch size for sampling.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    batch_size = args.batch_size
    
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
    diffusion.load_state_dict(torch.load(model_path)["model"])


    # Get bottleneck representations
    sample1,h1 = diffusion.sample_with_h(batch_size=batch_size)
    sample2,h2 = diffusion.sample_with_h(batch_size=batch_size)
    # Calculate the sum of bottleneck representations
    print("Shape of h1 and h2:", len(h1),len(h1[0]),h1[0][0].shape, len(h2),len(h2[0]))

    print("Shape of h1 and h2:", len(h1),len(h1[0]))
    for i in range(batch_size):
        count1 = triangleInGraph((sample1[i] > 0.5).reshape(20, 20).cpu().numpy(), V=20)
        count2 = triangleInGraph((sample2[i] > 0.5).reshape(20, 20).cpu().numpy(), V=20)
        print("Number of triangles in first and second sample graph:", count1, count2)
        num_edges1 = (sample1[i] > 0.5).sum().item()// 2
        num_edges2 = (sample2[i] > 0.5).sum().item()// 2
        print("Number of edges in first and second sample graph:", num_edges1, num_edges2)    
        
        
        # Decode
        with torch.no_grad():
            decoded = diffusion.sample_given_h(
                h_spaces=h_sum,
                batch_size=1,
            )

        e = (decoded[0] > 0.5).reshape(20, 20).cpu().numpy()
        count = triangleInGraph(e, V=20)
        print("Number of triangles in summed bottleneck h-space graph:", count)
        print("Number of edges in summed bottleneck h-space graph:", e.sum().item()// 2)