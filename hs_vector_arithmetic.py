import torch
from denoising_diffusion_pytorch import Unet1D,GaussianDiffusion1D
from count_triangles import triangleInGraph
import numpy as np
from tqdm import tqdm
import time

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Parse arguments for diffusion model.")
    parser.add_argument("--model_path","-m", type=str, default="results/model-4.pt", help="Path to the model checkpoint.")
    parser.add_argument("--batch_size","-b", type=int, default=1, help="Batch size for sampling.")
    return parser.parse_args()


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
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
    #print h1.shape, h2.shape
    h_sum = h1.copy()
    for i in range(1000):
        h_sum[i] += torch.abs(h2[i]- h1[i]) * 2
    
    start_time = time.time()
    for i in range(batch_size):
        print(f"{i}/{batch_size}")
        count1 = triangleInGraph((sample1[i] > 0.5).reshape(20, 20).cpu().numpy(), V=20)
        count2 = triangleInGraph((sample2[i] > 0.5).reshape(20, 20).cpu().numpy(), V=20)
        print("Num triangles of sample:", count1, count2)
        num_edges1 = (sample1[i] > 0.5).sum().item()// 2
        num_edges2 = (sample2[i] > 0.5).sum().item()// 2
        print("Num edges of sample:", num_edges1, num_edges2)    

        # Decode
        with torch.no_grad():
            decoded = diffusion.sample_given_h(
                h_spaces=[x[i,:,:] for x in h_sum],
                batch_size=1,
            )

        e = (decoded[0] > 0.5).reshape(20, 20).cpu().numpy()
        count_h = triangleInGraph(e, V=20)
        print("Num triangles of created:", count_h)
        num_edges_h = e.sum().item() // 2
        print("Num edges of created:", num_edges_h)
        if (num_edges2 > num_edges1 and num_edges_h > num_edges2) or (num_edges1 > num_edges2 and num_edges_h > num_edges1):
            print("Found a better sample!")
            print("Creating an even better sample...")
            with torch.no_grad():
                decoded = diffusion.sample_given_h(
                    h_spaces=[x[i,:,:]+(x[i,:,:]-y[i,:,:])*2 for x,y in zip(h_sum,h2)],
                    batch_size=1,
                )
            e = (decoded[0] > 0.5).reshape(20, 20).cpu().numpy()
            count_h = triangleInGraph(e, V=20)
            num_edges_h = e.sum().item() // 2
            print("Num triangles of created:", count_h)
            print("Num edges of created:", num_edges_h)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("which is ", (end_time - start_time) / batch_size, "seconds per sample")
    print("Done!")
            