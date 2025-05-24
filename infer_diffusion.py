import torch
from denoising_diffusion_pytorch import GaussianDiffusion1D, Unet1D
from count_triangles import count_triangles,triangleInGraph

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for diffusion model.")
    parser.add_argument('--batch_size','-b', type=int, default=16, help='Batch size for sampling.')
    return parser.parse_args()


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

if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    
    sampled_seq = diffusion.sample(batch_size = batch_size)
    print(sampled_seq.shape, sampled_seq[0].dtype, sampled_seq[0].device)


    summa_edges = 0
    max_edge = -1
    summa_triangles = 0
    for e in sampled_seq:
        # with treshold 0.5 make e into binary adjacency matrix
        e = (e > 0.5).reshape(20, 20).cpu().numpy()
        count = triangleInGraph(e,V=20)
        summa_triangles +=  count
        
        print("Number of triangles in sequence:", count)
        #number of edges in graph
        num_edges = 0
        for i in range(11):
            for j in range(i, 11):
                if e[i][j] == 1 or e[j][i] == 1:
                    num_edges += 1
        summa_edges += num_edges
        max_edge=max(num_edges, max_edge)
        print("Number of edges in sequence:", num_edges)

    print("avarage number of edges in sampled sequences:", summa_edges / len(sampled_seq))
    print("avarage number of triangles in sampled sequences:", summa_triangles / len(sampled_seq))
    print("Inference complete.")


        