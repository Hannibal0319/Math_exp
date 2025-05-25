import torch
from denoising_diffusion_pytorch import Unet1D,GaussianDiffusion1D
from count_triangles import triangleInGraph

def get_bottleneck(model, x, t):
    # Forward up to bottleneck (after mid_block2)
    x = model.init_conv(x)
    t_emb = model.time_mlp(t)
    h = []
    for block1, block2, attn, downsample in model.downs:
        x = block1(x, t_emb)
        h.append(x)
        x = block2(x, t_emb)
        x = attn(x)
        h.append(x)
        x = downsample(x)
    x = model.mid_block1(x, t_emb)
    x = model.mid_attn(x)
    x = model.mid_block2(x, t_emb)
    return x

def decode_from_bottleneck(model, bottleneck, t):
    # Only upsampling path, no skip connections
    x = bottleneck
    t_emb = model.time_mlp(t)
    for block1, block2, attn, upsample in model.ups:
        skip = torch.zeros_like(x)
        x = torch.cat((x, skip), dim=1)
        x = block1(x, t_emb)
        x = torch.cat((x, skip), dim=1)
        x = block2(x, t_emb)
        x = attn(x)
        x = upsample(x)
    r = torch.zeros_like(x)
    x = torch.cat((x, r), dim=1)
    x = model.final_res_block(x, t_emb)
    out = model.final_conv(x)
    return out

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

    # Sample two random noises
    noise1 = torch.randn(1, 1, seq_len).to(device)
    noise2 = torch.randn(1, 1, seq_len).to(device)
    t = torch.zeros((1,), dtype=torch.long, device=device)

    # Get bottleneck representations
    h1 = get_bottleneck(model, noise1, t)
    h2 = get_bottleneck(model, noise2, t)
    h_sum = h1 + h2

    # Decode
    with torch.no_grad():
        decoded = decode_from_bottleneck(model, h_sum, t)

    print("Decoded graph shape:", decoded.shape)
    print("Decoded graph (first sample):", decoded[0])

    e = (decoded[0] > 0.5).reshape(20, 20).cpu().numpy()
    count = triangleInGraph(e, V=20)
    print("Number of triangles in summed bottleneck h-space graph:", count)