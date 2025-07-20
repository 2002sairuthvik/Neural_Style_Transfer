import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.models import vgg19, VGG19_Weights

# ----------------------
# Config
# ----------------------
CONTENT_PATH = Path(r"C:\Users\manik\Downloads\Neural_Style_Transfer\annahathaway.png")
STYLE_DIR    = Path(r"C:\Users\manik\Downloads\Neural_Style_Transfer\style")
OUTPUT_DIR   = Path(r"C:\Users\manik\Downloads\Neural_Style_Transfer\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tweaks for stronger content retention + visible style on CPU
IMSIZE          = 256       # resize square to 256×256
STEPS           = 2000      # more optimization steps
PRINT_EVERY     = 100
SAVE_EVERY      = 500

CONTENT_WEIGHT  = 1.0       # α
STYLE_WEIGHT    = 1e5       # β (tone down from 1e6 to 1e5)
LR              = 0.02      # learning rate

USE_IMAGENET_NORM = True    # apply VGG's ImageNet normalization
DEVICE            = torch.device("cpu")
torch.set_num_threads(4)    # optional: limit CPU threads

# ----------------------
# Image I/O transforms
# ----------------------
def make_transform(size=None, imagenet_norm=False):
    tfms = []
    if size is not None:
        tfms.append(T.Resize((size, size)))
    tfms.append(T.ToTensor())
    if imagenet_norm:
        tfms.append(T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ))
    return T.Compose(tfms)

def load_image(path, transform, device=DEVICE):
    img = Image.open(path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    return img_t

def denorm(tensor):
    """Undo ImageNet normalization."""
    if not USE_IMAGENET_NORM:
        return tensor
    mean = torch.tensor([0.485, 0.456, 0.406],
                        device=tensor.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225],
                        device=tensor.device).view(1,3,1,1)
    return tensor * std + mean

# ----------------------
# VGG feature extractor
# ----------------------
class VGGFeatures(nn.Module):
    """
    Extract activations at conv1_1, conv2_1, conv3_1, conv4_1, conv5_1.
    """
    def __init__(self, imagenet_weights=True):
        super().__init__()
        weights = VGG19_Weights.IMAGENET1K_V1 if imagenet_weights else None
        vgg = vgg19(weights=weights).features
        self.chosen = {"0","5","10","19","28"}  # layer indices
        self.slice = vgg[:29].to(DEVICE).eval()
        for p in self.slice.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.slice):
            x = layer(x)
            if str(i) in self.chosen:
                feats.append(x)
        return feats

# ----------------------
# Gram matrix
# ----------------------
def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    f = feat.view(b, c, h*w)
    G = f @ f.transpose(1, 2)  # (b, c, c)
    return G / (c * h * w)

# ----------------------
# Single-style optimization
# ----------------------
def stylise_one(content_img, style_img, model, out_path,
                steps=STEPS, lr=LR,
                content_w=CONTENT_WEIGHT, style_w=STYLE_WEIGHT,
                print_every=PRINT_EVERY, save_every=SAVE_EVERY):
    # Initialize generated as copy of content (better content retention)
    generated = content_img.clone().requires_grad_(True)

    # Precompute targets (no grad needed)
    with torch.no_grad():
        content_targets = model(content_img)
        style_feats     = model(style_img)
        style_targets   = [gram_matrix(f) for f in style_feats]

    optimizer = optim.Adam([generated], lr=lr)

    for step in range(1, steps+1):
        optimizer.zero_grad()
        gen_feats = model(generated)

        # Content loss
        content_loss = 0.0
        for gf, cf in zip(gen_feats, content_targets):
            content_loss += torch.mean((gf - cf)**2)

        # Style loss
        style_loss = 0.0
        for gf, st_g in zip(gen_feats, style_targets):
            G = gram_matrix(gf)
            style_loss += torch.mean((G - st_g)**2)

        total_loss = content_w * content_loss + style_w * style_loss
        total_loss.backward()
        optimizer.step()

        # Clamp pixel range
        with torch.no_grad():
            generated.clamp_(0.0, 1.0)

        if step % print_every == 0 or step == 1:
            print(f"[{step:4d}/{steps}] total={total_loss.item():.2f} "
                  f"(c={content_loss.item():.2f}, s={style_loss.item():.2f})")

        if step % save_every == 0 or step == steps:
            out_img = denorm(generated.detach().cpu()).clamp(0,1)
            save_image(out_img,
                       out_path.with_name(f"{out_path.stem}_step{step}.png"))

    # final save
    out_img = denorm(generated.detach().cpu()).clamp(0,1)
    save_image(out_img, out_path.with_suffix(".png"))
    return generated.detach()

# ----------------------
# Main loop
# ----------------------
def main():
    tfm = make_transform(size=IMSIZE, imagenet_norm=USE_IMAGENET_NORM)
    content = load_image(CONTENT_PATH, tfm)

    style_files = sorted(
        p for p in STYLE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not style_files:
        raise FileNotFoundError(f"No style images in {STYLE_DIR!r}")

    feat_model = VGGFeatures(imagenet_weights=True)

    for style_path in style_files:
        print(f"\n--- Stylizing with {style_path.name} ---")
        style = load_image(style_path, tfm)
        out_base = OUTPUT_DIR / f"{CONTENT_PATH.stem}_{style_path.stem}"
        stylise_one(content, style, feat_model, out_base)

    print("\nAll done — outputs saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    torch.manual_seed(0)
    main()
