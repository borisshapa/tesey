import torch

if __name__ == "__main__":
    with open("model_cpu.t7", "rb") as f:
        model = torch.load(f)
    model.eval()
