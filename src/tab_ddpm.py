import torch
import torch.nn as nn

def setup():
    beta = torch.linspace(start=(10)**(-4), end=0.02, steps=1000)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    return alpha_bar

def q_sample(x_0, t):
    alpha_bar = setup()
    squezzed_alpha_bar = alpha_bar[t].unsqueeze(1)
    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(squezzed_alpha_bar)*x_0 + torch.sqrt((1-squezzed_alpha_bar))*eps

    return x_t, eps

class TabularMLP(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features=num_features + 1, out_features=256)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(in_features=256, out_features=512)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(in_features=512, out_features=768)
        self.act3 = nn.ReLU()

        self.output_layer = nn.Linear(in_features=768, out_features=num_features)


    def forward(self, x, t):
        normalized_t = t/1000
        normalized_t = normalized_t.view(-1, 1)
        combined_tensor = torch.cat([x, normalized_t], dim=1)
        layer1_res = self.layer1(combined_tensor)
        act1_res = self.act1(layer1_res)
        layer2_res = self.layer2(act1_res)
        act2_res = self.act2(layer2_res)
        layer3_res = self.layer3(act2_res)
        act3_res = self.act3(layer3_res)

        return self.output_layer(act3_res)

