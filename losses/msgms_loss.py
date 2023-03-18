import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# Define Prewitt operator:
class Prewitt(nn.Module):
    def __init__(self,device_ids):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
        Gx = torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]) / 3
        Gy = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]) / 3
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1).cuda(device=device_ids[0])
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

class Sobel(nn.Module):
    def __init__(self,device_ids):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0, bias=False)
        G1 = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4
        G2 = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]) / 4
        G3 = torch.tensor([[2.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, -1.0, -2.0]]) / 4
        G4 = torch.tensor([[0.0, -1.0, -2.0], [1.0, 0.0, -1.0], [2.0, 1.0, 0.0]]) / 4
        G = torch.cat([G1.unsqueeze(0), G2.unsqueeze(0),G3.unsqueeze(0), G4.unsqueeze(0)], 0)
        G = G.unsqueeze(1).cuda(device=device_ids[0])
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x


# Define the gradient magnitude similarity map:
def GMS(Ii, Ir, edge_filter, c=0.1):   #c=0.5
    x = torch.mean(Ii, dim=1, keepdim=True)
    y = torch.mean(Ir, dim=1, keepdim=True)
    g_I = edge_filter((x))
    g_Ir = edge_filter((y))
    g_map = (2 * g_I * g_Ir + c) / (g_I**2 + g_Ir**2 + c)
    return g_map


class MSGMS_Loss(nn.Module):
    def __init__(self,device_ids, pool_num = 4):
        super().__init__()
        self.GMS = partial(GMS, edge_filter=Sobel(device_ids))
        self.pool_num = pool_num

    def GMS_loss(self, Ii, Ir):
        return torch.mean(1 - self.GMS(Ii, Ir))

    def forward(self, Ii, Ir):
        total_loss = self.GMS_loss(Ii, Ir)

        for _ in range(self.pool_num):
            Ii = F.avg_pool2d(Ii, kernel_size=2, stride=2)
            Ir = F.avg_pool2d(Ir, kernel_size=2, stride=2)
            total_loss += self.GMS_loss(Ii, Ir)

        return total_loss / int(1+self.pool_num)


