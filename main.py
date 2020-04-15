from typing import Tuple

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import visdom
from PIL import Image
from tqdm import tqdm


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class GuidedFilter(nn.Module):
    def __init__(
        self,
        im_shape: Tuple[int],
        im_channels: int = 1,
        kernel_sym_size: int = 3,
        eps: float = 0.01,
        num_iter: int = 50,
        guide_learnable: bool = False,
        a_b_learnable: bool = False,
    ):
        super().__init__()
        if kernel_sym_size % 2 == 0:
            raise ValueError(f"kernel_sym_size should be odd. got {kernel_sym_size}")

        H, W = im_shape
        self.H = H
        self.W = W
        self.eps = eps
        self.num_iter = num_iter
        self._a = nn.Parameter(
            torch.Tensor(im_channels, H, W), requires_grad=a_b_learnable
        )
        self._b = nn.Parameter(torch.Tensor(1, H, W), requires_grad=a_b_learnable)

        if guide_learnable:
            self.guide_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=True),
                ResidualBlock(32, 32),
                nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True),
            )
        else:
            self.guide_encoder = nn.Identity()

        self.mean_filter = self.create_mean_filter(im_channels, kernel_sym_size)
        self.agg_filter = self.create_mean_filter(im_channels, kernel_sym_size)

    @staticmethod
    def create_mean_filter(channels, kernel_sym_size):
        fil = nn.Conv2d(
            channels,
            channels,
            (kernel_sym_size, kernel_sym_size),
            groups=channels,
            bias=False,
            padding=kernel_sym_size // 2,
        )
        fil.requires_grad_(False)
        nn.init.constant_(fil.weight, 1.0 / kernel_sym_size ** 2)
        return fil

    def __call__(self, im_p: torch.Tensor, im_I: torch.Tensor, gt: torch.Tensor = None):
        guide = im_I
        self._compute_closed_form_solution(im_p, im_I)
        gt = im_p if gt is None else gt
        learnable_params = [x for x in self.parameters() if x.requires_grad]
        if learnable_params:
            opt = torch.optim.Adam(learnable_params, lr=3e-4)
            pbar = tqdm(range(self.num_iter))
            for i in pbar:
                guide = self.guide_encoder(im_I)
                pred = self.pred(guide)
                l1_loss = nn.functional.l1_loss(pred, gt)
                ds_pred = nn.functional.interpolate(pred, (self.H // 2, self.W // 2))
                tv_loss = self.tv_loss(ds_pred, 0.25)
                loss = l1_loss + tv_loss
                pbar.set_description(f"L1 loss: {l1_loss}")
                loss.backward()
                opt.step()
                opt.zero_grad()

        return self.pred(guide)

    def _compute_closed_form_solution(self, im_p, im_I):
        corr_II = self.mean_filter(im_I * im_I)
        corr_Ip = self.mean_filter(im_I * im_p)
        mean_I = self.mean_filter(im_I)
        mean_p = self.mean_filter(im_p)
        eps = self.eps

        var_I = corr_II - mean_I * mean_I
        a = (corr_Ip - mean_I * mean_p) / (var_I + eps)
        b = mean_p - a * mean_I

        self._a.data = a.squeeze(0)
        self._b.data = b.squeeze(0)

    def _compute_a_b_loss(self, im_p, im_I):
        corr_II = self.mean_filter(im_I * im_I)
        corr_pp = self.mean_filter(im_p * im_p)
        corr_Ip = self.mean_filter(im_I * im_p)
        mean_I = self.mean_filter(im_I)
        mean_p = self.mean_filter(im_p)
        eps = self.eps

        a = self._a.unsqueeze(0).expand_as(im_I)
        b = self._b.unsqueeze(0).expand_as(im_I)
        a_sq = a ** 2
        b_sq = b ** 2

        loss = (
            b_sq
            + a_sq * (eps + corr_II)
            + 2 * a * b * mean_I
            - 2 * a * corr_Ip
            - 2 * b * mean_p
            + corr_pp
        )
        return torch.mean(loss)

    @staticmethod
    def tv_loss(pred, weight):
        tv_h = ((pred[:, :, 1:, :] - pred[:, :, :-1, :]).pow(2)).mean()
        tv_w = ((pred[:, :, :, 1:] - pred[:, :, :, :-1]).pow(2)).mean()
        return weight * (tv_h + tv_w)

    def pred(self, guide):
        a = self._a.unsqueeze(0).expand_as(guide)
        b = self._b.unsqueeze(0).expand_as(guide)
        return self.agg_filter(a) * guide + self.agg_filter(b)


def vis_show(x: torch.Tensor, title: str = "", bound: int = 0):
    y = x[0, :, bound : SHAPE[0] - bound, bound : SHAPE[1] - bound]
    y = y - y.min()
    y = y / y.max()
    vis.image(y, win=title, opts={"caption": title})


if __name__ == "__main__":
    vis = visdom.Visdom(port=6006)

    guide = Image.open("data/reference.png")
    depth = Image.open("data/target.png")
    gt_depth = Image.open("data/ground_truth.png")

    # guide = Image.open("data/portrait_max_quality.jpeg")
    # depth = None
    # gt_depth = None

    SHAPE = (640, 480)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gf = GuidedFilter(
        SHAPE,
        eps=1e-5,
        kernel_sym_size=15,
        num_iter=1000,
        guide_learnable=True,
        a_b_learnable=False,
    ).to(device)

    im_transform = T.Compose(
        [
            T.Grayscale(),
            T.Resize(SHAPE),
            T.ToTensor(),
            lambda x: x.to(dtype=torch.float32, device=device),
            lambda x: x.unsqueeze(0),
        ],
    )

    depth_transform = T.Compose(
        [
            T.Resize(SHAPE),
            T.ToTensor(),
            lambda x: x.to(dtype=torch.float32, device=device),
            lambda x: x.unsqueeze(0),
            lambda x: x / 4,
        ]
    )

    def normalize(x, mu=None, sigma=None):
        mu = x.mean() if mu is None else mu
        sigma = x.std() if sigma is None else sigma
        y = x - mu
        y = y / sigma
        return y, mu, sigma

    if depth is not None:
        gt_depth = depth_transform(gt_depth)
        depth = depth_transform(depth)
        gt_depth, mu_gt_depth, sigma_gt_depth = normalize(gt_depth)
        depth, mu_depth, sigma_depth = normalize(depth)

        guide = im_transform(guide)
        guide, mu_guide, sigma_guide = normalize(guide)

        res = gf(depth, guide, gt=gt_depth)

        vis_show(guide, "Guide", bound=10)
        vis_show(gt_depth, "GT", bound=10)
        vis_show(depth, "Depth", bound=10)
        vis_show(res, "Result", bound=10)
        err = torch.abs(res - gt_depth)
        vis_show(err, "Error", bound=10)
        vis.text(f"MAE: {torch.mean(err)}", win="MAE")

    else:
        guide = im_transform(guide)
        guide, mu_guide, sigma_guide = normalize(guide)
        res = gf(guide, guide)
        vis_show(guide, "Guide", bound=10)
        vis_show(res, "Result", bound=10)
