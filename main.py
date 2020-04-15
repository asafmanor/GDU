from typing import Tuple

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import visdom
from PIL import Image
from tqdm import tqdm


class GuidedFilter(nn.Module):
    def __init__(
        self,
        im_shape: Tuple[int],
        im_channels: int = 1,
        kernel_sym_size: int = 3,
        eps: float = 0.01,
        num_iter: int = 50,
        mean_learnable: bool = False,
        agg_learnable: bool = False,
    ):
        super().__init__()
        if kernel_sym_size % 2 == 0:
            raise ValueError(f"kernel_sym_size should be odd. got {kernel_sym_size}")

        H, W = im_shape
        self.eps = eps
        self.num_iter = num_iter
        self._a = nn.Parameter(torch.Tensor(im_channels, H, W), requires_grad=False)
        self._b = nn.Parameter(torch.Tensor(1, H, W), requires_grad=False)

        # nn.init.normal_(self._a, mean=1.0, std=0.1)
        # nn.init.normal_(self._b, mean=0.0, std=0.1)

        self.mean_filter = self.create_mean_filter(
            im_channels, kernel_sym_size, mean_learnable
        )
        self.agg_filter = self.create_mean_filter(
            im_channels, kernel_sym_size, agg_learnable
        )

    @staticmethod
    def create_mean_filter(channels, kernel_sym_size, learnable):
        fil = nn.Conv2d(
            channels,
            channels,
            (kernel_sym_size, kernel_sym_size),
            groups=channels,
            bias=False,
            padding=kernel_sym_size // 2,
        )
        fil.requires_grad_(learnable)
        # if learnable:
        #     nn.init.normal_(fil.weight, mean=1.0, std=0.01)
        if True:
            nn.init.constant_(fil.weight, 1.0 / kernel_sym_size ** 2)
        return fil

    def __call__(self, im_p: torch.Tensor, im_I: torch.Tensor, gt: torch.Tensor = None):
        self._compute_closed_form_solution(im_p, im_I)
        if gt is not None:
            learnable_params = [x for x in self.parameters() if x.requires_grad]
            opt = torch.optim.Adam(learnable_params, lr=3e-4)

            pbar = tqdm(range(self.num_iter))
            for i in pbar:
                pred = self.pred(im_I)
                loss = torch.nn.functional.mse_loss(pred, gt)
                pbar.set_description(f"MSE loss: {loss}")
                opt.zero_grad()
                loss.backward()
                opt.step()

        return self.pred(im_I)

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

    def pred(self, im_I):
        a = self._a.unsqueeze(0).expand_as(im_I)
        b = self._b.unsqueeze(0).expand_as(im_I)
        return self.agg_filter(a) * im_I + self.agg_filter(b)


def vis_show(x: torch.Tensor, title: str = "", bound: int = 0):
    y = x[:, :, bound : SHAPE[0] - bound, bound : SHAPE[1] - bound]
    y = y - y.min()
    y = y / y.max()
    vis.image(y, {"title": title})


if __name__ == "__main__":
    vis = visdom.Visdom()

    guide = Image.open("data/reference.png")
    depth = Image.open("data/target.png")
    gt_depth = Image.open("data/ground_truth.png")

    # guide = Image.open("data/portrait_max_quality.jpeg")
    # depth = None
    # gt_depth = None

    SHAPE = (640, 480)

    gf = GuidedFilter(
        SHAPE,
        eps=1e-5,
        kernel_sym_size=15,
        num_iter=20,
        mean_learnable=False,
        agg_learnable=True,
    )

    im_transform = T.Compose(
        [T.Grayscale(), T.Resize(SHAPE), T.ToTensor(), lambda x: x.unsqueeze(0),],
    )

    depth_transform = T.Compose(
        [
            T.Resize(SHAPE),
            T.ToTensor(),
            lambda x: x.to(torch.float32),
            lambda x: x.unsqueeze(0),
            lambda x: x / 4,
        ]
    )

    def normalize(x):
        mu = x.mean()
        sigma = x.std()
        y = x - mu
        y = y / sigma
        return y, mu, sigma

    if depth is not None:
        depth = depth_transform(depth)
        gt_depth = depth_transform(gt_depth)
        depth, mu_depth, sigma_depth = normalize(depth)
        gt_depth, mu_gt_depth, sigma_gt_depth = normalize(gt_depth)
        guide = im_transform(guide)
        guide, mu_guide, sigma_guide = normalize(guide)
        res = gf(depth, guide, gt_depth)
        vis_show(gt_depth, "GT", bound=10)
        vis_show(res, "Result", bound=10)

    else:
        guide = im_transform(guide)
        guide, mu_guide, sigma_guide = normalize(guide)
        res = gf(guide, guide)
        vis_show(guide, "Guide", bound=10)
        vis_show(res, "Result", bound=10)
