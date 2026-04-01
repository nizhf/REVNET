from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import torch

COLORS = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "gray": (0.5, 0.5, 0.5),
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
}


def save_pcd_tensor(xyz, color, filename: str):
    pcd = create_pcd_tensor(xyz, color)
    o3d.io.write_point_cloud(filename, pcd)


def create_pcd_tensor(xyz, color):
    pcd = o3d.geometry.PointCloud()
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if isinstance(color, (tuple, list)):
            color = np.ones_like(xyz) * np.array(color)
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def paint_coarse_missing(coarse, missing_num=128, coarse_color=COLORS["cyan"], missing_color=COLORS["red"]):
    color = np.ones(coarse.shape) * np.array(coarse_color)
    color[-missing_num:, :] = np.array(missing_color)
    return color


def paint_coarse_missing_fine(
    coarse, fine, missing_num=128, coarse_color=COLORS["cyan"], fine_color=COLORS["yellow"], missing_color=COLORS["red"]
):
    # (N, 3)
    coarse_missing_color = paint_coarse_missing(coarse, missing_num, coarse_color, missing_color)
    fine_color = np.ones(fine.shape) * np.array(fine_color)
    color = np.concatenate([coarse_missing_color, fine_color], axis=0)
    if isinstance(coarse, torch.Tensor):
        xyz = torch.cat([coarse, fine], dim=0).cpu().numpy()
    else:
        xyz = np.concatenate([coarse, fine], axis=0)
    return xyz, color


def visualize_KITTI(
    path,
    data_list,
    titles=["input", "pred"],
    cmap=["bwr", "autumn"],
    zdir="y",
    xlim=(-1, 1),
    ylim=(-1, 1),
    zlim=(-1, 1),
):
    fig = plt.figure(figsize=(6 * len(data_list), 6))
    cmax = data_list[-1][:, 0].max()

    for i in range(len(data_list)):
        data = data_list[i][:-2048] if i == 1 else data_list[i]
        color = data[:, 0] / cmax
        ax = fig.add_subplot(1, len(data_list), i + 1, projection="3d")
        ax.view_init(30, -120)
        b = ax.scatter(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            zdir=zdir,
            c=color,
            vmin=-1,
            vmax=1,
            cmap=cmap[0],
            s=4,
            linewidth=0.05,
            edgecolors="black",
        )
        ax.set_title(titles[i])

        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0)
    pic_path = path / "visualize.png"
    fig.savefig(pic_path)

    save_pcd_tensor(data_list[0], None, str(path / "input.ply"))
    save_pcd_tensor(data_list[1], None, str(path / "pred.ply"))
    plt.close(fig)
