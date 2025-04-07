
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_trajectories(vo, gt, aligned_vo=None, title="Trajectory"):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot(vo[:, 0], vo[:, 1], vo[:, 2], 'o-', label="VO Trajectory")
    
    if aligned_vo is not None:
        ax.plot(aligned_vo[:, 0], aligned_vo[:, 1], aligned_vo[:, 2], 'x--', label="Aligned VO Trajectory")

    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'r', label="Ground Truth")

    ax.set_title(title)
    ax.set_xlabel("X (East)")
    ax.set_ylabel("Y (North)")
    ax.set_zlabel("Z (Up)")
    ax.legend()
    plt.show()

    
def align_trajectories(vo, gt):
    assert vo.shape == gt.shape
    mu_vo = np.mean(vo, axis=0)
    mu_gt = np.mean(gt, axis=0)

    vo_centered = vo - mu_vo
    gt_centered = gt - mu_gt

    H = vo_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    scale = np.trace(R @ H) / np.sum(vo_centered ** 2)
    t = mu_gt.T - scale * R @ mu_vo.T

    aligned_vo = (scale * R @ vo.T).T + t.T
    return aligned_vo, scale

def compute_ate_rmse(vo, gt):
    assert vo.shape == gt.shape
    return np.sqrt(np.mean(np.sum((vo - gt) ** 2, axis=1)))

