import gtsam
import numpy as np
from gtsam import Pose3, Rot3, Point3
from gtsam import NonlinearFactorGraph, Values, BetweenFactorPose3, noiseModel
from gtsam import LevenbergMarquardtOptimizer, PriorFactorPose3

def build_pose_graph(vo_trajectory, loop_constraints, noise_sigma=0.1, loop_sigma=0.5):
    '''
    Build and optimize a pose graph using GTSAM.

    Args:
        vo_trajectory (list of (R, t)): Raw visual odometry poses.
        loop_constraints (list of LoopClosureConstraint): Detected loop closures.
        noise_sigma (float): Noise for VO edges.
        loop_sigma (float): Noise for loop closures.

    Returns:
        optimized_values (gtsam.Values): Optimized poses keyed by frame index.
    '''
    
    graph = NonlinearFactorGraph()
    initial = Values()

    # Noise models
    vo_noise = noiseModel.Isotropic.Sigma(6, noise_sigma)
    loop_noise = noiseModel.Isotropic.Sigma(6, loop_sigma)
    prior_noise = noiseModel.Isotropic.Sigma(6, 1e-6)

    # Add prior at frame 0
    R0, t0 = vo_trajectory[0]
    pose0 = Pose3(Rot3(R0), Point3(t0.flatten()))
    graph.add(PriorFactorPose3(0, pose0, prior_noise))
    initial.insert(0, pose0)

    # Add VO edges
    for i in range(1, len(vo_trajectory)):
        if vo_trajectory[i] is None or vo_trajectory[i-1] is None:
            continue

        R_prev, t_prev = vo_trajectory[i-1]
        R_curr, t_curr = vo_trajectory[i]

        # Convert to GTSAM Pose3
        pose_prev = Pose3(Rot3(R_prev), Point3(t_prev.flatten()))
        pose_curr = Pose3(Rot3(R_curr), Point3(t_curr.flatten()))

        # Compute relative pose
        T_prev = pose_prev.inverse()
        T_rel = T_prev.compose(pose_curr)

        graph.add(BetweenFactorPose3(i-1, i, T_rel, vo_noise))
        initial.insert(i, pose_curr)

    # Add loop closures
    for constraint in loop_constraints:
        i = constraint.curr_idx
        j = constraint.matched_idx
        R, t = constraint.relative_pose
        T_loop = Pose3(Rot3(R), Point3(t.flatten()))
        graph.add(BetweenFactorPose3(j, i, T_loop, loop_noise))

        # Insert guess for i if not already inserted
        if not initial.exists(i):
            if i < len(vo_trajectory) and vo_trajectory[i] is not None:
                R_i, t_i = vo_trajectory[i]
                initial.insert(i, Pose3(Rot3(R_i), Point3(t_i.flatten())))
        
        # Insert guess for j if not already inserted
        if not initial.exists(j):
            if j < len(vo_trajectory) and vo_trajectory[j] is not None:
                R_j, t_j = vo_trajectory[j]
                initial.insert(j, Pose3(Rot3(R_j), Point3(t_j.flatten())))

    # Optimize
    optimizer = LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()
    print(f"[DEBUG] Optimizer initialized with {initial.size()} poses")
    print(f"[DEBUG] Final graph has {graph.size()} factors")
    return result
