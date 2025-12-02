import os
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

from ekf_script import EKF_Paper
from standard_ekf import EKF_Standard


# collect stats on a instance-
def collect_instance_distance_stats(nusc, scene_indices=None, max_y=8.0, max_range=120.0):
    if scene_indices is None:
        scene_indices = range(len(nusc.scene))

    stats = []

    for scene_index in scene_indices:
        scene = nusc.scene[scene_index]
        sample_token = scene['first_sample_token']

        # instance_token -> list of distances
        dist_history = {}

        while True:
            sample = nusc.get('sample', sample_token)

            # ego pose from LIDAR_TOP
            lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
            p_ego = np.array(ego_pose['translation'])
            q_ego = Quaternion(ego_pose['rotation'])
            R_ego = q_ego.rotation_matrix.T

            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                if not ann['category_name'].startswith('vehicle.car'):
                    continue

                p_obj = np.array(ann['translation'])
                p_rel = R_ego @ (p_obj - p_ego)
                x_rel, y_rel = p_rel[0], p_rel[1]

                # only cars in front & within lateral window
                if x_rel <= 0 or abs(y_rel) > max_y:
                    continue

                d = np.hypot(x_rel, y_rel)
                if d > max_range:
                    continue

                inst = ann['instance_token']
                dist_history.setdefault(inst, []).append(d)

            if sample['next'] == '':
                break
            sample_token = sample['next']

        # compute stats per instance in this scene
        for inst, dists in dist_history.items():
            dists = np.array(dists)
            if len(dists) < 3:
                continue
            stats.append({
                'scene_index': scene_index,
                'instance_token': inst,
                'min_dist': float(np.min(dists)),
                'max_dist': float(np.max(dists)),
                'mean_dist': float(np.mean(dists)),
                'num_samples': int(len(dists)),
            })

    return stats


# ekf on instance
def run_ekf_on_instance(nusc, ekf, scene_index, instance_token):
    """
    Run EKF on a specific instance_token in a given scene.
    """
    scene = nusc.scene[scene_index]

    def get_ego_pose(sample):
        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        p_ego = np.array(ego_pose['translation'])
        q_ego = Quaternion(ego_pose['rotation'])
        R_ego = q_ego.rotation_matrix.T
        return p_ego, q_ego, R_ego

    def get_state(sample_token, instance_token):
        sample = nusc.get('sample', sample_token)
        p_ego, q_ego, R_ego = get_ego_pose(sample)

        ann = None
        for tok in sample['anns']:
            ann_candidate = nusc.get('sample_annotation', tok)
            if ann_candidate['instance_token'] == instance_token:
                ann = ann_candidate
                break
        if ann is None:
            return None, None

        p_rel = R_ego @ (np.array(ann['translation']) - p_ego)
        yaw_obj = Quaternion(ann['rotation']).yaw_pitch_roll[0]
        yaw_ego = q_ego.yaw_pitch_roll[0]
        return np.array([p_rel[0], p_rel[1], yaw_obj - yaw_ego]), sample['timestamp'] * 1e-6

    def closest_point(pts, center, max_dist=10.0):
        d_xy = np.hypot(pts[0, :] - center[0], pts[1, :] - center[1])
        mask = (d_xy < max_dist) & (pts[2, :] > -2.0) & (pts[2, :] < 3.0)
        if not np.any(mask):
            return None
        idx = np.argmin(d_xy[mask])
        return pts[0, np.where(mask)[0][idx]], pts[1, np.where(mask)[0][idx]]

    def get_measurement(sample_token, instance_token):
        sample = nusc.get('sample', sample_token)
        p_ego, q_ego, R_ego = get_ego_pose(sample)

        ann = None
        for tok in sample['anns']:
            ann_candidate = nusc.get('sample_annotation', tok)
            if ann_candidate['instance_token'] == instance_token:
                ann = ann_candidate
                break
        if ann is None:
            return None

        center_ego = R_ego @ (np.array(ann['translation']) - p_ego)
        yaw_obj = Quaternion(ann['rotation']).yaw_pitch_roll[0]
        yaw_ego = q_ego.yaw_pitch_roll[0]
        theta = yaw_obj - yaw_ego

        # LiDAR
        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_cs = nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])
        pc_lidar = LidarPointCloud.from_file(
            os.path.join(nusc.dataroot, lidar_sd['filename'])
        )
        pc_lidar.transform(
            transform_matrix(lidar_cs['translation'],
                             Quaternion(lidar_cs['rotation']),
                             inverse=False)
        )
        lidar_xy = closest_point(pc_lidar.points[:3, :], center_ego)
        x_lidar, y_lidar = lidar_xy if lidar_xy else (center_ego[0], center_ego[1])

        # Radar
        if 'RADAR_FRONT' in sample['data']:
            radar_sd = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
            radar_cs = nusc.get('calibrated_sensor', radar_sd['calibrated_sensor_token'])
            pc_radar = RadarPointCloud.from_file(
                os.path.join(nusc.dataroot, radar_sd['filename'])
            )
            pc_radar.transform(
                transform_matrix(radar_cs['translation'],
                                 Quaternion(radar_cs['rotation']),
                                 inverse=False)
            )
            radar_xy = closest_point(pc_radar.points[:3, :], center_ego, max_dist=20.0)
            x_radar, y_radar = radar_xy if radar_xy else (center_ego[0], center_ego[1])
        else:
            x_radar, y_radar = center_ego[0], center_ego[1]

        return np.array([x_lidar, y_lidar, x_radar, y_radar, theta])

    # Build trajectory for this instance
    true_states = []
    times = []
    tokens = []

    sample_token = scene['first_sample_token']
    while True:
        state, t = get_state(sample_token, instance_token)
        if state is not None:
            true_states.append(state)
            times.append(t)
            tokens.append(sample_token)
        sample = nusc.get('sample', sample_token)
        if sample['next'] == '':
            break
        sample_token = sample['next']

    true_states = np.array(true_states)
    times = np.array(times)

    if len(true_states) < 3:
        raise RuntimeError("Not enough samples for this instance to run EKF.")

    # Velocity
    dt_seq = np.diff(times, prepend=times[0])
    V = np.zeros(len(true_states))
    for k in range(1, len(true_states)):
        dt = max(dt_seq[k], 1e-3)
        V[k] = np.hypot(true_states[k, 0] - true_states[k-1, 0],
                        true_states[k, 1] - true_states[k-1, 1]) / dt

    x_true = np.column_stack([
        true_states[:, 0],  # x
        true_states[:, 1],  # y
        V,                  # speed
        true_states[:, 2]   # theta
    ])

    # Distance
    dists = np.linalg.norm(true_states[:, :2], axis=1)

    # Measurements
    z_list = []
    for tok in tokens:
        z = get_measurement(tok, instance_token)
        z_list.append(z if z is not None else np.full(5, np.nan))
    z_all = np.array(z_list)

    # Init EKF
    x0 = x_true[0].copy()
    x0[0] += 1.0
    x0[1] += 0.5
    ekf.set_initial_state(x0)

    est_states = []
    for k in range(len(x_true)):
        ekf.predict(dt_seq[k])
        if not np.any(np.isnan(z_all[k])):
            ekf.update(z_all[k], dt_seq[k])
        est_states.append(ekf.phi.copy())

    return times, x_true, np.array(est_states), z_all, dists


def compute_rmse(x_true, est_states):
    err = est_states - x_true
    err_theta = (err[:, 3] + np.pi) % (2 * np.pi) - np.pi
    rmse_x = np.sqrt(np.mean(err[:, 0]**2))
    rmse_y = np.sqrt(np.mean(err[:, 1]**2))
    rmse_v = np.sqrt(np.mean(err[:, 2]**2))
    rmse_th = np.sqrt(np.mean(err_theta**2))
    return rmse_x, rmse_y, rmse_v, rmse_th

def plot_ekf_result(times, x_true, est_states, z_all, title_prefix=""):

    # Only keep samples where object is in front
    front_mask = x_true[:, 0] > 0

    times_f     = times[front_mask]
    x_true_f    = x_true[front_mask]
    est_states_f = est_states[front_mask]
    z_all_f     = z_all[front_mask]

    t = times_f - times_f[0]
    err = est_states_f - x_true_f

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # x position
    axs[0, 0].plot(t, x_true_f[:, 0], label="True", linewidth=2)
    axs[0, 0].plot(t, est_states_f[:, 0], "--", label="EKF", linewidth=2)
    axs[0, 0].scatter(t, z_all_f[:, 0], s=10, alpha=0.5, label="LiDAR")
    axs[0, 0].scatter(t, z_all_f[:, 2], s=10, alpha=0.5, label="Radar")

    axs[0, 0].set_ylabel("x (m)")
    axs[0, 0].set_title(f"{title_prefix} Longitudinal Position")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # y position
    axs[0, 1].plot(t, x_true_f[:, 1], label="True", linewidth=2)
    axs[0, 1].plot(t, est_states_f[:, 1], "--", label="EKF", linewidth=2)
    axs[0, 1].scatter(t, z_all_f[:, 1], s=10, alpha=0.5, label="LiDAR")
    axs[0, 1].scatter(t, z_all_f[:, 3], s=10, alpha=0.5, label="Radar")

    axs[0, 1].set_ylabel("y (m)")
    axs[0, 1].set_title(f"{title_prefix} Lateral Position")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # x error
    axs[1, 0].plot(t, err[:, 0], linewidth=2)
    axs[1, 0].set_ylabel("x error (m)")
    axs[1, 0].set_xlabel("time (s)")
    axs[1, 0].grid(True)

    # y error
    axs[1, 1].plot(t, err[:, 1], linewidth=2)
    axs[1, 1].set_ylabel("y error (m)")
    axs[1, 1].set_xlabel("time (s)")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


# standard to reliability function EKF 
if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot='data', verbose=True)

   
    scene_indices = [0, 1]
    stats = collect_instance_distance_stats(nusc, scene_indices=scene_indices)

    # close and far to test on all distances
    close_cases = [s for s in stats
                   if s['max_dist'] < 20 and s['num_samples'] >= 8]
    mid_cases   = [s for s in stats
                   if 20 <= s['max_dist'] < 50 and s['num_samples'] >= 8]
    far_cases   = [s for s in stats
                   if s['max_dist'] >= 50 and s['num_samples'] >= 8]

    test_cases = []
    if close_cases:
        print("Close cases found:")
        test_cases.append(close_cases[0])
    if mid_cases:
        print("Mid-range cases found:")
        test_cases.append(mid_cases[0])
    if far_cases:
        print("Far cases found:")
        test_cases.append(far_cases[0])

    print(test_cases)
    
    Q = np.diag([0.5, 0.05, 0.05, 0.05])
    R = np.diag([0.05, 0.05, 0.05, 0.05, 0.05])

    # Reliability parameters
    a1 = 0.10
    b1 = 0.55
    X_lidar_reli = 25.0
    Y_lidar_reli = 0.0

    a2 = 0.07
    b2 = 0.35
    X_radar_reli = 45.0
    Y_radar_reli = 0.0

    # Loop over test instances and compare
    for case in test_cases:
        scene_idx = case['scene_index']
        inst_token = case['instance_token']

        print("\n" + "=" * 70)
        print(f"Scene {scene_idx}, instance {inst_token}")
        
        # Standard EKF
        ekf_std = EKF_Standard(Q, R,a1, b1, X_lidar_reli, Y_lidar_reli,a2, b2, X_radar_reli, Y_radar_reli)
        times, x_true, est_std, z_all, dists = run_ekf_on_instance(nusc, ekf_std, scene_idx, inst_token)
        plot_ekf_result(times, x_true, est_std, z_all, title_prefix="Standard EKF:")

        rmse_std = compute_rmse(x_true, est_std)

        # Reliabitity EKF
        ekf_rel = EKF_Paper(Q, R,a1, b1, X_lidar_reli, Y_lidar_reli,a2, b2, X_radar_reli, Y_radar_reli)
        times2, x_true2, est_rel, z_all2, dists2 = run_ekf_on_instance(nusc, ekf_rel, scene_idx, inst_token)
        plot_ekf_result(times2, x_true2, est_rel, z_all2, title_prefix="Reliability EKF:")

        rmse_rel = compute_rmse(x_true2, est_rel)

        # Print RMSE comparison
        print("Standard EKF RMSE:")
        print(f"  x: {rmse_std[0]:.3f} m")
        print(f"  y: {rmse_std[1]:.3f} m")
        print(f"  V: {rmse_std[2]:.3f} m/s")
        print(f"  θ: {rmse_std[3]:.3f} rad")

        print("Reliability EKF RMSE:")
        print(f"  x: {rmse_rel[0]:.3f} m")
        print(f"  y: {rmse_rel[1]:.3f} m")
        print(f"  V: {rmse_rel[2]:.3f} m/s")
        print(f"  θ: {rmse_rel[3]:.3f} rad")