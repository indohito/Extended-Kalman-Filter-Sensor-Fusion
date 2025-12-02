import os
import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from ekf_script import EKF_Paper
from standard_ekf import EKF_Standard

def run_ekf_on_nuscenes_leadcar(nusc, ekf, scene_index=0):
    """Run EKF on a nuScenes scene tracking a lead car using LiDAR & RADAR."""
    
    scene = nusc.scene[scene_index]
    
    def get_ego_pose(sample):
        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        p_ego = np.array(ego_pose['translation'])
        q_ego = Quaternion(ego_pose['rotation'])
        R_ego = q_ego.rotation_matrix.T
        return p_ego, q_ego, R_ego
    
    def find_lead_car():
        sample_token = scene['first_sample_token']
        while True:
            sample = nusc.get('sample', sample_token)
            p_ego, _, R_ego = get_ego_pose(sample)
            
            candidates = []
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                if not ann['category_name'].startswith('vehicle.car'):
                    continue
                
                p_rel = R_ego @ (np.array(ann['translation']) - p_ego)
                if p_rel[0] > 0 and abs(p_rel[1]) < 5.0:
                    candidates.append((ann['instance_token'], p_rel[0]))
            
            if candidates:
                return min(candidates, key=lambda t: t[1])[0]  # Return closest car
            
            if sample['next'] == '':
                break
            sample_token = sample['next']
        raise RuntimeError("No lead car found in scene.")
    
    def get_lead_state(sample_token, instance_token):
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
        pc_lidar = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_sd['filename']))
        pc_lidar.transform(transform_matrix(lidar_cs['translation'], 
            Quaternion(lidar_cs['rotation']), inverse=False))
        lidar_xy = closest_point(pc_lidar.points[:3, :], center_ego)
        x_lidar, y_lidar = lidar_xy if lidar_xy else (center_ego[0], center_ego[1])
        
        # Radar
        if 'RADAR_FRONT' in sample['data']:
            radar_sd = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
            radar_cs = nusc.get('calibrated_sensor', radar_sd['calibrated_sensor_token'])
            pc_radar = RadarPointCloud.from_file(os.path.join(nusc.dataroot, radar_sd['filename']))
            pc_radar.transform(transform_matrix(radar_cs['translation'], 
                Quaternion(radar_cs['rotation']), inverse=False))
            radar_xy = closest_point(pc_radar.points[:3, :], center_ego, max_dist=20.0)
            x_radar, y_radar = radar_xy if radar_xy else (center_ego[0], center_ego[1])
        else:
            x_radar, y_radar = center_ego[0], center_ego[1]
        
        return np.array([x_lidar, y_lidar, x_radar, y_radar, theta])
    
    # Find lead car and build trajectory
    #lead_token = find_target_car(mode='farthest', min_x=30.0, max_y=5.0)
    lead_token = find_lead_car()
   
    true_states = []
    times = []
    tokens = []
    
    sample_token = scene['first_sample_token']
    while True:
        state, t = get_lead_state(sample_token, lead_token)
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
    
    # Compute velocity
    dt_seq = np.diff(times, prepend=times[0])
    V = np.zeros(len(true_states))
    for k in range(1, len(true_states)):
        dt = max(dt_seq[k], 1e-3)
        V[k] = np.hypot(true_states[k, 0] - true_states[k-1, 0], 
                       true_states[k, 1] - true_states[k-1, 1]) / dt
    
    x_true = np.column_stack([true_states[:, 0], true_states[:, 1], V, true_states[:, 2]])
    
    # get distances for reliability functions
    dists = np.linalg.norm(true_states[:, :2], axis=1)

    # Get measurements
    z_list = []
    for tok in tokens:
        z = get_measurement(tok, lead_token)
        z_list.append(z if z is not None else np.full(5, np.nan))
    z_all = np.array(z_list)
    
    # Run EKF
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

    est_states = np.array(est_states)
    
    return times, x_true, est_states, z_all, dists



def plot_results(times, x_true, est_states, z_all):
    """Plot results and print RMSE"""
    t = times - times[0]
    

    # RMSE
    err = est_states - x_true
    err_theta = (err[:, 3] + np.pi) % (2 * np.pi) - np.pi
    print("RMSE:")
    print(f"  x: {np.sqrt(np.mean(err[:, 0]**2)):.3f} m")
    print(f"  y: {np.sqrt(np.mean(err[:, 1]**2)):.3f} m")
    print(f"  V: {np.sqrt(np.mean(err[:, 2]**2)):.3f} m/s")
    print(f"  θ: {np.sqrt(np.mean(err_theta**2)):.3f} rad")
    
    # Plots
    plt.figure(figsize=(10,4))
    plt.plot(t, dists, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Distance to lead car (m)")
    plt.title("True ego → lead vehicle distance over time")
    plt.grid(True)
    plt.tight_layout()

    print("Distance stats (m):")
    print(f"  min:   {np.min(dists):.3f}")
    print(f"  max:   {np.max(dists):.3f}")
    print(f"  mean:  {np.mean(dists):.3f}")
    print(f"  med:   {np.median(dists):.3f}")
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    
    axs[0, 0].plot(t, x_true[:, 0], label='True', linewidth=2)
    axs[0, 0].plot(t, est_states[:, 0], '--', label='EKF', linewidth=2)
    axs[0, 0].scatter(t, z_all[:, 0], s=10, alpha=0.5, label='LiDAR')
    axs[0, 0].scatter(t, z_all[:, 2], s=10, alpha=0.5, label='Radar')
    axs[0, 0].set_ylabel('x (m)')
    axs[0, 0].set_title('Longitudinal Position')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(t, x_true[:, 1], label='True', linewidth=2)
    axs[0, 1].plot(t, est_states[:, 1], '--', label='EKF', linewidth=2)
    axs[0, 1].scatter(t, z_all[:, 1], s=10, alpha=0.5, label='LiDAR')
    axs[0, 1].scatter(t, z_all[:, 3], s=10, alpha=0.5, label='Radar')
    axs[0, 1].set_ylabel('y (m)')
    axs[0, 1].set_title('Lateral Position')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(t, err[:, 0])
    axs[1, 0].set_ylabel('x error (m)')
    axs[1, 0].set_xlabel('time (s)')
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(t, err[:, 1])
    axs[1, 1].set_ylabel('y error (m)')
    axs[1, 1].set_xlabel('time (s)')
    axs[1, 1].grid(True)
    
    plt.tight_layout()


if __name__ == "__main__":
    nusc = NuScenes(version='v1.0-mini', dataroot='data', verbose=True)
    
    # EKF parameters
    Q = np.diag([0.5, 0.05, 0.05, 0.05])
    R = np.diag([0.05, 0.05, 0.05, 0.05, 0.05])
    #Q = np.diag([0.05, 0.05, 0.02, 0.01])
    #R = np.diag([0.5, 0.1, 1.5, 3.0, 0.02])

    a1 = 0.05
    b1 = 0.05
    X_lidar_reli = 30
    X_radar_reli = 80
    a2 = .06
    b2 = .25
    Y_lidar_reli = 20
    Y_radar_reli = 80

    #ekf = EKF_Standard(Q, R, a1, b1, X_lidar_reli, Y_lidar_reli, a2, b2, X_radar_reli, Y_radar_reli)
    ekf = EKF_Paper(Q, R, a1, b1, X_lidar_reli, Y_lidar_reli, a2, b2, X_radar_reli, Y_radar_reli)
    

    times, x_true, est_states, z_all, dists = run_ekf_on_nuscenes_leadcar(nusc, ekf, scene_index=0)

    print(f"\nProcessed {len(times)} timesteps")
    plot_results(times, x_true, est_states, z_all)
    plt.show()
