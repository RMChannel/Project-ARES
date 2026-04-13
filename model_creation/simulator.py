import torch
import numpy as np

class GPUSimulator:
    def __init__(self, num_instances, track, device, out_of_bounds_dist=4.0):
        self.N = num_instances
        self.track = track
        self.n_track = len(track)
        self.dt = 1 / 30.0
        self.device = device
        self.out_of_bounds_dist = out_of_bounds_dist

        # Car parameters (from car-sim)
        self.mass = 1000.0
        self.tire_grip = 1.2
        self.wheel_base = 2.5
        self.max_steering_angle = np.radians(30)

        next_p = torch.roll(track, -1, dims=0)
        diff = next_p - track
        self.track_headings = torch.atan2(diff[:, 1], diff[:, 0])

        self.reset()

    def reset(self, mask=None):
        if mask is None:
            self.pos = self.track[0].clone().unsqueeze(0).repeat(self.N, 1)
            self.heading = self.track_headings[0].clone().repeat(self.N)
            self.vel = torch.zeros((self.N, 2), device=self.device)
            self.speed = torch.zeros(self.N, device=self.device)
            self.accel_g = torch.zeros(self.N, device=self.device)
            self.prev_nearest_idx = torch.zeros(self.N, dtype=torch.long, device=self.device)
        else:
            self.pos[mask] = self.track[0]
            self.heading[mask] = self.track_headings[0]
            self.vel[mask] = 0.0
            self.speed[mask] = 0.0
            self.accel_g[mask] = 0.0
            self.prev_nearest_idx[mask] = 0

    def get_observation(self):
        dists = torch.cdist(self.pos, self.track)
        dist_to_center, nearest_idx = torch.min(dists, dim=1)

        ideal_h = self.track_headings[nearest_idx]
        angle_to_center = (ideal_h - self.heading + np.pi) % (2 * np.pi) - np.pi

        side_x = -torch.sin(self.heading)
        side_y = torch.cos(self.heading)
        lat_vel = self.vel[:, 0] * side_x + self.vel[:, 1] * side_y

        look_aheads = [20, 50, 100, 200]
        curvatures = []
        for la in look_aheads:
            f_idx = (nearest_idx + la) % self.n_track
            f_h = self.track_headings[f_idx]
            curv = (f_h - ideal_h + np.pi) % (2 * np.pi) - np.pi
            curvatures.append(curv / np.pi)

        obs = torch.stack([
            self.speed * 0.01,
            self.accel_g,
            lat_vel * 0.1,
            dist_to_center / self.out_of_bounds_dist,
            angle_to_center / np.pi,
            torch.sin(self.heading),
            torch.cos(self.heading),
            *curvatures
        ], dim=1)

        return obs, dist_to_center, angle_to_center, nearest_idx

    def step(self, actions):
        throttle_input = torch.clamp(actions[:, 0], min=0)
        brake_input = torch.clamp(-actions[:, 0], min=0)
        steering_input = actions[:, 1]

        force = throttle_input * 1500.0 - brake_input * 1000.0
        acceleration = force / self.mass
        
        # Ackermann steering
        steering_angle = steering_input * self.max_steering_angle
        # To avoid division by zero or extreme values at low speed
        angular_velocity = self.speed * torch.tan(steering_angle) / self.wheel_base
        
        self.heading += angular_velocity * self.dt
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

        ax = torch.cos(self.heading) * acceleration
        ay = torch.sin(self.heading) * acceleration
        
        self.vel[:, 0] += ax * self.dt
        self.vel[:, 1] += ay * self.dt
        
        # Friction/Drag
        self.vel *= 0.99
        
        self.pos += self.vel * self.dt
        
        new_speed = torch.norm(self.vel, dim=1)
        self.accel_g = (new_speed - self.speed) / (self.dt * 9.81)
        self.speed = new_speed

        obs, dist_to_center, angle_to_center, nearest_idx = self.get_observation()

        out_of_bounds = dist_to_center > self.out_of_bounds_dist
        dones = out_of_bounds.clone()

        if out_of_bounds.any():
            self.reset(mask=out_of_bounds)
            obs_reset, _, _, _ = self.get_observation()
            obs[out_of_bounds] = obs_reset[out_of_bounds]

        return obs, dist_to_center, angle_to_center, nearest_idx, dones
