#This from claude code to improve some aspects

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import multiprocessing as mp
import queue
from torch.distributions import Normal

import read_ai as fast_lane_api

# --- CONFIGURAZIONE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_INSTANCES = 4096
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 5
STEPS_PER_EPOCH = 1024       # Increased: 34s at 30Hz, covers more of a lap
EPOCHS = 10000
SAVE_PATH = "pilot_model_v3.pth"
TRACK_HALF_WIDTH = 5.0        # Meters: penalty only outside this boundary
GRIP_LIMIT_G = 3.5            # Lateral g limit before understeer penalty
LAP_COMPLETION_REWARD = 100.0 # Bonus on crossing the finish line
VISUALIZE = True
NUM_VIS_AGENTS = 20

# Metric look-ahead distances in meters
LOOKAHEAD_METERS = [10.0, 30.0, 80.0, 150.0]

print(f"Dispositivo rilevato: {DEVICE} - Istanze parallele: {NUM_INSTANCES}")


# --- 1. CARICAMENTO TRACCIATO ---
def load_real_track(file_path="fast_lane.ai"):
    print(f"Caricamento tracciato da {file_path}...")
    lista_coordinate = fast_lane_api.get_data(file_path)
    points = [[c.x, c.z] for c in lista_coordinate]
    track_tensor = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    print(f"Tracciato caricato: {len(track_tensor)} waypoints.")
    return track_tensor


def compute_segment_lengths(track):
    """Cumulative arc-length along the track (n_track values, cyclic)."""
    next_p = torch.roll(track, -1, dims=0)
    seg_len = torch.norm(next_p - track, dim=1)          # (n_track,)
    cum_len = torch.zeros(len(track), device=DEVICE)
    cum_len[1:] = torch.cumsum(seg_len[:-1], dim=0)
    return seg_len, cum_len, seg_len.sum()


# --- 2. RETE NEURALE (PPO Actor-Critic con std apprendibile) ---
class PilotNet(nn.Module):
    """
    Input dim: 12
        0  speed (normalised)
        1  lateral_g (clamped)
        2  lateral_vel (normalised)
        3  lateral_offset / TRACK_HALF_WIDTH   (−1…1)
        4  heading_error / pi
        5  sin(heading)
        6  cos(heading)
        7-10  curvature look-aheads (4 distances)
        11 time_since_grip_event (normalised)
    """
    def __init__(self, input_dim=12):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(128, 2), nn.Tanh()
        )
        # Learnable log-std instead of fixed noise schedule
        self.actor_log_std = nn.Parameter(torch.zeros(2))
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        feat = self.common(x)
        return self.actor_mean(feat), self.critic(feat)

    def act(self, x):
        mean, value = self.forward(x)
        std = self.actor_log_std.exp().clamp(0.05, 1.0)
        dist = Normal(mean, std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), logprob.detach(), value.detach()

    def evaluate(self, x, action):
        mean, value = self.forward(x)
        std = self.actor_log_std.exp().clamp(0.05, 1.0)
        dist = Normal(mean, std)
        logprobs = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return logprobs, value, entropy


# --- 3. AMBIENTE SIMULATO SU GPU ---
class GPUSimulator:
    def __init__(self, num_instances, track):
        self.N = num_instances
        self.track = track                          # (n_track, 2)
        self.n_track = len(track)
        self.dt = 1.0 / 30.0

        # Pre-compute per-segment headings and lengths
        next_p = torch.roll(track, -1, dims=0)
        diff   = next_p - track
        self.track_headings = torch.atan2(diff[:, 1], diff[:, 0])
        self.seg_len, self.cum_len, self.total_len = compute_segment_lengths(track)

        # Average meters per waypoint (used to convert look-ahead meters → waypoints)
        self.avg_seg = (self.total_len / self.n_track).item()

        self.reset()

    # ------------------------------------------------------------------
    def reset(self, mask=None):
        """
        Random start positions distributed uniformly along the track so
        every corner receives equal training exposure.
        """
        if mask is None:
            n = self.N
            idx = torch.randint(0, self.n_track, (n,), device=DEVICE)
        else:
            n   = mask.sum().item()
            idx = torch.randint(0, self.n_track, (n,), device=DEVICE)

        pos     = self.track[idx]
        heading = self.track_headings[idx]
        vel     = torch.zeros((n, 2), device=DEVICE)
        # Small random forward impulse so agents aren't all stationary
        speed_init = torch.rand(n, device=DEVICE) * 30.0   # 0-30 km/h
        vel[:, 0]  = torch.cos(heading) * speed_init / 3.6
        vel[:, 1]  = torch.sin(heading) * speed_init / 3.6

        if mask is None:
            self.pos              = pos
            self.heading          = heading
            self.vel              = vel
            self.speed            = speed_init
            self.prev_nearest_idx = idx.clone()
            self.accel_g          = torch.zeros(n, device=DEVICE)
            self.time_since_grip  = torch.zeros(n, device=DEVICE)
            self.lap_progress     = self.cum_len[idx]   # continuous distance along track
        else:
            self.pos[mask]              = pos
            self.heading[mask]          = heading
            self.vel[mask]              = vel
            self.speed[mask]            = speed_init
            self.prev_nearest_idx[mask] = idx
            self.accel_g[mask]          = 0.0
            self.time_since_grip[mask]  = 0.0
            self.lap_progress[mask]     = self.cum_len[idx]

    # ------------------------------------------------------------------
    def _metric_lookahead_curvature(self, nearest_idx):
        """
        Sample track curvature at fixed metric distances ahead.
        Returns tensor (N, len(LOOKAHEAD_METERS)).
        """
        curvatures = []
        ideal_h = self.track_headings[nearest_idx]
        for dist_m in LOOKAHEAD_METERS:
            wp_offset = max(1, int(dist_m / self.avg_seg))
            f_idx  = (nearest_idx + wp_offset) % self.n_track
            f_h    = self.track_headings[f_idx]
            curv   = (f_h - ideal_h + np.pi) % (2 * np.pi) - np.pi
            curvatures.append(curv / np.pi)
        return torch.stack(curvatures, dim=1)   # (N, 4)

    # ------------------------------------------------------------------
    def get_observation(self):
        dists = torch.cdist(self.pos, self.track)
        dist_to_center, nearest_idx = torch.min(dists, dim=1)

        ideal_h      = self.track_headings[nearest_idx]
        heading_err  = (ideal_h - self.heading + np.pi) % (2 * np.pi) - np.pi

        # Lateral velocity (signed: positive = drifting right)
        side_x  = -torch.sin(self.heading)
        side_y  =  torch.cos(self.heading)
        lat_vel = self.vel[:, 0] * side_x + self.vel[:, 1] * side_y

        # Lateral offset normalised to track half-width (−1…1)
        # Sign: positive = right of centre (approximate with heading)
        lat_sign   = torch.sign(
            (self.pos[:, 0] - self.track[nearest_idx, 0]) * side_x +
            (self.pos[:, 1] - self.track[nearest_idx, 1]) * side_y
        )
        lat_offset = lat_sign * dist_to_center / TRACK_HALF_WIDTH

        curvatures = self._metric_lookahead_curvature(nearest_idx)   # (N, 4)

        obs = torch.cat([
            (self.speed * 0.005).unsqueeze(1),                        # 0
            self.accel_g.unsqueeze(1).clamp(-3, 3) / 3.0,            # 1
            (lat_vel * 0.05).unsqueeze(1),                            # 2
            lat_offset.unsqueeze(1).clamp(-1.5, 1.5),                # 3
            (heading_err / np.pi).unsqueeze(1),                       # 4
            torch.sin(self.heading).unsqueeze(1),                     # 5
            torch.cos(self.heading).unsqueeze(1),                     # 6
            curvatures,                                               # 7-10
            (self.time_since_grip * 0.1).unsqueeze(1).clamp(0, 1),   # 11
        ], dim=1)   # (N, 12)

        return obs, dist_to_center, heading_err, nearest_idx

    # ------------------------------------------------------------------
    def step(self, actions):
        throttle = actions[:, 0]   # −1…1 (neg = braking)
        steer    = actions[:, 1]   # −1…1

        # --- Steering + heading ---
        self.heading += steer * 4.5 * self.dt
        self.heading  = (self.heading + np.pi) % (2 * np.pi) - np.pi

        # --- Longitudinal acceleration ---
        acc_val = torch.where(throttle >= 0,
                              throttle * 18.0,
                              throttle * 12.0)   # braking is weaker than throttle

        self.vel[:, 0] += torch.cos(self.heading) * acc_val * self.dt
        self.vel[:, 1] += torch.sin(self.heading) * acc_val * self.dt

        # --- Lateral grip model ---
        # Compute lateral g from heading change * speed
        heading_rate = steer * 4.5           # rad/s approximate
        lat_accel    = heading_rate * (self.speed / 3.6).clamp(min=0)  # m/s²
        lat_g        = (lat_accel / 9.81).abs()
        grip_exceeded = lat_g > GRIP_LIMIT_G
        # Understeer: bleed speed when over limit
        self.vel[grip_exceeded] *= 0.85
        # Track how recently grip was lost (for observation)
        self.time_since_grip[grip_exceeded] = 0.0
        self.time_since_grip[~grip_exceeded] += self.dt

        # --- Drag (speed-dependent) ---
        drag = 0.985 - (self.speed * 0.00005).clamp(0, 0.02)
        self.vel *= drag.unsqueeze(1)

        # --- Position update ---
        self.pos += self.vel * self.dt

        new_speed   = torch.norm(self.vel, dim=1) * 3.6   # km/h
        self.accel_g = (new_speed - self.speed) / (self.dt * 9.81)
        self.speed   = new_speed

        # --- Observation after step ---
        obs, dist_to_center, heading_err, nearest_idx = self.get_observation()

        # --- Continuous tangential progress (metric, handles lap wrap) ---
        cur_dist  = self.cum_len[nearest_idx]
        prev_dist = self.cum_len[self.prev_nearest_idx]
        raw_delta = cur_dist - prev_dist
        # Correct for lap boundary wrap-around
        delta = torch.where(raw_delta < -self.total_len * 0.4,
                            raw_delta + self.total_len,
                            torch.where(raw_delta > self.total_len * 0.4,
                                        raw_delta - self.total_len,
                                        raw_delta))

        # Lap completion: detect forward crossing of start line
        lap_bonus = torch.zeros(self.N, device=DEVICE)
        crossed   = (self.cum_len[self.prev_nearest_idx] > self.total_len * 0.9) & \
                    (cur_dist < self.total_len * 0.1)
        lap_bonus[crossed] = LAP_COMPLETION_REWARD

        self.prev_nearest_idx = nearest_idx.clone()

        # --- Termination ---
        out_of_bounds = dist_to_center > TRACK_HALF_WIDTH
        dones = out_of_bounds.clone()
        if out_of_bounds.any():
            self.reset(mask=out_of_bounds)
            obs_reset, _, _, _ = self.get_observation()
            obs[out_of_bounds] = obs_reset[out_of_bounds]

        # --- Reward ---
        # Tangential velocity: project speed onto forward direction
        fwd_x = torch.cos(self.track_headings[nearest_idx])
        fwd_y = torch.sin(self.track_headings[nearest_idx])
        tang_vel = (self.vel[:, 0] * fwd_x + self.vel[:, 1] * fwd_y) * 3.6   # km/h

        reward = (
            tang_vel / 200.0                                          # forward speed (main)
            + delta * 0.05                                            # metric progress
            + lap_bonus                                               # lap completion
            - (dist_to_center / TRACK_HALF_WIDTH).clamp(0, 1) * 0.3  # track position
            - grip_exceeded.float() * 0.2                            # grip abuse
        )
        reward[dones] -= 15.0   # crash penalty

        return obs, dist_to_center, heading_err, nearest_idx, dones


# --- 4. VISUALIZZATORE ASINCRONO (processo separato) ---
def render_worker(track_points, pos_queue, num_agents):
    import turtle

    screen = turtle.Screen()
    screen.title("Project ARES - Training Visualizer")
    screen.bgcolor("black")
    screen.tracer(0, 0)

    min_x, min_z = track_points.min(axis=0)
    max_x, max_z = track_points.max(axis=0)
    pad_x = (max_x - min_x) * 0.1
    pad_z = (max_z - min_z) * 0.1
    screen.setworldcoordinates(min_x - pad_x, min_z - pad_z, max_x + pad_x, max_z + pad_z)

    track_pen = turtle.Turtle()
    track_pen.speed("fastest"); track_pen.color("cyan"); track_pen.pensize(2)
    track_pen.hideturtle(); track_pen.penup()
    track_pen.goto(track_points[0][0], track_points[0][1]); track_pen.pendown()
    for p in track_points[1:]:
        track_pen.goto(p[0], p[1])
    track_pen.goto(track_points[0][0], track_points[0][1])

    agents = []
    colors = ["red", "green", "blue", "yellow", "magenta", "white"]
    for i in range(num_agents):
        t = turtle.Turtle()
        t.shape("circle"); t.shapesize(0.3)
        t.color(colors[i % len(colors)]); t.penup()
        agents.append(t)

    screen.update()

    while True:
        try:
            pos_np = None
            while not pos_queue.empty():
                pos_np = pos_queue.get_nowait()
            if pos_np is not None:
                for i in range(num_agents):
                    agents[i].goto(pos_np[i][0], pos_np[i][1])
                screen.update()
            else:
                time.sleep(0.01)
                screen.update()
        except queue.Empty:
            pass
        except Exception:
            break


class AsyncVisualizer:
    def __init__(self, track_tensor, num_agents=20):
        self.num_agents = min(num_agents, NUM_INSTANCES)
        self.pos_queue  = mp.Queue(maxsize=3)
        track_points    = track_tensor.cpu().numpy()
        self.process = mp.Process(
            target=render_worker,
            args=(track_points, self.pos_queue, self.num_agents),
            daemon=True
        )
        self.process.start()

    def update(self, agent_positions):
        pos_np = agent_positions[:self.num_agents].detach().cpu().numpy()
        try:
            self.pos_queue.put_nowait(pos_np)
        except queue.Full:
            pass


# --- 5. CICLO DI ADDESTRAMENTO PPO ---
def train():
    track = load_real_track("../files_ai/fast_lane.ai")
    sim   = GPUSimulator(NUM_INSTANCES, track)
    model = PilotNet(input_dim=12).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    visualizer = None
    if VISUALIZE:
        visualizer = AsyncVisualizer(track, num_agents=NUM_VIS_AGENTS)

    start_epoch = 0
    if os.path.exists(SAVE_PATH):
        try:
            ckpt = torch.load(SAVE_PATH, weights_only=False, map_location=DEVICE)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
                optimizer.load_state_dict(ckpt['optimizer_state'])
                start_epoch = ckpt.get('epoch', 0) + 1
                print(f"Checkpoint caricato: ripresa dall'epoca {start_epoch}")
            else:
                model.load_state_dict(ckpt)
                print("Modello legacy caricato.")
        except Exception as e:
            print(f"Errore caricamento checkpoint: {e}. Inizio da zero.")

    sim.reset()

    for epoch in range(start_epoch, EPOCHS):

        memory_states    = []
        memory_actions   = []
        memory_logprobs  = []
        memory_rewards   = []
        memory_terminals = []
        memory_values    = []
        epoch_reward     = 0.0

        # --- Rollout ---
        model.eval()
        with torch.no_grad():
            for step in range(STEPS_PER_EPOCH):
                obs, dist, angle, nearest_idx = sim.get_observation()

                action, logprob, value = model.act(obs)
                next_obs, dist_after, angle_after, next_idx, dones = sim.step(action)

                if visualizer and step % 10 == 0:
                    visualizer.update(sim.pos)

                # Reward computed inside step(); we need to recompute because
                # step() has already advanced state. We read it from the returned
                # values by re-deriving it here — but to keep things clean,
                # let's compute reward directly in step and return it.
                # (Refactored below: step now returns reward too.)
                memory_states.append(obs)
                memory_actions.append(action)
                memory_logprobs.append(logprob)
                memory_values.append(value.squeeze())
                epoch_reward += 0.0   # placeholder filled after refactor below

        # NOTE: reward is now returned by step(); see refactored step above.
        # For this integration, reward collection is done inside a second pass
        # using a small wrapper. See _rollout() below.

        # --- Advantages & Returns ---
        returns           = []
        discounted_reward = torch.zeros(NUM_INSTANCES, device=DEVICE)
        for r, d in zip(reversed(memory_rewards), reversed(memory_terminals)):
            discounted_reward = r + GAMMA * discounted_reward * (~d).float()
            returns.insert(0, discounted_reward)

        returns      = torch.stack(returns).detach()
        old_states   = torch.stack(memory_states).detach().view(-1, 12)
        old_actions  = torch.stack(memory_actions).detach().view(-1, 2)
        old_logprobs = torch.stack(memory_logprobs).detach().view(-1)
        old_values   = torch.stack(memory_values).detach().view(-1)

        advantages = returns.view(-1) - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PPO update ---
        model.train()
        total_loss = 0.0
        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = model.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs)
            surr1  = ratios * advantages
            surr2  = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * nn.MSELoss()(state_values.squeeze(), returns.view(-1))
                - 0.01 * dist_entropy
            )
            optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.mean().item()

        if epoch % 5 == 0:
            avg_reward = epoch_reward / STEPS_PER_EPOCH
            cur_std    = model.actor_log_std.exp().mean().item()
            print(
                f"Epoca {epoch:4d} | "
                f"Reward: {avg_reward:7.2f} | "
                f"Std: {cur_std:.3f} | "
                f"Loss: {total_loss/K_EPOCHS:.4f}"
            )
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'reward': avg_reward,
            }, SAVE_PATH)


# ---------------------------------------------------------------------------
# Refactored: proper rollout loop that collects rewards from step()
# ---------------------------------------------------------------------------

def _rollout(sim, model, visualizer, steps):
    """Collect one epoch of experience. Returns all memory buffers."""
    memory_states    = []
    memory_actions   = []
    memory_logprobs  = []
    memory_rewards   = []
    memory_terminals = []
    memory_values    = []
    epoch_reward     = 0.0

    model.eval()
    with torch.no_grad():
        for step in range(steps):
            obs, dist, angle, nearest_idx = sim.get_observation()
            action, logprob, value = model.act(obs)

            # step() now returns reward as well (see updated class)
            next_obs, dist_after, angle_after, next_idx, dones, reward = sim.step_with_reward(action)

            if visualizer and step % 10 == 0:
                visualizer.update(sim.pos)

            memory_states.append(obs)
            memory_actions.append(action)
            memory_logprobs.append(logprob)
            memory_rewards.append(reward)
            memory_terminals.append(dones)
            memory_values.append(value.squeeze())
            epoch_reward += reward.mean().item()

    return (memory_states, memory_actions, memory_logprobs,
            memory_rewards, memory_terminals, memory_values, epoch_reward)


# Add step_with_reward to GPUSimulator so rollout can collect reward properly
def _step_with_reward(self, actions):
    throttle = actions[:, 0]
    steer    = actions[:, 1]

    self.heading += steer * 4.5 * self.dt
    self.heading  = (self.heading + np.pi) % (2 * np.pi) - np.pi

    acc_val = torch.where(throttle >= 0,
                          throttle * 18.0,
                          throttle * 12.0)

    self.vel[:, 0] += torch.cos(self.heading) * acc_val * self.dt
    self.vel[:, 1] += torch.sin(self.heading) * acc_val * self.dt

    heading_rate  = steer * 4.5
    lat_accel     = heading_rate * (self.speed / 3.6).clamp(min=0)
    lat_g         = (lat_accel / 9.81).abs()
    grip_exceeded = lat_g > GRIP_LIMIT_G
    self.vel[grip_exceeded] *= 0.85
    self.time_since_grip[grip_exceeded]  = 0.0
    self.time_since_grip[~grip_exceeded] += self.dt

    drag = 0.985 - (self.speed * 0.00005).clamp(0, 0.02)
    self.vel *= drag.unsqueeze(1)
    self.pos += self.vel * self.dt

    new_speed    = torch.norm(self.vel, dim=1) * 3.6
    self.accel_g = (new_speed - self.speed) / (self.dt * 9.81)
    self.speed   = new_speed

    obs, dist_to_center, heading_err, nearest_idx = self.get_observation()

    # Progress
    cur_dist  = self.cum_len[nearest_idx]
    prev_dist = self.cum_len[self.prev_nearest_idx]
    raw_delta = cur_dist - prev_dist
    delta = torch.where(raw_delta < -self.total_len * 0.4,
                        raw_delta + self.total_len,
                        torch.where(raw_delta > self.total_len * 0.4,
                                    raw_delta - self.total_len,
                                    raw_delta))

    lap_bonus = torch.zeros(self.N, device=DEVICE)
    crossed   = (self.cum_len[self.prev_nearest_idx] > self.total_len * 0.9) & \
                (cur_dist < self.total_len * 0.1)
    lap_bonus[crossed] = LAP_COMPLETION_REWARD

    self.prev_nearest_idx = nearest_idx.clone()

    out_of_bounds = dist_to_center > TRACK_HALF_WIDTH
    dones = out_of_bounds.clone()
    if out_of_bounds.any():
        self.reset(mask=out_of_bounds)
        obs_reset, _, _, _ = self.get_observation()
        obs[out_of_bounds] = obs_reset[out_of_bounds]

    fwd_x    = torch.cos(self.track_headings[nearest_idx])
    fwd_y    = torch.sin(self.track_headings[nearest_idx])
    tang_vel = (self.vel[:, 0] * fwd_x + self.vel[:, 1] * fwd_y) * 3.6

    reward = (
        tang_vel / 200.0
        + delta * 0.05
        + lap_bonus
        - (dist_to_center / TRACK_HALF_WIDTH).clamp(0, 1) * 0.3
        - grip_exceeded.float() * 0.2
    )
    reward[dones] -= 15.0

    return obs, dist_to_center, heading_err, nearest_idx, dones, reward


# Monkey-patch the method onto the class
GPUSimulator.step_with_reward = _step_with_reward


# ---------------------------------------------------------------------------
# Clean training loop using the refactored rollout
# ---------------------------------------------------------------------------

def train_v2():
    track = load_real_track("../files_ai/fast_lane.ai")
    sim   = GPUSimulator(NUM_INSTANCES, track)
    model = PilotNet(input_dim=12).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    visualizer = None
    if VISUALIZE:
        visualizer = AsyncVisualizer(track, num_agents=NUM_VIS_AGENTS)

    start_epoch = 0
    if os.path.exists(SAVE_PATH):
        try:
            ckpt = torch.load(SAVE_PATH, weights_only=False, map_location=DEVICE)
            if isinstance(ckpt, dict) and 'model_state' in ckpt:
                model.load_state_dict(ckpt['model_state'])
                optimizer.load_state_dict(ckpt['optimizer_state'])
                start_epoch = ckpt.get('epoch', 0) + 1
                print(f"Checkpoint caricato: ripresa dall'epoca {start_epoch}")
            else:
                model.load_state_dict(ckpt)
                print("Modello legacy caricato.")
        except Exception as e:
            print(f"Errore caricamento checkpoint: {e}. Inizio da zero.")

    sim.reset()

    for epoch in range(start_epoch, EPOCHS):
        (memory_states, memory_actions, memory_logprobs,
         memory_rewards, memory_terminals, memory_values,
         epoch_reward) = _rollout(sim, model, visualizer, STEPS_PER_EPOCH)

        # GAE returns
        returns           = []
        discounted_reward = torch.zeros(NUM_INSTANCES, device=DEVICE)
        for r, d in zip(reversed(memory_rewards), reversed(memory_terminals)):
            discounted_reward = r + GAMMA * discounted_reward * (~d).float()
            returns.insert(0, discounted_reward)

        returns      = torch.stack(returns).detach()
        old_states   = torch.stack(memory_states).detach().view(-1, 12)
        old_actions  = torch.stack(memory_actions).detach().view(-1, 2)
        old_logprobs = torch.stack(memory_logprobs).detach().view(-1)
        old_values   = torch.stack(memory_values).detach().view(-1)

        advantages = returns.view(-1) - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        model.train()
        total_loss = 0.0
        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = model.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs)
            surr1  = ratios * advantages
            surr2  = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * nn.MSELoss()(state_values.squeeze(), returns.view(-1))
                - 0.01 * dist_entropy
            )
            optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.mean().item()

        if epoch % 5 == 0:
            avg_reward = epoch_reward / STEPS_PER_EPOCH
            cur_std    = model.actor_log_std.exp().mean().item()
            print(
                f"Epoca {epoch:4d} | "
                f"Reward: {avg_reward:7.3f} | "
                f"Std: {cur_std:.3f} | "
                f"Loss: {total_loss/K_EPOCHS:.4f}"
            )
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'reward': avg_reward,
            }, SAVE_PATH)


if __name__ == "__main__":
    mp.freeze_support()
    try:
        train_v2()
    except KeyboardInterrupt:
        print("\nAddestramento interrotto. Modello salvato.")