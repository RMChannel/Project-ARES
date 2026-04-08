import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import multiprocessing as mp
import queue
from torch.distributions import Normal

import read_ai as fast_lane_api

# --- CONFIGURAZIONE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Memory budget (7.6 GB GPU):
#   rollout buffer = NUM_INSTANCES * STEPS_PER_EPOCH * 12 * 4 bytes
#   512 * 512 * 12 * 4 = ~12 MB  →  safe; mini-batch PPO keeps gradients small
NUM_INSTANCES         = 512
STEPS_PER_EPOCH       = 512
MINI_BATCH_SIZE       = 4096   # PPO update chunks; adjust down if still OOM
LR                    = 3e-4
GAMMA                 = 0.99
EPS_CLIP              = 0.2
K_EPOCHS              = 3      # ridotto da 5 per evitare overfitting sul buffer
EPOCHS                = 10000
SAVE_PATH             = "pilot_model_v3.pth"
TRACK_HALF_WIDTH      = 7.0    # metres – allargato da 5.0 per geometria reale
GRIP_LIMIT_G          = 3.5    # lateral g before understeer penalty
LAP_COMPLETION_REWARD = 100.0  # reward on crossing the finish line
MILESTONE_METERS      = 500.0  # bonus intermedio ogni N metri
MILESTONE_REWARD      = 5.0    # bonus per milestone
VISUALIZE             = True
NUM_VIS_AGENTS        = 20

# Metric look-ahead distances (metres)
LOOKAHEAD_METERS = [10.0, 30.0, 80.0, 150.0]

print(f"Dispositivo rilevato: {DEVICE} - Istanze parallele: {NUM_INSTANCES}")


# ---------------------------------------------------------------------------
# 1. TRACK LOADING
# ---------------------------------------------------------------------------
def load_real_track(file_path="fast_lane.ai"):
    print(f"Caricamento tracciato da {file_path}...")
    lista_coordinate = fast_lane_api.get_data(file_path)
    points = [[c.x, c.z] for c in lista_coordinate]
    track_tensor = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    print(f"Tracciato caricato: {len(track_tensor)} waypoints.")
    return track_tensor


def compute_segment_lengths(track):
    """Cumulative arc-length along the track (cyclic)."""
    next_p  = torch.roll(track, -1, dims=0)
    seg_len = torch.norm(next_p - track, dim=1)
    cum_len = torch.zeros(len(track), device=DEVICE)
    cum_len[1:] = torch.cumsum(seg_len[:-1], dim=0)
    return seg_len, cum_len, seg_len.sum()


# ---------------------------------------------------------------------------
# 2. NEURAL NETWORK  (input_dim = 12)
# ---------------------------------------------------------------------------
class PilotNet(nn.Module):
    """
    Inputs (12):
      0  speed (normalised)
      1  lateral_g (clamped, normalised)
      2  lateral_vel (normalised)
      3  lateral_offset / TRACK_HALF_WIDTH  → [-1, 1]
      4  heading_error / pi
      5  sin(heading)
      6  cos(heading)
      7-10  curvature look-aheads at 10/30/80/150 m
      11 time_since_grip_event (normalised, clamped)
    """
    def __init__(self, input_dim=12):
        super().__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
        )
        self.actor_mean    = nn.Sequential(nn.Linear(128, 2), nn.Tanh())
        self.actor_log_std = nn.Parameter(torch.zeros(2))  # learnable std
        self.critic        = nn.Linear(128, 1)

    def forward(self, x):
        feat = self.common(x)
        return self.actor_mean(feat), self.critic(feat)

    def act(self, x):
        mean, value = self.forward(x)
        std   = self.actor_log_std.exp().clamp(0.05, 1.0)
        dist  = Normal(mean, std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), logprob.detach(), value.detach()

    def evaluate(self, x, action):
        mean, value = self.forward(x)
        std     = self.actor_log_std.exp().clamp(0.05, 1.0)
        dist    = Normal(mean, std)
        logprobs = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return logprobs, value, entropy


# ---------------------------------------------------------------------------
# 3. GPU SIMULATOR
# ---------------------------------------------------------------------------
class GPUSimulator:
    def __init__(self, num_instances, track):
        self.N       = num_instances
        self.track   = track
        self.n_track = len(track)
        self.dt      = 1.0 / 30.0

        next_p = torch.roll(track, -1, dims=0)
        diff   = next_p - track
        self.track_headings          = torch.atan2(diff[:, 1], diff[:, 0])
        self.seg_len, self.cum_len, self.total_len = compute_segment_lengths(track)
        self.avg_seg = (self.total_len / self.n_track).item()

        # Curriculum spawn: fraction of track to spawn on (0.2 → 1.0)
        self.spawn_fraction = 0.2

        self.reset()

    # ------------------------------------------------------------------
    def reset(self, mask=None):
        """Random start positions – curriculum: only on first spawn_fraction of track."""
        max_wp = max(1, int(self.n_track * self.spawn_fraction))
        if mask is None:
            n   = self.N
            idx = torch.randint(0, max_wp, (n,), device=DEVICE)
        else:
            n   = int(mask.sum().item())
            idx = torch.randint(0, max_wp, (n,), device=DEVICE)

        pos     = self.track[idx]
        heading = self.track_headings[idx]
        vel     = torch.zeros((n, 2), device=DEVICE)
        speed_init       = torch.rand(n, device=DEVICE) * 30.0  # 0-30 km/h
        vel[:, 0]        = torch.cos(heading) * speed_init / 3.6
        vel[:, 1]        = torch.sin(heading) * speed_init / 3.6

        if mask is None:
            self.pos              = pos
            self.heading          = heading
            self.vel              = vel
            self.speed            = speed_init
            self.prev_nearest_idx = idx.clone()
            self.accel_g          = torch.zeros(n, device=DEVICE)
            self.time_since_grip  = torch.zeros(n, device=DEVICE)
            self.cumulative_progress = torch.zeros(n, device=DEVICE)
            self.milestone_count     = torch.zeros(n, dtype=torch.long, device=DEVICE)
        else:
            self.pos[mask]              = pos
            self.heading[mask]          = heading
            self.vel[mask]              = vel
            self.speed[mask]            = speed_init
            self.prev_nearest_idx[mask] = idx
            self.accel_g[mask]          = 0.0
            self.time_since_grip[mask]  = 0.0
            self.cumulative_progress[mask] = 0.0
            self.milestone_count[mask]     = 0

    # ------------------------------------------------------------------
    def _metric_lookahead_curvature(self, nearest_idx):
        ideal_h    = self.track_headings[nearest_idx]
        curvatures = []
        for dist_m in LOOKAHEAD_METERS:
            wp_offset = max(1, int(dist_m / self.avg_seg))
            f_idx = (nearest_idx + wp_offset) % self.n_track
            f_h   = self.track_headings[f_idx]
            curv  = (f_h - ideal_h + np.pi) % (2 * np.pi) - np.pi
            curvatures.append(curv / np.pi)
        return torch.stack(curvatures, dim=1)  # (N, 4)

    # ------------------------------------------------------------------
    def get_observation(self):
        dists = torch.cdist(self.pos, self.track)
        dist_to_center, nearest_idx = torch.min(dists, dim=1)

        ideal_h     = self.track_headings[nearest_idx]
        heading_err = (ideal_h - self.heading + np.pi) % (2 * np.pi) - np.pi

        side_x  = -torch.sin(self.heading)
        side_y  =  torch.cos(self.heading)
        lat_vel = self.vel[:, 0] * side_x + self.vel[:, 1] * side_y

        lat_sign   = torch.sign(
            (self.pos[:, 0] - self.track[nearest_idx, 0]) * side_x +
            (self.pos[:, 1] - self.track[nearest_idx, 1]) * side_y
        )
        lat_offset = lat_sign * dist_to_center / TRACK_HALF_WIDTH

        curvatures = self._metric_lookahead_curvature(nearest_idx)

        obs = torch.cat([
            (self.speed * 0.005).unsqueeze(1),
            self.accel_g.unsqueeze(1).clamp(-3, 3) / 3.0,
            (lat_vel * 0.05).unsqueeze(1),
            lat_offset.unsqueeze(1).clamp(-1.5, 1.5),
            (heading_err / np.pi).unsqueeze(1),
            torch.sin(self.heading).unsqueeze(1),
            torch.cos(self.heading).unsqueeze(1),
            curvatures,
            (self.time_since_grip * 0.1).unsqueeze(1).clamp(0, 1),
        ], dim=1)  # (N, 12)

        return obs, dist_to_center, heading_err, nearest_idx

    # ------------------------------------------------------------------
    def step_with_reward(self, actions):
        throttle = actions[:, 0]
        steer    = actions[:, 1]

        # Heading
        self.heading += steer * 4.5 * self.dt
        self.heading  = (self.heading + np.pi) % (2 * np.pi) - np.pi

        # Longitudinal
        acc_val = torch.where(throttle >= 0, throttle * 18.0, throttle * 12.0)
        self.vel[:, 0] += torch.cos(self.heading) * acc_val * self.dt
        self.vel[:, 1] += torch.sin(self.heading) * acc_val * self.dt

        # Grip model
        heading_rate  = steer * 4.5
        lat_accel     = heading_rate * (self.speed / 3.6).clamp(min=0)
        lat_g         = (lat_accel / 9.81).abs()
        grip_exceeded = lat_g > GRIP_LIMIT_G
        self.vel[grip_exceeded] *= 0.85
        self.time_since_grip[grip_exceeded]  = 0.0
        self.time_since_grip[~grip_exceeded] += self.dt

        # Drag
        drag = 0.985 - (self.speed * 0.00005).clamp(0, 0.02)
        self.vel *= drag.unsqueeze(1)
        self.pos += self.vel * self.dt

        new_speed    = torch.norm(self.vel, dim=1) * 3.6
        self.accel_g = (new_speed - self.speed) / (self.dt * 9.81)
        self.speed   = new_speed

        obs, dist_to_center, heading_err, nearest_idx = self.get_observation()

        # Metric progress (handles lap wrap-around)
        cur_dist  = self.cum_len[nearest_idx]
        prev_dist = self.cum_len[self.prev_nearest_idx]
        raw_delta = cur_dist - prev_dist
        delta = torch.where(
            raw_delta < -self.total_len * 0.4, raw_delta + self.total_len,
            torch.where(raw_delta > self.total_len * 0.4,
                        raw_delta - self.total_len, raw_delta)
        )

        # Lap completion bonus
        lap_bonus = torch.zeros(self.N, device=DEVICE)
        crossed   = (self.cum_len[self.prev_nearest_idx] > self.total_len * 0.9) & \
                    (cur_dist < self.total_len * 0.1)
        lap_bonus[crossed] = LAP_COMPLETION_REWARD

        # Milestone bonus: +MILESTONE_REWARD every MILESTONE_METERS of cumulative progress
        self.cumulative_progress += delta.clamp(min=0)
        new_milestone_count = (self.cumulative_progress / MILESTONE_METERS).long()
        milestone_bonus = (new_milestone_count - self.milestone_count).clamp(min=0).float() * MILESTONE_REWARD
        self.milestone_count = torch.max(self.milestone_count, new_milestone_count)

        self.prev_nearest_idx = nearest_idx.clone()

        # Termination
        out_of_bounds = dist_to_center > TRACK_HALF_WIDTH
        dones = out_of_bounds.clone()
        if out_of_bounds.any():
            self.reset(mask=out_of_bounds)
            obs_reset, _, _, _ = self.get_observation()
            obs[out_of_bounds] = obs_reset[out_of_bounds]

        # Reward — denser signal
        fwd_x    = torch.cos(self.track_headings[nearest_idx])
        fwd_y    = torch.sin(self.track_headings[nearest_idx])
        tang_vel = (self.vel[:, 0] * fwd_x + self.vel[:, 1] * fwd_y) * 3.6

        reward = (
            tang_vel / 100.0                                           # doubled from /200
            + delta.clamp(min=0) * 0.10                                # doubled from 0.05
            + lap_bonus
            + milestone_bonus                                          # NEW: intermedio
            - (dist_to_center / TRACK_HALF_WIDTH).clamp(0, 1) * 0.3
            - grip_exceeded.float() * 0.2
        )
        reward[dones] -= 15.0

        # Store per-step metrics for logging
        self._last_grip_exceeded = grip_exceeded
        self._last_lap_crossed   = crossed
        self._last_dist          = dist_to_center

        return obs, dist_to_center, heading_err, nearest_idx, dones, reward


# ---------------------------------------------------------------------------
# 4. ASYNC VISUALIZER
# ---------------------------------------------------------------------------
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

    pen = turtle.Turtle()
    pen.speed("fastest"); pen.color("cyan"); pen.pensize(2); pen.hideturtle()
    pen.penup(); pen.goto(track_points[0][0], track_points[0][1]); pen.pendown()
    for p in track_points[1:]:
        pen.goto(p[0], p[1])
    pen.goto(track_points[0][0], track_points[0][1])

    colors  = ["red", "green", "blue", "yellow", "magenta", "white"]
    agents  = []
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
        self.process    = mp.Process(
            target=render_worker,
            args=(track_tensor.cpu().numpy(), self.pos_queue, self.num_agents),
            daemon=True,
        )
        self.process.start()

    def update(self, agent_positions):
        try:
            self.pos_queue.put_nowait(
                agent_positions[:self.num_agents].detach().cpu().numpy()
            )
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# 5. ROLLOUT HELPER
# ---------------------------------------------------------------------------
def _rollout(sim, model, visualizer, steps):
    memory_states    = []
    memory_actions   = []
    memory_logprobs  = []
    memory_rewards   = []
    memory_terminals = []
    memory_values    = []
    epoch_reward     = 0.0

    # Accumulatori per metriche
    total_grip_count = 0
    total_lap_count  = 0
    total_dist_sum   = 0.0
    total_steps_n    = 0

    model.eval()
    with torch.no_grad():
        for step in range(steps):
            obs, _, _, _ = sim.get_observation()
            action, logprob, value = model.act(obs)
            _, _, _, _, dones, reward = sim.step_with_reward(action)

            if visualizer and step % 10 == 0:
                visualizer.update(sim.pos)

            memory_states.append(obs)
            memory_actions.append(action)
            memory_logprobs.append(logprob)
            memory_rewards.append(reward)
            memory_terminals.append(dones)
            memory_values.append(value.squeeze())
            epoch_reward += reward.mean().item()

            # Accumula metriche
            total_grip_count += sim._last_grip_exceeded.sum().item()
            total_lap_count  += sim._last_lap_crossed.sum().item()
            total_dist_sum   += sim._last_dist.mean().item()
            total_steps_n    += 1

    metrics = {
        'avg_dist':  total_dist_sum / max(total_steps_n, 1),
        'lap_count': int(total_lap_count),
        'grip_pct':  100.0 * total_grip_count / max(total_steps_n * sim.N, 1),
    }

    return (memory_states, memory_actions, memory_logprobs,
            memory_rewards, memory_terminals, memory_values,
            epoch_reward, metrics)


# ---------------------------------------------------------------------------
# 6. TRAINING LOOP
# ---------------------------------------------------------------------------
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
        # --- Curriculum spawn: expand spawn zone over first 500 epochs ---
        if epoch < 500:
            sim.spawn_fraction = 0.2 + 0.8 * (epoch / 500.0)
        else:
            sim.spawn_fraction = 1.0

        # --- Collect experience ---
        (memory_states, memory_actions, memory_logprobs,
         memory_rewards, memory_terminals, memory_values,
         epoch_reward, metrics) = _rollout(sim, model, visualizer, STEPS_PER_EPOCH)

        # --- Adaptive entropy coefficient ---
        cur_std = model.actor_log_std.exp().mean().item()
        if cur_std > 0.5:
            ent_coef = 0.005    # forza convergenza
        elif cur_std < 0.15:
            ent_coef = 0.02     # previeni premature convergence
        else:
            ent_coef = 0.01     # zona buona

        # --- Discounted returns ---
        returns           = []
        discounted_reward = torch.zeros(NUM_INSTANCES, device=DEVICE)
        for r, d in zip(reversed(memory_rewards), reversed(memory_terminals)):
            discounted_reward = r + GAMMA * discounted_reward * (~d).float()
            returns.insert(0, discounted_reward)

        returns      = torch.stack(returns).detach()                    # (T, N)
        old_states   = torch.stack(memory_states).detach().view(-1, 12) # (T*N, 12)
        old_actions  = torch.stack(memory_actions).detach().view(-1, 2) # (T*N, 2)
        old_logprobs = torch.stack(memory_logprobs).detach().view(-1)   # (T*N,)
        old_values   = torch.stack(memory_values).detach().view(-1)     # (T*N,)

        advantages = returns.view(-1) - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- Mini-batch PPO update ---
        n_total    = old_states.shape[0]
        model.train()
        total_loss = 0.0
        n_updates  = 0

        for _ in range(K_EPOCHS):
            perm = torch.randperm(n_total, device=DEVICE)
            for start in range(0, n_total, MINI_BATCH_SIZE):
                idx = perm[start : start + MINI_BATCH_SIZE]

                logprobs, state_values, dist_entropy = model.evaluate(
                    old_states[idx], old_actions[idx]
                )
                ratios = torch.exp(logprobs - old_logprobs[idx])
                mb_adv = advantages[idx]
                surr1  = ratios * mb_adv
                surr2  = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * mb_adv

                loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * nn.MSELoss()(state_values.squeeze(),
                                         returns.view(-1)[idx])
                    - ent_coef * dist_entropy          # adaptive entropy
                ).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                total_loss += loss.item()
                n_updates  += 1

        # --- Logging & checkpoint ---
        if epoch % 5 == 0:
            avg_reward = epoch_reward / STEPS_PER_EPOCH
            print(
                f"Epoca {epoch:4d} | "
                f"Reward: {avg_reward:7.3f} | "
                f"Std: {cur_std:.3f} | "
                f"Loss: {total_loss / max(n_updates, 1):.4f} | "
                f"AvgDist: {metrics['avg_dist']:.2f} | "
                f"LapCross: {metrics['lap_count']} | "
                f"GripPct: {metrics['grip_pct']:.1f}% | "
                f"EntCoef: {ent_coef:.4f} | "
                f"Spawn: {sim.spawn_fraction:.0%}"
            )
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'reward': avg_reward,
            }, SAVE_PATH)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mp.freeze_support()
    try:
        train()
    except KeyboardInterrupt:
        print("\nAddestramento interrotto. Modello salvato.")
