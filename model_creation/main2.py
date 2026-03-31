import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import multiprocessing as mp
import queue
from torch.distributions import Normal

# Importiamo il tuo file fast_lane_api (deve essere nella stessa cartella)
import read_ai as fast_lane_api

# --- CONFIGURAZIONE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_INSTANCES = 4096  # Ridotto per stabilità e velocità di training
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 5
STEPS_PER_EPOCH = 512
EPOCHS = 10000
SAVE_PATH = "pilot_model.pth"
OUT_OF_BOUNDS_DIST = 4
VISUALIZE = True
NUM_VIS_AGENTS = 20

print(f"Dispositivo rilevato: {DEVICE} - Istanze parallele: {NUM_INSTANCES}")


# --- 1. GENERATORE CIRCUITO ---
def load_real_track(file_path="fast_lane.ai"):
    print(f"Caricamento tracciato da {file_path}...")
    lista_coordinate = fast_lane_api.get_data(file_path)

    # Estraiamo x e z
    points = [[c.x, c.z] for c in lista_coordinate]
    track_tensor = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    print(f"Tracciato caricato con successo: {len(track_tensor)} waypoints.")

    return track_tensor


# --- 2. IL CERVELLO (PPO Actor-Critic) ---
class PilotNet(nn.Module):
    def __init__(self, input_dim=11):
        super(PilotNet, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 2),
            nn.Tanh()
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.common(x)
        return self.actor(x), self.critic(x)

    def act(self, x, noise_std):
        action_mean, value = self.forward(x)
        std = torch.full_like(action_mean, noise_std)
        dist = Normal(action_mean, std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), action_logprob.detach(), value.detach()

    def evaluate(self, x, action, noise_std):
        action_mean, value = self.forward(x)
        std = torch.full_like(action_mean, noise_std)
        dist = Normal(action_mean, std)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        return action_logprobs, value, dist_entropy


# --- 3. AMBIENTE SIMULATO SU GPU ---
class GPUSimulator:
    def __init__(self, num_instances, track):
        self.N = num_instances
        self.track = track
        self.n_track = len(track)
        self.dt = 1 / 30.0

        next_p = torch.roll(track, -1, dims=0)
        diff = next_p - track
        self.track_headings = torch.atan2(diff[:, 1], diff[:, 0])

        self.reset()

    def reset(self, mask=None):
        if mask is None:
            self.pos = self.track[0].clone().unsqueeze(0).repeat(self.N, 1)
            self.heading = self.track_headings[0].clone().repeat(self.N)
            self.vel = torch.zeros((self.N, 2), device=DEVICE)
            self.speed = torch.zeros(self.N, device=DEVICE)
            self.accel_g = torch.zeros(self.N, device=DEVICE)
            self.prev_nearest_idx = torch.zeros(self.N, dtype=torch.long, device=DEVICE)
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

        # Look-ahead a diverse distanze
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
            dist_to_center / OUT_OF_BOUNDS_DIST,
            angle_to_center / np.pi,
            torch.sin(self.heading),
            torch.cos(self.heading),
            *curvatures
        ], dim=1)

        return obs, dist_to_center, angle_to_center, nearest_idx

    def step(self, actions):
        throttle = actions[:, 0]
        steer = actions[:, 1]

        # Dinamica leggermente più realistica
        self.heading += steer * 5.0 * self.dt
        acc_val = throttle * 18.0

        self.vel[:, 0] += torch.cos(self.heading) * acc_val * self.dt
        self.vel[:, 1] += torch.sin(self.heading) * acc_val * self.dt
        
        # Attrito proporzionale alla velocità
        drag = 0.98 - (self.speed * 0.0001)
        self.vel *= drag.unsqueeze(1)

        self.pos += self.vel * self.dt

        new_speed = torch.norm(self.vel, dim=1) * 3.6
        self.accel_g = (new_speed - self.speed) / (self.dt * 9.81)
        self.speed = new_speed

        obs, dist_to_center, angle_to_center, nearest_idx = self.get_observation()

        out_of_bounds = dist_to_center > OUT_OF_BOUNDS_DIST
        # Penalità se troppo lento o fermo per troppo tempo? Non per ora.
        
        dones = out_of_bounds.clone()

        if out_of_bounds.any():
            self.reset(mask=out_of_bounds)
            # Ricalcola obs dopo il reset per le istanze morte
            obs_reset, _, _, _ = self.get_observation()
            obs[out_of_bounds] = obs_reset[out_of_bounds]

        return obs, dist_to_center, angle_to_center, nearest_idx, dones


# --- 4. VISUALIZZATORE (MULTIPROCESSING) ---
def render_worker(track_points, pos_queue, num_agents):
    """Gira su un processo separato per non bloccare la GPU"""
    import turtle

    screen = turtle.Screen()
    screen.title("Project ARES - Async Training Visualizer")
    screen.bgcolor("black")
    screen.tracer(0, 0)

    min_x, min_z = track_points.min(axis=0)
    max_x, max_z = track_points.max(axis=0)

    pad_x = (max_x - min_x) * 0.1
    pad_z = (max_z - min_z) * 0.1
    screen.setworldcoordinates(min_x - pad_x, min_z - pad_z, max_x + pad_x, max_z + pad_z)

    track_pen = turtle.Turtle()
    track_pen.speed("fastest")
    track_pen.color("cyan")
    track_pen.pensize(2)
    track_pen.hideturtle()
    track_pen.penup()
    track_pen.goto(track_points[0][0], track_points[0][1])
    track_pen.pendown()
    for p in track_points[1:]:
        track_pen.goto(p[0], p[1])
    track_pen.goto(track_points[0][0], track_points[0][1])

    agents = []
    colors = ["red", "green", "blue", "yellow", "magenta", "white"]
    for i in range(num_agents):
        t = turtle.Turtle()
        t.shape("circle")
        t.shapesize(0.3)
        t.color(colors[i % len(colors)])
        t.penup()
        agents.append(t)

    screen.update()

    while True:
        try:
            pos_np = None
            # Svuota la coda per prendere solo l'ultimo frame
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
        self.pos_queue = mp.Queue(maxsize=3)

        track_points = track_tensor.cpu().numpy()

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


# --- 5. CICLO DI ADDESTRAMENTO ---
def train():
    track = load_real_track("../files_ai/fast_lane.ai")

    sim = GPUSimulator(NUM_INSTANCES, track)
    model = PilotNet(input_dim=11).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    visualizer = None
    if VISUALIZE:
        visualizer = AsyncVisualizer(track, num_agents=NUM_VIS_AGENTS)

    start_epoch = 0

    if os.path.exists(SAVE_PATH):
        try:
            checkpoint = torch.load(SAVE_PATH, weights_only=False, map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                print(f"Checkpoint caricato: ripresa dall'epoca {start_epoch}")
            else:
                model.load_state_dict(checkpoint)
                print("Modello legacy caricato.")
        except Exception as e:
            print(f"Errore caricamento checkpoint: {e}. Inizio da zero.")

    # Reset solo all'inizio assoluto
    sim.reset()

    for epoch in range(start_epoch, EPOCHS):
        # Noise adattivo: decresce ma non troppo velocemente
        noise_std = max(0.1, 0.4 * (0.999 ** epoch))

        memory_states = []
        memory_actions = []
        memory_logprobs = []
        memory_rewards = []
        memory_is_terminals = []
        memory_values = []

        epoch_reward = 0.0

        # --- 1. RACCOLTA DATI (Rollout) ---
        model.eval()
        with torch.no_grad():
            for step in range(STEPS_PER_EPOCH):
                obs, dist, angle, nearest_idx = sim.get_observation()
                
                action, logprob, value = model.act(obs, noise_std)
                
                next_obs, dist_after, angle_after, next_idx, dones = sim.step(action)

                if visualizer and step % 10 == 0:
                    visualizer.update(sim.pos)

                # Reward più bilanciato
                raw_progress = next_idx.float() - sim.prev_nearest_idx.float()
                half_track = sim.n_track / 2.0
                progress = (raw_progress + half_track) % sim.n_track - half_track
                progress = progress * (~dones).float()
                
                # FIX: Aggiorna prev_nearest_idx solo per chi non ha resettato
                sim.prev_nearest_idx[~dones] = next_idx[~dones]

                # Velocità proiettata sulla direzione giusta (tangential speed)
                # Se l'angolo è > 90°, cos(angle) è negativo, punendo la marcia contromano
                tangential_factor = torch.cos(angle_after)
                
                reward = (
                    (sim.speed / 150.0) * tangential_factor  # Premia velocità solo se nella direzione giusta
                    + (progress * 0.3)                       # Aumentato incentivo al progresso
                    - (dist_after / OUT_OF_BOUNDS_DIST) * 0.5 # Penale fuori traiettoria
                    - (torch.abs(angle_after) / np.pi) * 0.5  # Aumentata penale orientamento errato
                )
                reward[dones] -= 10.0 # Penale per crash aumentata

                memory_states.append(obs)
                memory_actions.append(action)
                memory_logprobs.append(logprob)
                memory_rewards.append(reward)
                memory_is_terminals.append(dones)
                memory_values.append(value.squeeze())
                
                epoch_reward += reward.mean().item()

        # --- 2. CALCOLO VANTAGGI E RITORNI ---
        returns = []
        discounted_reward = torch.zeros(NUM_INSTANCES, device=DEVICE)
        for r, d in zip(reversed(memory_rewards), reversed(memory_is_terminals)):
            discounted_reward = r + (GAMMA * discounted_reward * (~d).float())
            returns.insert(0, discounted_reward)

        returns = torch.stack(returns).detach()
        old_states = torch.stack(memory_states).detach().view(-1, 11)
        old_actions = torch.stack(memory_actions).detach().view(-1, 2)
        old_logprobs = torch.stack(memory_logprobs).detach().view(-1)
        old_values = torch.stack(memory_values).detach().view(-1)
        
        advantages = returns.view(-1) - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 3. OTTIMIZZAZIONE PPO ---
        model.train()
        for _ in range(K_EPOCHS):
            # Valutazione nuove azioni
            logprobs, state_values, dist_entropy = model.evaluate(old_states, old_actions, noise_std)
            
            # Rapporto tra policy (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Surrogate Loss (PPO Clip)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values.squeeze(), returns.view(-1)) - 0.01 * dist_entropy

            optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        # Logging e salvataggio
        if epoch % 5 == 0:
            avg_reward = epoch_reward / STEPS_PER_EPOCH
            print(f"Epoca {epoch:4d} | Reward: {avg_reward:7.2f} | Noise: {noise_std:.3f} | Loss: {loss.mean().item():.4f}")

            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'reward': avg_reward,
            }, SAVE_PATH)


if __name__ == "__main__":
    # Su alcuni OS è necessario per il multiprocessing
    mp.freeze_support()
    try:
        train()
    except KeyboardInterrupt:
        print("\nAddestramento interrotto. Modello salvato.")