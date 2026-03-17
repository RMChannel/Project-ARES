import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from torch.distributions import Normal

# --- CONFIGURAZIONE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_INSTANCES = 2048
LR = 1e-4
GAMMA = 0.99
STEPS_PER_EPOCH = 200
EPOCHS = 5000
SAVE_PATH = "pilot_model.pth"
OUT_OF_BOUNDS_DIST = 50.0  # metri oltre i quali si resetta l'istanza

print(f"Dispositivo rilevato: {DEVICE} - Istanze parallele: {NUM_INSTANCES}")

# --- 1. GENERATORE CIRCUITO ---
def generate_track(num_points=1000, radius=200):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    variation = np.sin(angles * 3) * 30 + np.cos(angles * 5) * 15
    x = (radius + variation) * np.cos(angles)
    y = (radius + variation) * np.sin(angles)
    points = np.stack([x, y], axis=1)
    return torch.tensor(points, dtype=torch.float32, device=DEVICE)

# --- 2. IL CERVELLO (Rete Neurale) ---
class PilotNet(nn.Module):
    def __init__(self, input_dim=7):
        super(PilotNet, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Actor: decide Accelerazione e Sterzo
        self.actor = nn.Sequential(
            nn.Linear(128, 2),
            nn.Tanh()
        )
        # Critic: valuta quanto è buona la situazione attuale
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.common(x)
        return self.actor(x), self.critic(x)

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
        """
        Resetta tutte le istanze (mask=None) oppure solo quelle indicate dalla mask booleana.
        FIX #10: accel_g viene azzerata correttamente ad ogni reset.
        """
        if mask is None:
            self.pos = self.track[0].clone().unsqueeze(0).repeat(self.N, 1)
            self.heading = self.track_headings[0].clone().repeat(self.N)
            self.vel = torch.zeros((self.N, 2), device=DEVICE)
            self.speed = torch.zeros(self.N, device=DEVICE)
            self.accel_g = torch.zeros(self.N, device=DEVICE)  # FIX #10
            self.prev_nearest_idx = torch.zeros(self.N, dtype=torch.long, device=DEVICE)
        else:
            self.pos[mask] = self.track[0]
            self.heading[mask] = self.track_headings[0]
            self.vel[mask] = 0.0
            self.speed[mask] = 0.0
            self.accel_g[mask] = 0.0  # FIX #10
            self.prev_nearest_idx[mask] = 0

    def get_observation(self):
        dists = torch.cdist(self.pos, self.track)
        dist_to_center, nearest_idx = torch.min(dists, dim=1)

        ideal_h = self.track_headings[nearest_idx]
        angle_to_center = (ideal_h - self.heading + np.pi) % (2 * np.pi) - np.pi

        side_x = -torch.sin(self.heading)
        side_y = torch.cos(self.heading)
        lat_vel = self.vel[:, 0] * side_x + self.vel[:, 1] * side_y

        future_idx = (nearest_idx + 50) % self.n_track
        future_h = self.track_headings[future_idx]
        next_curve_dir = torch.sign((future_h - ideal_h + np.pi) % (2 * np.pi) - np.pi)

        # FIX #1: angle_to_center (era scritto angle_center)
        # FIX #4: next_curve_dir ora inclusa nell'osservazione
        obs = torch.stack([
            self.speed * 0.01,           # normalizzato (velocità tipica ~100 km/h → ~1.0)
            self.accel_g,
            self.heading * 0.1,
            lat_vel * 0.1,
            dist_to_center / OUT_OF_BOUNDS_DIST,  # normalizzato 0..1
            angle_to_center / np.pi,     # normalizzato -1..1
            next_curve_dir               # FIX #4: era calcolata ma mai inserita
        ], dim=1)

        return obs, dist_to_center, angle_to_center, nearest_idx

    def step(self, actions):
        """
        FIX #2: step() ora restituisce l'osservazione aggiornata,
        in modo che il reward usi lo stato DOPO l'azione.
        """
        throttle = actions[:, 0]
        steer = actions[:, 1]

        self.heading += steer * 4.0 * self.dt
        acc_val = throttle * 15.0

        self.vel[:, 0] += torch.cos(self.heading) * acc_val * self.dt
        self.vel[:, 1] += torch.sin(self.heading) * acc_val * self.dt
        self.vel *= 0.97  # attrito

        self.pos += self.vel * self.dt

        new_speed = torch.norm(self.vel, dim=1) * 3.6
        self.accel_g = (new_speed - self.speed) / (self.dt * 9.81)
        self.speed = new_speed

        # FIX #11: reset selettivo per istanze fuori pista
        obs, dist_to_center, angle_to_center, nearest_idx = self.get_observation()
        out_of_bounds = dist_to_center > OUT_OF_BOUNDS_DIST
        if out_of_bounds.any():
            self.reset(mask=out_of_bounds)
            # Ricalcola obs dopo il reset parziale
            obs, dist_to_center, angle_to_center, nearest_idx = self.get_observation()

        return obs, dist_to_center, angle_to_center, nearest_idx  # FIX #2

# --- 4. CICLO DI ADDESTRAMENTO ---
def train():
    track = generate_track()
    sim = GPUSimulator(NUM_INSTANCES, track)
    model = PilotNet(input_dim=7).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_epoch = 0

    # FIX #5 + FIX #13: caricamento robusto del checkpoint con metadati
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"Checkpoint caricato: ripresa dall'epoca {start_epoch}, "
                  f"reward precedente: {checkpoint.get('reward', 'N/A'):.2f}")
        else:
            # Compatibilità con il vecchio formato (solo state_dict)
            model.load_state_dict(checkpoint)
            print("Modello legacy caricato (solo pesi).")

    for epoch in range(start_epoch, EPOCHS):
        # FIX #8: esplorazione che decade nel tempo
        noise_std = max(0.05, 0.3 * (1.0 - epoch / EPOCHS))

        states, actions, log_probs_list, rewards, values = [], [], [], [], []
        sim.reset()

        epoch_reward = 0.0

        for _ in range(STEPS_PER_EPOCH):
            obs, dist, angle, nearest_idx = sim.get_observation()

            action_means, val = model(obs)

            # FIX #7: distribuzione esplicita per log_probs corretti
            std = torch.full_like(action_means, noise_std)
            dist_policy = Normal(action_means, std)
            action = dist_policy.sample()
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist_policy.log_prob(action).sum(dim=-1)  # FIX #3: ora calcolato e salvato

            # FIX #2: usa l'obs aggiornata dopo step
            next_obs, dist_after, angle_after, next_idx = sim.step(action)

            # FIX #9: reward bilanciato e normalizzato
            # - velocità normalizzata (target ~100 km/h → 1.0)
            # - distanza normalizzata sul raggio fuori-pista
            # - angolo normalizzato su pi greco
            # FIX #12: bonus di progresso sul giro
            progress = (next_idx.float() - sim.prev_nearest_idx.float()) % sim.n_track
            sim.prev_nearest_idx = next_idx.clone()

            reward = (
                sim.speed / 100.0                          # velocità (0..~2)
                - dist_after / OUT_OF_BOUNDS_DIST          # distanza dal centro (0..1)
                - torch.abs(angle_after) / np.pi           # angolo errato (0..1)
                + progress * 0.05                          # FIX #12: progresso sul giro
            )

            states.append(obs)
            actions.append(action)
            log_probs_list.append(log_prob)  # FIX #3: lista usata nell'update
            values.append(val.squeeze())
            rewards.append(reward)
            epoch_reward += reward.mean().item()

        # --- Update Rete ---
        returns = []
        R = torch.zeros(NUM_INSTANCES, device=DEVICE)
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        returns = torch.stack(returns)

        # FIX #6: normalizzazione dei returns per stabilità del gradiente
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns = returns.detach()

        values_t = torch.stack(values)
        states_t = torch.stack(states)
        actions_t = torch.stack(actions)
        log_probs_t = torch.stack(log_probs_list)  # FIX #3

        advantage = returns - values_t.detach()

        # Ricalcola distribuzione sugli stati raccolti
        action_means, current_values = model(states_t)
        std_t = torch.full_like(action_means, noise_std)
        dist_new = Normal(action_means, std_t)
        new_log_probs = dist_new.log_prob(actions_t).sum(dim=-1)

        # FIX #7: vera policy gradient loss
        actor_loss = -(advantage * new_log_probs).mean()
        critic_loss = nn.MSELoss()(current_values.squeeze(), returns)

        # Entropy bonus per mantenere esplorazione
        entropy_bonus = dist_new.entropy().mean()
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping per stabilità
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 10 == 0:
            avg_reward = epoch_reward / STEPS_PER_EPOCH
            print(f"Epoca {epoch:5d} | Reward Medio: {avg_reward:8.3f} | "
                  f"Noise σ: {noise_std:.3f} | Loss: {loss.item():.4f}")

            # FIX #5 + FIX #13: salvataggio con metadati completi
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'reward': avg_reward,
                'noise_std': noise_std,
            }, SAVE_PATH)

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nAddestramento interrotto. Modello salvato.")