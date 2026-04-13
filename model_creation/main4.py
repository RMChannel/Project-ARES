import torch
import torch.optim as optim
import numpy as np
import os
import multiprocessing as mp
import time

from model import PilotNet
from simulator import GPUSimulator
from visualizer import AsyncVisualizer
import read_ai as fast_lane_api

# --- CONFIGURAZIONE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_INSTANCES = 1024
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 5
STEPS_PER_EPOCH = 2048
EPOCHS = 10000
SAVE_PATH = "pilot_model_v4.pth"
OUT_OF_BOUNDS_DIST = 4.0
VISUALIZE = True
NUM_VIS_AGENTS = 20

print(f"Project ARES - Main 4 (CarSim Engine)")
print(f"Dispositivo: {DEVICE} - Istanze: {NUM_INSTANCES}")

def load_real_track(file_path="../files_ai/fast_lane.ai"):
    print(f"Caricamento tracciato da {file_path}...")
    lista_coordinate = fast_lane_api.get_data(file_path)
    points = [[c.x, c.z] for c in lista_coordinate]
    track_tensor = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    print(f"Tracciato caricato: {len(track_tensor)} waypoints.")
    return track_tensor

def train():
    track = load_real_track()
    sim = GPUSimulator(NUM_INSTANCES, track, DEVICE, out_of_bounds_dist=OUT_OF_BOUNDS_DIST)
    model = PilotNet(input_dim=11).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    visualizer = None
    if VISUALIZE:
        visualizer = AsyncVisualizer(track, num_agents=NUM_VIS_AGENTS, num_instances=NUM_INSTANCES)

    start_epoch = 0
    if os.path.exists(SAVE_PATH):
        try:
            checkpoint = torch.load(SAVE_PATH, weights_only=False, map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                print(f"Checkpoint caricato: epoca {start_epoch}")
            else:
                model.load_state_dict(checkpoint)
                print("Modello legacy caricato.")
        except Exception as e:
            print(f"Errore caricamento checkpoint: {e}")

    for epoch in range(start_epoch, EPOCHS):
        noise_std = max(0.1, 0.4 * (0.999 ** epoch))
        
        memory_states = []
        memory_actions = []
        memory_logprobs = []
        memory_rewards = []
        memory_is_terminals = []
        memory_values = []
        
        epoch_reward = 0.0
        
        model.eval()
        with torch.no_grad():
            for step in range(STEPS_PER_EPOCH):
                obs, dist, angle, nearest_idx = sim.get_observation()
                action, logprob, value = model.act(obs, noise_std)
                
                next_obs, dist_after, angle_after, next_idx, dones = sim.step(action)
                
                if visualizer and step % 10 == 0:
                    visualizer.update(sim.pos)

                # Reward logic
                raw_progress = next_idx.float() - sim.prev_nearest_idx.float()
                half_track = sim.n_track / 2.0
                progress = (raw_progress + half_track) % sim.n_track - half_track
                progress = progress * (~dones).float()
                
                sim.prev_nearest_idx[~dones] = next_idx[~dones]
                
                # tangential factor (alignment with track)
                tangential_factor = torch.cos(angle_after)
                
                reward = (
                    (sim.speed / 50.0) * tangential_factor
                    + (progress * 0.3)
                    - (dist_after / OUT_OF_BOUNDS_DIST) * 0.5
                    - (torch.abs(angle_after) / np.pi) * 0.5
                )
                reward[dones] -= 10.0
                
                memory_states.append(obs)
                memory_actions.append(action)
                memory_logprobs.append(logprob)
                memory_rewards.append(reward)
                memory_is_terminals.append(dones)
                memory_values.append(value.squeeze())
                
                epoch_reward += reward.mean().item()

        # Update PPO
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

        model.train()
        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = model.evaluate(old_states, old_actions, noise_std)
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * torch.nn.MSELoss()(state_values.squeeze(), returns.view(-1)) - 0.01 * dist_entropy
            
            optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

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
    mp.freeze_support()
    try:
        train()
    except KeyboardInterrupt:
        print("\nInterrotto.")
