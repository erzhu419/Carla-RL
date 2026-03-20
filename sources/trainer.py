import os
import sys
import settings
from sources import ARTDQNAgent, TensorBoard, STOP, ACTIONS, ACTIONS_NAMES
from collections import deque
import time
import random
import numpy as np
import pickle
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from threading import Thread


# Trainer class
class ARTDQNTrainer(ARTDQNAgent):
    def __init__(self, model_path):
        self.model_path = model_path
        self.show_conv_cam = False
        self.device = torch.device('cpu')   # set properly inside run()
        self._conv_output = None
        self.model = self.create_model(prediction=False)
        self._current_lr = settings.OPTIMIZER_LEARNING_RATE
        self._current_decay = settings.OPTIMIZER_DECAY
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self._current_lr,
            weight_decay=self._current_decay,
        )

    def init2(self, stop, logdir, trainer_stats, episode, epsilon, discount, update_target_every,
              last_target_update, min_reward, agent_show_preview, save_checkpoint_every,
              seconds_per_episode, duration, optimizer_shared, models, car_npcs):

        # Target network (copy of main)
        self.target_model = self.create_model(prediction=True)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.replay_memory = deque(maxlen=settings.REPLAY_MEMORY_SIZE)

        self.logdir = logdir if logdir else "logs/{}-{}".format(settings.MODEL_NAME, int(time.time()))
        self.tensorboard = TensorBoard(log_dir=self.logdir)
        self.tensorboard.step = episode.value

        self.last_target_update = last_target_update
        self.last_log_episode = 0
        self.tps = 0
        self.last_checkpoint = 0
        self.save_model = False

        # Shared objects
        self.stop = stop
        self.trainer_stats = trainer_stats
        self.episode = episode
        self.epsilon = epsilon
        self.discount = discount
        self.update_target_every = update_target_every
        self.min_reward = min_reward
        self.agent_show_preview = agent_show_preview
        self.save_checkpoint_every = save_checkpoint_every
        self.seconds_per_episode = seconds_per_episode
        self.duration = duration
        self.optimizer_shared = optimizer_shared
        self.models = models
        self.car_npcs = car_npcs

        # Sync shared optimizer stats
        self.optimizer_shared[0] = self._current_lr
        self.optimizer_shared[1] = self._current_decay

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # ---- helper: numpy images (B, H, W, C) → float tensor (B, C, H, W) on device ----
    def _img_to_tensor(self, img_array):
        arr = np.array(img_array, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        return torch.from_numpy(arr).to(self.device)

    def _kmh_to_tensor(self, kmh_array):
        arr = (np.array(kmh_array, dtype=np.float32) - 50.0) / 50.0
        return torch.from_numpy(arr).to(self.device)

    def train(self):
        if len(self.replay_memory) < settings.MIN_REPLAY_MEMORY_SIZE:
            return False

        minibatch = random.sample(self.replay_memory, settings.MINIBATCH_SIZE)

        # ---- current states ----
        cur_imgs = self._img_to_tensor([t[0][0] for t in minibatch])
        cur_kmh = self._kmh_to_tensor([[t[0][1]] for t in minibatch]) if 'kmh' in settings.AGENT_ADDITIONAL_DATA else None

        # ---- next states ----
        nxt_imgs = self._img_to_tensor([t[3][0] for t in minibatch])
        nxt_kmh = self._kmh_to_tensor([[t[3][1]] for t in minibatch]) if 'kmh' in settings.AGENT_ADDITIONAL_DATA else None

        self.model.eval()
        with torch.no_grad():
            current_qs_list = self.model(cur_imgs, cur_kmh).cpu().numpy()
            future_qs_list  = self.target_model(nxt_imgs, nxt_kmh).cpu().numpy()

        # Build targets
        X_img, X_kmh, y = [], [], []
        for idx, (cur_state, action, reward, _, done) in enumerate(minibatch):
            new_q = reward if done else (reward + self.discount.value * np.max(future_qs_list[idx]))
            current_qs = current_qs_list[idx].copy()
            current_qs[action] = new_q
            X_img.append(cur_state[0])
            if 'kmh' in settings.AGENT_ADDITIONAL_DATA:
                X_kmh.append([cur_state[1]])
            y.append(current_qs)

        # Logging flag
        log_this_step = False
        if self.tensorboard.step > self.last_log_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        X_img_t = self._img_to_tensor(X_img)
        X_kmh_t = self._kmh_to_tensor(X_kmh) if X_kmh else None
        y_t = torch.FloatTensor(np.array(y)).to(self.device)

        # Training step
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(X_img_t, X_kmh_t)
        loss = F.mse_loss(pred, y_t)
        loss.backward()
        self.optimizer.step()

        # Tensorboard callback equivalent (log per episode, not per step)
        if log_this_step:
            self.tensorboard.on_epoch_end(self.tensorboard.step, {'loss': loss.item()})

        # Dynamic lr/decay update from optimizer_shared flags
        if self.optimizer_shared[2] == 1:
            self.optimizer_shared[2] = 0
            new_lr = self.optimizer_shared[3]
            for pg in self.optimizer.param_groups:
                pg['lr'] = new_lr
            self._current_lr = new_lr

        if self.optimizer_shared[4] == 1:
            self.optimizer_shared[4] = 0
            new_decay = self.optimizer_shared[5]
            for pg in self.optimizer.param_groups:
                pg['weight_decay'] = new_decay
            self._current_decay = new_decay

        self.optimizer_shared[0] = self._current_lr
        self.optimizer_shared[1] = self._current_decay

        # Update target network
        if self.tensorboard.step >= self.last_target_update + self.update_target_every.value:
            self.target_model.load_state_dict(self.model.state_dict())
            self.last_target_update += self.update_target_every.value
            print(f"[Trainer] Target network updated at step {self.tensorboard.step}")

        print(f"[Trainer] Step {self.tensorboard.step}: Loss={loss.item():.4f}")

        self.tensorboard.step += 1
        return True

    def get_lr_decay(self):
        return self._current_lr, self._current_decay

    def serialize_weights(self):
        # Move all tensors to CPU before pickling (CUDA tensors can't cross processes)
        cpu_sd = {k: v.cpu() for k, v in self.model.state_dict().items()}
        return pickle.dumps(cpu_sd)

    def init_serialized_weights(self, weights, weights_iteration):
        self.weights = weights
        self.weights.raw = self.serialize_weights()
        self.weights_iteration = weights_iteration

    def _model_config(self):
        in_channels = 1 if settings.AGENT_IMG_TYPE == 1 else 3  # 1=grayscaled
        return {
            'base': settings.MODEL_BASE,
            'head': settings.MODEL_HEAD,
            'input_shape': (settings.IMG_HEIGHT, settings.IMG_WIDTH, in_channels),
            'outputs': len(settings.ACTIONS),
            'model_settings': settings.MODEL_SETTINGS,
        }

    def save_to_path(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self._model_config(),
        }, path)

    def train_in_loop(self):
        self.tps_counter = deque(maxlen=20)
        while True:
            step_start = time.time()
            if self.stop.value == STOP.stopping:
                return
            if self.stop.value in [STOP.carla_simulator_error, STOP.restarting_carla_simulator]:
                self.trainer_stats[0] = TRAINER_STATE.paused
                time.sleep(1)
                continue

            if not self.train():
                self.trainer_stats[0] = TRAINER_STATE.waiting
                if self.stop.value in [STOP.at_checkpoint, STOP.now]:
                    self.stop.value = STOP.stopping
                time.sleep(0.01)
                continue

            self.trainer_stats[0] = TRAINER_STATE.training
            self.weights.raw = self.serialize_weights()
            with self.weights_iteration.get_lock():
                self.weights_iteration.value += 1

            frame_time = time.time() - step_start
            self.tps_counter.append(frame_time)
            self.trainer_stats[1] = len(self.tps_counter) / sum(self.tps_counter)

            if self.save_model:
                self.save_to_path(self.save_model)
                self.save_model = False

            # Checkpointing
            checkpoint_number = self.episode.value // self.save_checkpoint_every.value
            if checkpoint_number > self.last_checkpoint or self.stop.value == STOP.now:
                ckpt_path = f'checkpoint/{settings.MODEL_NAME}_{self.episode.value}.model'
                self.models.append(ckpt_path)
                hparams = {
                    'duration': self.duration.value,
                    'episode': self.episode.value,
                    'epsilon': list(self.epsilon),
                    'discount': self.discount.value,
                    'update_target_every': self.update_target_every.value,
                    'last_target_update': self.last_target_update,
                    'min_reward': self.min_reward.value,
                    'agent_show_preview': [list(p) for p in self.agent_show_preview],
                    'save_checkpoint_every': self.save_checkpoint_every.value,
                    'seconds_per_episode': self.seconds_per_episode.value,
                    'model_path': ckpt_path,
                    'logdir': self.logdir,
                    'weights_iteration': self.weights_iteration.value,
                    'car_npcs': list(self.car_npcs),
                    'models': list(set(self.models)),
                }
                self.save_to_path(ckpt_path)
                with open('checkpoint/hparams_new.json', 'w', encoding='utf-8') as f:
                    json.dump(hparams, f)
                try:
                    os.remove('checkpoint/hparams.json')
                except: pass
                try:
                    os.rename('checkpoint/hparams_new.json', 'checkpoint/hparams.json')
                    self.last_checkpoint = checkpoint_number
                except Exception as e:
                    print(str(e))

            if self.stop.value in [STOP.at_checkpoint, STOP.now]:
                self.stop.value = STOP.stopping


# Trainer states
@dataclass
class TRAINER_STATE:
    starting = 0
    waiting = 1
    training = 2
    finished = 3
    paused = 4

TRAINER_STATE_MESSAGE = {
    0: 'STARTING', 1: 'WAITING', 2: 'TRAINING', 3: 'FINISHED', 4: 'PAUSED',
}


def check_weights_size(model_path, weights_size):
    trainer = ARTDQNTrainer(model_path)
    weights_size.value = len(trainer.serialize_weights())


def run(model_path, logdir, stop, weights, weights_iteration, episode, epsilon, discount,
        update_target_every, last_target_update, min_reward, agent_show_preview,
        save_checkpoint_every, seconds_per_episode, duration, transitions, tensorboard_stats,
        trainer_stats, episode_stats, optimizer_shared, models, car_npcs, carla_settings_stats, carla_fps):

    # Configure GPU
    if settings.TRAINER_GPU is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{settings.TRAINER_GPU}')
    elif torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if device.type == 'cuda':
        torch.cuda.set_device(device)

    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    if device.type == 'cuda':
        torch.cuda.manual_seed(1)

    trainer = ARTDQNTrainer(model_path)
    trainer.device = device
    trainer.model = trainer.model.to(device)
    trainer.optimizer = torch.optim.Adam(
        trainer.model.parameters(),
        lr=settings.OPTIMIZER_LEARNING_RATE,
        weight_decay=settings.OPTIMIZER_DECAY,
    )
    trainer._current_lr = settings.OPTIMIZER_LEARNING_RATE
    trainer._current_decay = settings.OPTIMIZER_DECAY

    trainer.init2(stop, logdir, trainer_stats, episode, epsilon, discount, update_target_every,
                  last_target_update, min_reward, agent_show_preview, save_checkpoint_every,
                  seconds_per_episode, duration, optimizer_shared, models, car_npcs)
    trainer.init_serialized_weights(weights, weights_iteration)

    trainer_stats[0] = TRAINER_STATE.waiting

    trainer_thread = Thread(target=trainer.train_in_loop, daemon=True)
    trainer_thread.start()

    raw_rewards = deque(maxlen=settings.AGENTS * 10)
    weighted_rewards = deque(maxlen=settings.AGENTS * 10)
    episode_times = deque(maxlen=settings.AGENTS * 10)
    frame_times = deque(maxlen=settings.AGENTS * 2)
    configured_actions = [getattr(ACTIONS, a) for a in settings.ACTIONS]

    while stop.value != 3:
        if episode.value > trainer.tensorboard.step:
            trainer.tensorboard.step = episode.value

        for _ in range(transitions.qsize()):
            try: trainer.update_replay_memory(transitions.get(True, 0.1))
            except: break

        while not tensorboard_stats.empty():
            agent_episode, reward, agent_epsilon, episode_time, frame_time, weighted_reward, *avg_predicted_qs = \
                tensorboard_stats.get_nowait()

            raw_rewards.append(reward)
            weighted_rewards.append(weighted_reward)
            episode_times.append(episode_time)
            frame_times.append(frame_time)

            episode_stats[0] = min(raw_rewards)
            episode_stats[1] = sum(raw_rewards) / len(raw_rewards)
            episode_stats[2] = max(raw_rewards)
            episode_stats[3] = min(episode_times)
            episode_stats[4] = sum(episode_times) / len(episode_times)
            episode_stats[5] = max(episode_times)
            episode_stats[6] = sum(frame_times) / len(frame_times)
            episode_stats[7] = min(weighted_rewards)
            episode_stats[8] = sum(weighted_rewards) / len(weighted_rewards)
            episode_stats[9] = max(weighted_rewards)

            tensorboard_q_stats = {}
            for action, (avg_q, std_q, use_q) in enumerate(zip(avg_predicted_qs[0::3], avg_predicted_qs[1::3], avg_predicted_qs[2::3])):
                if avg_q != -10**6:
                    episode_stats[action * 3 + 10] = avg_q
                    key = f'q_all_actions_avg' if action == 0 else f'q_action_{action-1}_{ACTIONS_NAMES[configured_actions[action-1]]}_avg'
                    tensorboard_q_stats[key] = avg_q
                if std_q != -10**6:
                    episode_stats[action * 3 + 11] = std_q
                    key = f'q_all_actions_std' if action == 0 else f'q_action_{action-1}_{ACTIONS_NAMES[configured_actions[action-1]]}_std'
                    tensorboard_q_stats[key] = std_q
                if use_q != -10**6:
                    episode_stats[action * 3 + 12] = use_q
                    if action > 0:
                        tensorboard_q_stats[f'q_action_{action-1}_{ACTIONS_NAMES[configured_actions[action-1]]}_usage_pct'] = use_q

            carla_stats = {}
            for process_no in range(settings.CARLA_HOSTS_NO):
                for idx, stat in enumerate(['carla_{}_car_npcs', 'carla_{}_weather_sun_azimuth',
                                            'carla_{}_weather_sun_altitude', 'carla_{}_weather_clouds_pct',
                                            'carla_{}_weather_wind_pct', 'carla_{}_weather_rain_pct']):
                    if carla_settings_stats[process_no][idx] != -1:
                        carla_stats[stat.format(process_no + 1)] = carla_settings_stats[process_no][idx]
                carla_stats[f'carla_{process_no + 1}_fps'] = carla_fps[process_no].value

            trainer.tensorboard.update_stats(
                step=agent_episode,
                reward_raw_avg=episode_stats[1], reward_raw_min=episode_stats[0], reward_raw_max=episode_stats[2],
                reward_weighted_avg=episode_stats[8], reward_weighted_min=episode_stats[7], reward_weighted_max=episode_stats[9],
                epsilon=agent_epsilon, episode_time_avg=episode_stats[4], episode_time_min=episode_stats[3],
                episode_time_max=episode_stats[5], agent_fps_avg=episode_stats[6],
                optimizer_lr=optimizer_shared[0], optimizer_decay=optimizer_shared[1],
                **tensorboard_q_stats, **carla_stats
            )

            if episode_stats[7] >= min_reward.value:
                trainer.save_model = (
                    f'models/{settings.MODEL_NAME}__{episode_stats[2]:_>7.2f}max_'
                    f'{episode_stats[1]:_>7.2f}avg_{episode_stats[0]:_>7.2f}min__{int(time.time())}.model'
                )

        time.sleep(0.01)

    trainer_thread.join()
    trainer_stats[0] = TRAINER_STATE.finished
