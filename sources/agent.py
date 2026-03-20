import os
import sys
import settings
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from sources import CarlaEnv, STOP, models, ACTIONS_NAMES
from collections import deque
from threading import Thread
from dataclasses import dataclass
import cv2


def _get_device(gpu_setting, agent_id=None):
    if not torch.cuda.is_available():
        return torch.device('cpu')
    if gpu_setting is None:
        return torch.device('cuda')
    if isinstance(gpu_setting, int):
        return torch.device(f'cuda:{gpu_setting}')
    if isinstance(gpu_setting, list) and agent_id is not None:
        return torch.device(f'cuda:{gpu_setting[agent_id]}')
    return torch.device('cuda')


# Agent class
class ARTDQNAgent:
    def __init__(self, model_path=False, id=None):

        self.model_path = model_path
        self.show_conv_cam = (id + 1) in settings.CONV_CAM_AGENTS if id is not None else False
        self.device = torch.device('cpu')  # set properly in run() before use
        self._conv_output = None  # filled by forward hook when convcam enabled

        # Main model
        self.model = self.create_model(prediction=True)

        self.weights_iteration = 0
        self.terminate = False

    # Create or load model
    def create_model(self, prediction=False):
        in_channels = 1 if settings.AGENT_IMG_TYPE == AGENT_IMAGE_TYPE.grayscaled else 3
        input_shape = (settings.IMG_HEIGHT, settings.IMG_WIDTH, in_channels)

        if self.model_path:
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            # Reconstruct architecture from saved config
            cfg = checkpoint.get('model_config', {})
            base_name = cfg.get('base', settings.MODEL_BASE)
            head_name = cfg.get('head', settings.MODEL_HEAD)
            outputs = cfg.get('outputs', len(settings.ACTIONS))
            ms = cfg.get('model_settings', settings.MODEL_SETTINGS)
            saved_shape = cfg.get('input_shape', input_shape)

            base = getattr(models, 'model_base_' + base_name)(saved_shape)
            model = getattr(models, 'model_head_' + head_name)(base, saved_shape, outputs=outputs, model_settings=ms)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            base = getattr(models, 'model_base_' + settings.MODEL_BASE)(input_shape)
            model = getattr(models, 'model_head_' + settings.MODEL_HEAD)(
                base, input_shape, outputs=len(settings.ACTIONS), model_settings=settings.MODEL_SETTINGS)

        self._extract_model_info(model)

        if self.show_conv_cam:
            self._register_conv_hook(model)

        return model

    def _register_conv_hook(self, model):
        # Find the last Conv2d in the base and register a forward hook
        last_conv = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            def _hook(module, inp, out):
                self._conv_output = out.detach().cpu().numpy()
            last_conv.register_forward_hook(_hook)

    def _extract_model_info(self, model):
        """Build model name string by walking PyTorch module tree."""
        model_architecture = []
        cnn_kernels = []
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                cnn_kernels.append(str(m.out_channels))
                model_architecture.append(f'C{m.out_channels}')
            elif isinstance(m, nn.Flatten):
                model_architecture.append('F')
            elif isinstance(m, nn.Linear):
                model_architecture.append(f'D{m.out_features}')

        settings.MODEL_NAME = settings.MODEL_NAME.replace('#MODEL_ARCHITECTURE#', '-'.join(model_architecture))
        settings.MODEL_NAME = settings.MODEL_NAME.replace('#CNN_KERNELS#', '-'.join(cnn_kernels))

    def compile_model(self, model, lr, decay):
        """No-op for agent (optimizer lives in trainer). Kept for interface compatibility."""
        pass

    def decode_weights(self, weights):
        return pickle.loads(weights.raw)

    def update_weights(self):
        state_dict = self.decode_weights(self.weights)
        self.model.load_state_dict(state_dict)

    def update_weights_in_loop(self):
        if settings.UPDATE_WEIGHTS_EVERY <= 0:
            return
        while True:
            if self.terminate:
                return
            if self.trainer_weights_iteration.value >= self.weights_iteration + settings.UPDATE_WEIGHTS_EVERY:
                self.weights_iteration = self.trainer_weights_iteration.value + settings.UPDATE_WEIGHTS_EVERY
                self.update_weights()
            else:
                time.sleep(0.001)

    def get_qs(self, state):
        # state[0]: numpy (H, W, C);  state[1]: kmh scalar
        img = np.array(state[0], dtype=np.float32).transpose(2, 0, 1) / 255.0  # (C, H, W)
        img_t = torch.FloatTensor(img).unsqueeze(0).to(self.device)  # (1, C, H, W)

        kmh_t = None
        if 'kmh' in settings.AGENT_ADDITIONAL_DATA:
            kmh_val = (float(state[1]) - 50.0) / 50.0
            kmh_t = torch.FloatTensor([[kmh_val]]).to(self.device)

        self.model.eval()
        with torch.no_grad():
            q_values = self.model(img_t, kmh_t).cpu().numpy()  # (1, n_actions)

        if self.show_conv_cam and self._conv_output is not None:
            # _conv_output shape: (1, filters, H, W) → convert to (1, H, W, filters) for Keras compat
            conv_out = self._conv_output.transpose(0, 2, 3, 1)
            return [q_values[0], conv_out]
        return [q_values[0]]

    def prepare_image(self, image, create=False):
        if settings.AGENT_IMG_TYPE == AGENT_IMAGE_TYPE.rgb:
            return image
        elif settings.AGENT_IMG_TYPE == AGENT_IMAGE_TYPE.grayscaled:
            return np.expand_dims(np.dot(image, [0.299, 0.587, 0.114]).astype('uint8'), -1)
        elif settings.AGENT_IMG_TYPE == AGENT_IMAGE_TYPE.stacked:
            if create:
                image = np.dot(image, [0.299, 0.587, 0.114]).astype('uint8')
                self.image = np.stack([image, image, image], axis=-1)
            else:
                self.image = np.roll(self.image, 1, -1)
                self.image[..., 0] = np.dot(image, [0.299, 0.587, 0.114]).astype('uint8')
            return self.image


# Image types
@dataclass
class AGENT_IMAGE_TYPE:
    rgb = 0
    grayscaled = 1
    stacked = 2


# Agent states
@dataclass
class AGENT_STATE:
    starting = 0
    playing = 1
    restarting = 2
    finished = 3
    error = 4
    paused = 5


AGENT_STATE_MESSAGE = {
    0: 'STARTING', 1: 'PLAYING', 2: 'RESTARING',
    3: 'FINISHED', 4: 'ERROR', 5: 'PAUSED',
}


def run(id, carla_instance, stop, pause, episode, epsilon, show_preview, weights, weights_iteration, transitions, tensorboard_stats, agent_stats, carla_frametimes, seconds_per_episode):

    # Set GPU for this agent process
    device = _get_device(settings.AGENT_GPU, id)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    torch.manual_seed(id)

    agent = ARTDQNAgent(id=id)
    agent.device = device
    agent.model = agent.model.to(device)
    agent.weights = weights
    agent.trainer_weights_iteration = weights_iteration
    agent.update_weights()

    # Wait for Carla
    while True:
        if stop.value == STOP.stopping:
            agent_stats[0] = AGENT_STATE.finished
            return
        try:
            env = CarlaEnv(carla_instance, seconds_per_episode)
            break
        except:
            agent_stats[0] = AGENT_STATE.error
            time.sleep(1)

    agent_stats[0] = AGENT_STATE.starting
    env.frametimes = carla_frametimes
    fps_counter = deque(maxlen=60)

    weight_updater = Thread(target=agent.update_weights_in_loop, daemon=True)
    weight_updater.start()

    # Warm-up prediction
    agent.get_qs([np.ones((env.im_height, env.im_width, 1 if settings.AGENT_IMG_TYPE == AGENT_IMAGE_TYPE.grayscaled else 3)), [0]])

    agent_stats[0] = AGENT_STATE.playing

    while stop.value != STOP.stopping:
        if stop.value == STOP.restarting_carla_simulator:
            time.sleep(0.1)
            continue
        if pause.value == 1:
            pause.value = 2
            agent_stats[0] = AGENT_STATE.paused
        if pause.value == 2:
            time.sleep(0.1)
            continue
        if pause.value == 3:
            pause.value = 0
            agent_stats[0] = AGENT_STATE.starting
            try: env.destroy_agents()
            except: pass
            try:
                env = CarlaEnv(carla_instance, seconds_per_episode)
                env.frametimes = carla_frametimes
            except: pass
            time.sleep(1)
            continue

        episode_reward = 0
        step = 1
        predicted_qs = [[] for _ in range(env.action_space_size + 1)]
        predicted_actions = [0 for _ in range(env.action_space_size + 1)]

        try:
            env.destroy_agents()
            current_state = env.reset()
            current_state[0] = agent.prepare_image(current_state[0], create=True)
        except:
            agent_stats[0] = AGENT_STATE.error
            try: env.destroy_agents()
            except: pass
            try:
                env = CarlaEnv(carla_instance, seconds_per_episode)
                env.frametimes = carla_frametimes
            except: pass
            time.sleep(1)
            continue

        if settings.UPDATE_WEIGHTS_EVERY == 0:
            agent.update_weights()

        agent_stats[0] = AGENT_STATE.playing
        env.collision_hist = []
        done = False
        episode_start = episode_end = time.time()
        last_processed_cam_update = 0
        conv_min = conv_max = None

        while True:
            agent_stats[1] = step
            step_start = time.time()

            print(f"[Agent {id}] Episode {episode.value}, Step {step}, Speed {env.kmh:.1f} kmh")

            if settings.AGENT_SYNCED:
                wait_start = time.time()
                while True:
                    if env.last_cam_update > last_processed_cam_update:
                        last_processed_cam_update = env.last_cam_update
                        break
                    if time.time() > wait_start + 1:
                        break
                    time.sleep(0.001)

            if np.random.random() > epsilon[0]:
                qs = agent.get_qs(current_state)
                action = np.argmax(qs[0])
                for i in range(env.action_space_size):
                    predicted_qs[0].append(qs[0][i])
                    predicted_qs[i + 1].append(qs[0][i])
                predicted_actions[0] += 1
                predicted_actions[action + 1] += 1

                if agent.show_conv_cam and len(qs) > 1:
                    conv_min = np.min(qs[1]) if conv_min is None else 0.8 * conv_min + 0.2 * np.min(qs[1])
                    conv_max = np.max(qs[1]) if conv_max is None else 0.8 * conv_max + 0.2 * np.max(qs[1])
                    if conv_max != conv_min:
                        conv_preview = ((qs[1] - conv_min) * 255 / (conv_max - conv_min)).astype(np.uint8)
                        conv_preview = np.moveaxis(conv_preview, 1, 2)
                        conv_preview = conv_preview.reshape((conv_preview.shape[0], conv_preview.shape[1] * conv_preview.shape[2]))
                        i = 1
                        while not (conv_preview.shape[1] / qs[1].shape[1]) % (i * i):
                            i *= 2
                        i //= 2
                        conv_reorganized = np.zeros((conv_preview.shape[0] * i, conv_preview.shape[1] // i), dtype=np.uint8)
                        for start in range(i):
                            conv_reorganized[start * conv_preview.shape[0]:(start + 1) * conv_preview.shape[0], :] = \
                                conv_preview[:, (conv_preview.shape[1] // i) * start:(conv_preview.shape[1] // i) * (start + 1)]
                        cv2.imshow(f'Agent {id + 1} - Convcam', conv_reorganized)
                        cv2.waitKey(1)
            else:
                action = np.random.randint(0, env.action_space_size)

            try:
                new_state, reward, done, _ = env.step(action)
            except:
                agent_stats[0] = AGENT_STATE.error
                time.sleep(1)
                break

            if show_preview[0] == 1:
                cv2.imshow(f'Agent {id+1} - preview', new_state[0])
                cv2.waitKey(1)
                env.preview_camera_enabled = False

            new_state[0] = agent.prepare_image(new_state[0])

            if show_preview[0] == 2:
                cv2.imshow(f'Agent {id+1} - preview', new_state[0])
                cv2.waitKey(1)
                env.preview_camera_enabled = False

            if show_preview[0] >= 10 or show_preview[0] == 3:
                if show_preview[0] == 3:
                    env.preview_camera_enabled = show_preview[1:]
                else:
                    env.preview_camera_enabled = settings.PREVIEW_CAMERA_RES[10 - int(show_preview[0])]
                if env.preview_camera is not None:
                    cv2.imshow(f'Agent {id+1} - preview', env.preview_camera)
                    cv2.waitKey(1)

            try:
                if not show_preview[0] and cv2.getWindowProperty(f'Agent {id+1} - preview', 0) >= 0:
                    cv2.destroyWindow(f'Agent {id + 1} - preview')
                    env.preview_camera_enabled = False
            except: pass

            episode_reward += reward
            if settings.AGENT_IMG_TYPE != AGENT_IMAGE_TYPE.stacked or step >= 3:
                transitions.put_nowait((current_state, action, reward, new_state, done))
            current_state = new_state

            if done:
                episode_end = time.time()
                break

            time_diff = episode_start + step / settings.EPISODE_FPS - time.time()
            time_diff2 = step_start + 1 / settings.EPISODE_FPS - time.time()
            if time_diff > 0:
                time.sleep(min(0.125, time_diff))
            elif time_diff2 > 0:
                time.sleep(min(0.125, time_diff2))

            step += 1
            fps_counter.append(time.time() - step_start)
            agent_stats[2] = len(fps_counter) / sum(fps_counter)

        try: env.destroy_agents()
        except: pass

        if done:
            episode_time = episode_end - episode_start
            average_fps = step / episode_time
            reward_factor = settings.EPISODE_FPS / average_fps
            episode_reward_weighted = (episode_reward - reward) * reward_factor + reward

            avg_predicted_qs = []
            for i in range(env.action_space_size + 1):
                if len(predicted_qs[i]):
                    avg_predicted_qs += [sum(predicted_qs[i]) / len(predicted_qs[i]),
                                         np.std(predicted_qs[i]),
                                         100 * predicted_actions[i] / predicted_actions[0]]
                else:
                    avg_predicted_qs += [-10**6, -10**6, -10**6]

            with episode.get_lock():
                episode.value += 1
                print(f"[Agent {id}] Episode {episode.value} finished. Reward: {episode_reward:.2f}, Steps: {step}")
                tensorboard_stats.put([episode.value, episode_reward, epsilon[0], episode_time,
                                       agent_stats[2], episode_reward_weighted] + avg_predicted_qs)

            if epsilon[0] > epsilon[2]:
                with epsilon.get_lock():
                    epsilon[0] *= epsilon[1]
                    epsilon[0] = max(epsilon[2], epsilon[0])

        agent_stats[0] = AGENT_STATE.restarting
        agent_stats[1] = 0
        agent_stats[2] = 0

    agent.terminate = True
    weight_updater.join()
    agent_stats[0] = AGENT_STATE.finished
    transitions.cancel_join_thread()
    tensorboard_stats.cancel_join_thread()
    carla_frametimes.cancel_join_thread()


def play(model_path, pause, console_print_callback):
    device = _get_device(settings.AGENT_GPU)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    agent = ARTDQNAgent(model_path, id=0)
    agent.device = device
    agent.model = agent.model.to(device)

    env = CarlaEnv(0, playing=True)
    env.frametimes = deque(maxlen=60)
    fps_counter = deque(maxlen=60)

    agent.get_qs([np.ones((env.im_height, env.im_width, 1 if settings.AGENT_IMG_TYPE == AGENT_IMAGE_TYPE.grayscaled else 3)), [0]])
    env.preview_camera_enabled = settings.PREVIEW_CAMERA_RES[0]

    while True:
        if pause.value == 1:
            pause.value = 2
        if pause.value == 2:
            time.sleep(0.1)
            continue
        if pause.value == 3:
            pause.value = 0
            try: env.destroy_agents()
            except: pass
            try:
                env = CarlaEnv(0, playing=True)
                env.frametimes = deque(maxlen=60)
                env.preview_camera_enabled = settings.PREVIEW_CAMERA_RES[0]
            except: pass
            time.sleep(1)
            continue

        current_state = env.reset()
        env.collision_hist = []
        current_state[0] = agent.prepare_image(current_state[0], create=True)
        last_processed_cam_update = 0
        conv_min = conv_max = None

        while True:
            step_start = time.time()
            if settings.AGENT_SYNCED:
                wait_start = time.time()
                while True:
                    if env.last_cam_update > last_processed_cam_update:
                        last_processed_cam_update = env.last_cam_update
                        break
                    if time.time() > wait_start + 1:
                        break
                    time.sleep(0.001)

            qs = agent.get_qs(current_state)
            action = np.argmax(qs[0])

            if agent.show_conv_cam and len(qs) > 1:
                conv_min = np.min(qs[1]) if conv_min is None else 0.8 * conv_min + 0.2 * np.min(qs[1])
                conv_max = np.max(qs[1]) if conv_max is None else 0.8 * conv_max + 0.2 * np.max(qs[1])
                if conv_max != conv_min:
                    conv_preview = ((qs[1] - conv_min) * 255 / (conv_max - conv_min)).astype(np.uint8)
                    conv_preview = np.moveaxis(conv_preview, 1, 2)
                    conv_preview = conv_preview.reshape((conv_preview.shape[0], conv_preview.shape[1] * conv_preview.shape[2]))
                    i = 1
                    while not (conv_preview.shape[1] / qs[1].shape[1]) % (i * i):
                        i *= 2
                    i //= 2
                    conv_reorganized = np.zeros((conv_preview.shape[0] * i, conv_preview.shape[1] // i), dtype=np.uint8)
                    for start in range(i):
                        conv_reorganized[start * conv_preview.shape[0]:(start + 1) * conv_preview.shape[0], :] = \
                            conv_preview[:, (conv_preview.shape[1] // i) * start:(conv_preview.shape[1] // i) * (start + 1)]
                    cv2.imshow(f'Agent - Convcam', conv_reorganized)
                    cv2.waitKey(1)

            new_state, reward, done, _ = env.step(action)
            new_state[0] = agent.prepare_image(new_state[0])

            if env.preview_camera is not None:
                cv2.imshow(f'Agent - preview', env.preview_camera)
                cv2.waitKey(1)

            current_state = new_state
            if done or pause.value > 0:
                break

            fps_counter.append(time.time() - step_start)
            console_print_callback(fps_counter, env, qs[0], action, ACTIONS_NAMES[env.actions[action]])

        env.destroy_agents()
