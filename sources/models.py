import torch
import torch.nn as nn
import torch.nn.functional as F
import settings

MODEL_NAME_PREFIX = ''


# ---
# Helper

def _compute_feature_size(base, input_shape):
    """Run a dummy forward pass (on CPU) to compute the flattened feature size."""
    h, w, c = input_shape
    with torch.no_grad():
        dummy = torch.zeros(1, c, h, w)
        out = base(dummy)
    return out.shape[1]


# ---
# Model bases (feature extractors) — each returns an nn.Module
# Input tensor convention: (B, C, H, W)  (images stored as HWC are transposed before feeding)

def model_base_Xception(input_shape):
    raise NotImplementedError(
        "Xception base not implemented. Use a torchvision model or switch to another base.")


class _TestCNN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(5, stride=3, padding=2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(5, stride=3, padding=2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(5, stride=3, padding=2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(3, stride=2, padding=1),
            nn.Flatten(),
        )
    def forward(self, x): return self.net(x)

def model_base_test_CNN(input_shape):
    h, w, c = input_shape
    return _TestCNN(c)


class _64x3CNN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(5, stride=3, padding=2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(5, stride=3, padding=2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(5, stride=3, padding=2),
            nn.Flatten(),
        )
    def forward(self, x): return self.net(x)

def model_base_64x3_CNN(input_shape):
    h, w, c = input_shape
    return _64x3CNN(c)


class _4CNN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 5, padding=2), nn.ReLU(),
            nn.AvgPool2d(5, stride=3, padding=2),
            nn.Conv2d(64, 64, 5, padding=2), nn.ReLU(),
            nn.AvgPool2d(5, stride=3, padding=2),
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(),
            nn.AvgPool2d(3, stride=2, padding=1),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AvgPool2d(3, stride=2, padding=1),
            nn.Flatten(),
        )
    def forward(self, x): return self.net(x)

def model_base_4_CNN(input_shape):
    h, w, c = input_shape
    return _4CNN(c)


class _5ResidualCNN(nn.Module):
    """5-layer CNN with residual (concat) skip connections — mirrors model_base_5_residual_CNN."""
    def __init__(self, in_ch):
        super().__init__()
        c0 = in_ch
        self.conv1 = nn.Conv2d(c0, 64, 7, padding=3)
        c1 = 64 + c0
        self.pool1 = nn.AvgPool2d(5, stride=3, padding=2)

        self.conv2 = nn.Conv2d(c1, 64, 5, padding=2)
        c2 = 64 + c1
        self.pool2 = nn.AvgPool2d(5, stride=3, padding=2)

        self.conv3 = nn.Conv2d(c2, 128, 5, padding=2)
        c3 = 128 + c2
        self.pool3 = nn.AvgPool2d(5, stride=2, padding=2)

        self.conv4 = nn.Conv2d(c3, 256, 5, padding=2)
        c4 = 256 + c3
        self.pool4 = nn.AvgPool2d(5, stride=2, padding=2)

        self.conv5 = nn.Conv2d(c4, 512, 3, padding=1)
        self.pool5 = nn.AvgPool2d(3, stride=2, padding=1)
        self.flatten = nn.Flatten()

        # keep reference for convcam hook
        self.last_conv = self.conv5

    def forward(self, x):
        h = F.relu(self.conv1(x));  h = self.pool1(torch.cat([h, x], dim=1))
        h2 = F.relu(self.conv2(h)); h = self.pool2(torch.cat([h2, h], dim=1))
        h3 = F.relu(self.conv3(h)); h = self.pool3(torch.cat([h3, h], dim=1))
        h4 = F.relu(self.conv4(h)); h = self.pool4(torch.cat([h4, h], dim=1))
        h5 = F.relu(self.conv5(h)); h = self.flatten(self.pool5(h5))
        return h

def model_base_5_residual_CNN(input_shape):
    h, w, c = input_shape
    return _5ResidualCNN(c)


class _5ResidualCNNNoAct(nn.Module):
    """Same as _5ResidualCNN but without relu on the skip-concatenated branches."""
    def __init__(self, in_ch):
        super().__init__()
        c0 = in_ch
        self.conv1 = nn.Conv2d(c0, 64, 7, padding=3)
        c1 = 64 + c0
        self.pool1 = nn.AvgPool2d(5, stride=3, padding=2)

        self.conv2 = nn.Conv2d(c1, 64, 5, padding=2)
        c2 = 64 + c1
        self.pool2 = nn.AvgPool2d(5, stride=3, padding=2)

        self.conv3 = nn.Conv2d(c2, 128, 5, padding=2)
        c3 = 128 + c2
        self.pool3 = nn.AvgPool2d(5, stride=2, padding=2)

        self.conv4 = nn.Conv2d(c3, 256, 5, padding=2)
        c4 = 256 + c3
        self.pool4 = nn.AvgPool2d(5, stride=2, padding=2)

        self.conv5 = nn.Conv2d(c4, 512, 3, padding=1)
        self.pool5 = nn.AvgPool2d(3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.last_conv = self.conv5

    def forward(self, x):
        h = self.pool1(torch.cat([self.conv1(x), x], dim=1))
        h2 = self.pool2(torch.cat([self.conv2(h), h], dim=1))
        h3 = self.pool3(torch.cat([self.conv3(h2), h2], dim=1))
        h4 = self.pool4(torch.cat([self.conv4(h3), h3], dim=1))
        h = self.flatten(self.pool5(self.conv5(h4)))
        return h

def model_base_5_residual_CNN_noact(input_shape):
    h, w, c = input_shape
    return _5ResidualCNNNoAct(c)


class _5WideCNN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 7, stride=3, padding=3)

        # layer 2: 3 parallel paths
        self.c2a = nn.Conv2d(64, 64, 5, stride=3, padding=2)
        self.c2b = nn.Conv2d(64, 64, 3, stride=3, padding=1)
        self.pool2 = nn.AvgPool2d(3, stride=3, padding=1)  # 64 ch

        c2_out = 64 + 64 + 64  # 192

        # layer 3
        self.c3a = nn.Conv2d(c2_out, 128, 5, stride=2, padding=2)
        self.c3b = nn.Conv2d(c2_out, 128, 3, stride=2, padding=1)
        self.pool3 = nn.AvgPool2d(2, stride=2, padding=0)  # c2_out ch

        c3_out = 128 + 128 + c2_out  # 448

        # layer 4
        self.c4a = nn.Conv2d(c3_out, 256, 5, stride=2, padding=2)
        self.c4b = nn.Conv2d(c3_out, 256, 3, stride=2, padding=1)
        self.pool4 = nn.AvgPool2d(2, stride=2, padding=0)

        c4_out = 256 + 256 + c3_out  # 960

        # layer 5
        self.c5 = nn.Conv2d(c4_out, 512, 3, stride=2, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.last_conv = self.c5

    def forward(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = torch.cat([F.relu(self.c2a(h1)), F.relu(self.c2b(h1)), self.pool2(h1)], dim=1)
        h3 = torch.cat([F.relu(self.c3a(h2)), F.relu(self.c3b(h2)), self.pool3(h2)], dim=1)
        h4 = torch.cat([F.relu(self.c4a(h3)), F.relu(self.c4b(h3)), self.pool4(h3)], dim=1)
        h5 = F.relu(self.c5(h4))
        return self.flatten(self.gap(h5))

def model_base_5_wide_CNN(input_shape):
    h, w, c = input_shape
    return _5WideCNN(c)


class _5WideCNNNoAct(nn.Module):
    """5_wide_CNN without some relu activations on inner branches."""
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 7, stride=3, padding=3)

        self.c2a = nn.Conv2d(64, 64, 5, stride=3, padding=2)
        self.c2b = nn.Conv2d(64, 64, 3, stride=3, padding=1)
        self.pool2 = nn.AvgPool2d(3, stride=3, padding=1)
        c2_out = 192

        self.c3a = nn.Conv2d(c2_out, 128, 5, stride=2, padding=2)
        self.c3b = nn.Conv2d(c2_out, 128, 3, stride=2, padding=1)
        self.pool3 = nn.AvgPool2d(2, stride=2, padding=0)
        c3_out = 128 + 128 + c2_out

        self.c4a = nn.Conv2d(c3_out, 256, 5, stride=2, padding=2)
        self.c4b = nn.Conv2d(c3_out, 256, 3, stride=2, padding=1)
        self.pool4 = nn.AvgPool2d(2, stride=2, padding=0)
        c4_out = 256 + 256 + c3_out

        self.c5 = nn.Conv2d(c4_out, 512, 3, stride=2, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.last_conv = self.c5

    def forward(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = torch.cat([self.c2a(h1), self.c2b(h1), self.pool2(h1)], dim=1)
        h3 = torch.cat([self.c3a(h2), self.c3b(h2), self.pool3(h2)], dim=1)
        h4 = torch.cat([self.c4a(h3), self.c4b(h3), self.pool4(h3)], dim=1)
        return self.flatten(self.gap(self.c5(h4)))

def model_base_5_wide_CNN_noact(input_shape):
    h, w, c = input_shape
    return _5WideCNNNoAct(c)


# ---
# Model heads — wrap a base with output layers, return complete nn.Module

class _HiddenDenseDQN(nn.Module):
    """Base → optional kmh concat → Dense(hidden, relu) → Dense(outputs, linear)"""
    def __init__(self, base, feature_size, outputs, model_settings, use_kmh):
        super().__init__()
        self.base = base
        self.use_kmh = use_kmh
        in_size = feature_size + (1 if use_kmh else 0)
        self.fc1 = nn.Linear(in_size, model_settings['hidden_1_units'])
        self.fc2 = nn.Linear(model_settings['hidden_1_units'], outputs)

    def forward(self, x, kmh=None):
        feat = self.base(x)
        if self.use_kmh and kmh is not None:
            feat = torch.cat([feat, kmh], dim=1)
        return self.fc2(F.relu(self.fc1(feat)))


class _DirectDQN(nn.Module):
    """Base → optional kmh project (Dense4) concat → Dense(outputs, linear)"""
    def __init__(self, base, feature_size, outputs, model_settings, use_kmh):
        super().__init__()
        self.base = base
        self.use_kmh = use_kmh
        if use_kmh:
            self.kmh_fc = nn.Linear(1, 4)
            in_size = feature_size + 4
        else:
            in_size = feature_size
        self.out = nn.Linear(in_size, outputs)

    def forward(self, x, kmh=None):
        feat = self.base(x)
        if self.use_kmh and kmh is not None:
            feat = torch.cat([feat, F.relu(self.kmh_fc(kmh))], dim=1)
        return self.out(feat)


def model_head_hidden_dense(base, input_shape, outputs, model_settings):
    use_kmh = 'kmh' in settings.AGENT_ADDITIONAL_DATA
    feature_size = _compute_feature_size(base, input_shape)
    return _HiddenDenseDQN(base, feature_size, outputs, model_settings, use_kmh)


def model_head_direct(base, input_shape, outputs, model_settings):
    use_kmh = 'kmh' in settings.AGENT_ADDITIONAL_DATA
    feature_size = _compute_feature_size(base, input_shape)
    return _DirectDQN(base, feature_size, outputs, model_settings, use_kmh)
