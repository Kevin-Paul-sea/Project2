"""
Model definitions for CIFAR-10 experiments.
"""

import numpy as np
from torch import nn

try:
    from utils.nn import init_weights_
except ImportError:
    from ..utils.nn import init_weights_


def get_number_of_parameters(model):
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()
    return int(parameters_n)


def make_activation(name="relu", inplace=True):
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    if name == "elu":
        return nn.ELU(inplace=inplace)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    raise ValueError(f"unknown activation: {name}")


def conv_norm_act(in_channels, out_channels, batch_norm=False, activation="relu"):
    layers = [
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=not batch_norm,
        )
    ]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(make_activation(activation))
    return nn.Sequential(*layers)


class VGG_A(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True, activation="relu"):
        super().__init__()
        self.features = nn.Sequential(
            conv_norm_act(inp_ch, 64, batch_norm=False, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_norm_act(64, 128, batch_norm=False, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_norm_act(128, 256, batch_norm=False, activation=activation),
            conv_norm_act(256, 256, batch_norm=False, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_norm_act(256, 512, batch_norm=False, activation=activation),
            conv_norm_act(512, 512, batch_norm=False, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_norm_act(512, 512, batch_norm=False, activation=activation),
            conv_norm_act(512, 512, batch_norm=False, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            make_activation(activation, inplace=False),
            nn.Linear(512, 512),
            make_activation(activation, inplace=False),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _init_weights(self):
        for module in self.modules():
            init_weights_(module)


class VGG_A_BatchNorm(nn.Module):
    def __init__(
        self,
        inp_ch=3,
        num_classes=10,
        init_weights=True,
        activation="relu",
        classifier_batch_norm=False,
    ):
        super().__init__()
        self.features = nn.Sequential(
            conv_norm_act(inp_ch, 64, batch_norm=True, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_norm_act(64, 128, batch_norm=True, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_norm_act(128, 256, batch_norm=True, activation=activation),
            conv_norm_act(256, 256, batch_norm=True, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_norm_act(256, 512, batch_norm=True, activation=activation),
            conv_norm_act(512, 512, batch_norm=True, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_norm_act(512, 512, batch_norm=True, activation=activation),
            conv_norm_act(512, 512, batch_norm=True, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        classifier_layers = [nn.Linear(512, 512)]
        if classifier_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(512))
        classifier_layers.extend(
            [
                make_activation(activation, inplace=False),
                nn.Linear(512, 512),
            ]
        )
        if classifier_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(512))
        classifier_layers.extend(
            [
                make_activation(activation, inplace=False),
                nn.Linear(512, num_classes),
            ]
        )
        self.classifier = nn.Sequential(*classifier_layers)
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _init_weights(self):
        for module in self.modules():
            init_weights_(module)


class VGG_A_Dropout(nn.Module):
    def __init__(
        self,
        inp_ch=3,
        num_classes=10,
        init_weights=True,
        activation="relu",
        dropout=0.5,
    ):
        super().__init__()
        self.features = VGG_A(
            inp_ch=inp_ch,
            num_classes=num_classes,
            init_weights=False,
            activation=activation,
        ).features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            make_activation(activation, inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            make_activation(activation, inplace=False),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _init_weights(self):
        for module in self.modules():
            init_weights_(module)


class VGG_A_Light(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True, activation="relu"):
        super().__init__()
        self.features = nn.Sequential(
            conv_norm_act(inp_ch, 16, batch_norm=False, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_norm_act(16, 32, batch_norm=False, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            make_activation(activation, inplace=False),
            nn.Linear(128, 128),
            make_activation(activation, inplace=False),
            nn.Linear(128, num_classes),
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _init_weights(self):
        for module in self.modules():
            init_weights_(module)


class CIFARConvNet(nn.Module):
    def __init__(
        self,
        inp_ch=3,
        num_classes=10,
        width_multiplier=1.0,
        activation="relu",
        batch_norm=True,
        dropout=0.2,
        init_weights=True,
    ):
        super().__init__()
        channels = [max(8, int(base * width_multiplier)) for base in (64, 128, 256)]
        hidden = max(64, int(256 * width_multiplier))
        self.features = nn.Sequential(
            conv_norm_act(inp_ch, channels[0], batch_norm=batch_norm, activation=activation),
            conv_norm_act(channels[0], channels[0], batch_norm=batch_norm, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_norm_act(channels[0], channels[1], batch_norm=batch_norm, activation=activation),
            conv_norm_act(channels[1], channels[1], batch_norm=batch_norm, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_norm_act(channels[1], channels[2], batch_norm=batch_norm, activation=activation),
            conv_norm_act(channels[2], channels[2], batch_norm=batch_norm, activation=activation),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(channels[2], hidden),
            make_activation(activation, inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def _init_weights(self):
        for module in self.modules():
            init_weights_(module)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation="relu"):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            make_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = make_activation(activation)

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


class CIFARResidualNet(nn.Module):
    def __init__(
        self,
        inp_ch=3,
        num_classes=10,
        width_multiplier=1.0,
        activation="relu",
        dropout=0.1,
        init_weights=True,
    ):
        super().__init__()
        channels = [max(8, int(base * width_multiplier)) for base in (32, 64, 128)]
        hidden = max(64, int(256 * width_multiplier))
        self.stem = conv_norm_act(inp_ch, channels[0], batch_norm=True, activation=activation)
        self.stage1 = nn.Sequential(
            ResidualBlock(channels[0], channels[0], activation=activation),
            ResidualBlock(channels[0], channels[0], activation=activation),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(channels[0], channels[1], stride=2, activation=activation),
            ResidualBlock(channels[1], channels[1], activation=activation),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(channels[1], channels[2], stride=2, activation=activation),
            ResidualBlock(channels[2], channels[2], activation=activation),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(channels[2], hidden),
            make_activation(activation, inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)

    def _init_weights(self):
        for module in self.modules():
            init_weights_(module)


def build_model(
    name,
    num_classes=10,
    activation="relu",
    width_multiplier=1.0,
    dropout=0.2,
    batch_norm=True,
):
    name = str(name).lower()
    if name == "vgg_a":
        return VGG_A(num_classes=num_classes, activation=activation)
    if name == "vgg_a_bn":
        return VGG_A_BatchNorm(num_classes=num_classes, activation=activation)
    if name == "vgg_a_dropout":
        return VGG_A_Dropout(num_classes=num_classes, activation=activation, dropout=dropout)
    if name == "vgg_a_light":
        return VGG_A_Light(num_classes=num_classes, activation=activation)
    if name == "cifar_conv":
        return CIFARConvNet(
            num_classes=num_classes,
            width_multiplier=width_multiplier,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )
    if name == "cifar_residual":
        return CIFARResidualNet(
            num_classes=num_classes,
            width_multiplier=width_multiplier,
            activation=activation,
            dropout=dropout,
        )
    raise ValueError(f"unknown model: {name}")


MODEL_NAMES = (
    "cifar_conv",
    "cifar_residual",
    "vgg_a",
    "vgg_a_bn",
    "vgg_a_dropout",
    "vgg_a_light",
)


if __name__ == "__main__":
    for model_name in MODEL_NAMES:
        model = build_model(model_name)
        print(f"{model_name} parameters={get_number_of_parameters(model)}")
