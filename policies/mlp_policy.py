import numpy as np
import torch
import torch.nn as nn


class MLPPolicy(nn.Module):

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=8
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):

        obs = torch.tensor(
            obs,
            dtype=torch.float32
        )

        with torch.no_grad():
            action = self.forward(obs)

        return action.numpy()

    ##################################################################
    # ZO-RankSGD needs parameter vectors
    ##################################################################

    def get_parameters(self):

        params = []

        for p in self.parameters():
            params.append(
                p.data.cpu().numpy().flatten()
            )

        return np.concatenate(params)

    def set_parameters(self, theta):

        pointer = 0

        for p in self.parameters():

            numel = p.numel()

            values = theta[
                pointer:pointer + numel
            ]

            values = values.reshape(
                p.shape
            )

            p.data = torch.tensor(
                values,
                dtype=torch.float32
            )

            pointer += numel

    def parameter_dimension(self):

        return len(
            self.get_parameters()
        )