import numpy as np
import torch
import torch.nn as nn


class LinearPolicy(nn.Module):

    def __init__(
        self,
        obs_dim,
        act_dim
    ):
        super().__init__()

        self.linear = nn.Linear(
            obs_dim,
            act_dim
        )

    def forward(self, x):
        return self.linear(x)

    def act(self, obs):

        obs = torch.tensor(
            obs,
            dtype=torch.float32
        )

        with torch.no_grad():
            action = self.forward(obs)

        return action.numpy()

    #################################################
    # Parameter vector interface
    #################################################

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
                pointer:pointer+numel
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