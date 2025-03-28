import math
from timeit import default_timer as timer
from typing import Sequence, Tuple

from flax.linen.initializers import constant, orthogonal
import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from transformers import FlaxBertModel

from envs.pcgrl_env import PCGRLObs
from conf.config import EncoderConfig

try:
    import distrax
except:
    pass


def crop_rf(x, rf_size):
    mid_x = x.shape[1] // 2
    mid_y = x.shape[2] // 2
    return x[:, mid_x - math.floor(rf_size / 2):mid_x + math.ceil(rf_size / 2),
           mid_y - math.floor(rf_size / 2):mid_y + math.ceil(rf_size / 2)]


def crop_arf_vrf(x, arf_size, vrf_size):
    return crop_rf(x, arf_size), crop_rf(x, vrf_size)


class Dense(nn.Module):
    action_dim: Sequence[int]
    arf_size: int
    vrf_size: int
    activation: str = "tanh"
    hidden_dim: int = 700

    @nn.compact
    def __call__(self, map_x, flat_x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = jnp.concatenate((map_x.reshape((map_x.shape[0], -1)), flat_x), axis=-1)
        act = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        act = activation(act)
        act = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(act)
        act = activation(act)
        act = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(act)

        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class ConvForward2(nn.Module):
    """The way we crop out actions and values in ConvForward1 results in 
    values skipping conv layers, which is not what we intended. This matches
    the conv-dense model in the original paper without accounting for arf or 
    vrf."""
    action_dim: Sequence[int]
    act_shape: Tuple[int, int]
    hidden_dims: Tuple[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, map_x, flat_x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        flat_action_dim = self.action_dim * math.prod(self.act_shape)
        h1, h2 = self.hidden_dims

        map_x = nn.Conv(
            features=h1, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(map_x)
        act = activation(map_x)
        map_x = nn.Conv(
            features=h1, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(map_x)
        map_x = activation(map_x)

        map_x = act.reshape((act.shape[0], -1))
        x = jnp.concatenate((map_x, flat_x), axis=-1)

        x = nn.Dense(
            h2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        x = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = activation(x)

        act, critic = x, x

        act = nn.Dense(
            flat_action_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(act)

        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class ConvForward(nn.Module):
    action_dim: Sequence[int]
    act_shape: Tuple[int, int]
    arf_size: int
    vrf_size: int
    hidden_dims: Tuple[int]

    activation: str = "relu"

    @nn.compact
    def __call__(self, map_x, flat_x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        h1, h2 = self.hidden_dims

        flat_action_dim = self.action_dim * math.prod(self.act_shape)

        act, critic = crop_arf_vrf(map_x, self.arf_size, self.vrf_size)

        act = nn.Conv(
            features=h1, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(act)
        act = activation(act)
        act = nn.Conv(
            features=h1, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(act)
        act = activation(act)

        act = act.reshape((act.shape[0], -1))
        act = jnp.concatenate((act, flat_x), axis=-1)

        act = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(act)
        act = activation(act)

        act = nn.Dense(
            flat_action_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(act)

        critic = critic.reshape((critic.shape[0], -1))
        critic = jnp.concatenate((critic, flat_x), axis=-1)

        critic = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class NLPConvForward(nn.Module):
    action_dim: Sequence[int]
    act_shape: Tuple[int, int]
    arf_size: int
    vrf_size: int
    hidden_dims: Tuple[int]
    nlp_input_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, map_x, flat_x, nlp_x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        h1, h2 = self.hidden_dims

        flat_action_dim = self.action_dim * math.prod(self.act_shape)

        act, critic = crop_arf_vrf(map_x, self.arf_size, self.vrf_size)

        act = nn.Conv(
            features=h1, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(act)
        act = activation(act)
        act = nn.Conv(
            features=h1, kernel_size=(7, 7), strides=(2, 2), padding=(3, 3)
        )(act)
        act = activation(act)

        nlp_x = nn.Dense(
            h2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(nlp_x)

        nlp_x = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(nlp_x)

        act = act.reshape((act.shape[0], -1))
        act = jnp.concatenate((act, flat_x, nlp_x), axis=-1)

        act = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(act)
        act = activation(act)

        act = nn.Dense(
            flat_action_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(act)

        nlp_x = nn.Dense(
            h2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(nlp_x)

        nlp_x = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(nlp_x)

        critic = critic.reshape((critic.shape[0], -1))
        critic = jnp.concatenate((critic, flat_x, nlp_x), axis=-1)

        critic = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class EncoderNLPConvForward(nn.Module):
    config: EncoderConfig
    nlp_conv_forward: nn.Module
    encoder: nn.Module
    train_encoder: bool
    def __call__(self, map_x, flat_x, nlp_x, rng):
        if self.encoder is not None:
            inputs = {'x': nlp_x, 'train': self.train_encoder}

            if self.config.model == "mlp_vae":
                nlp_x = self.encoder(**inputs, rng=rng, deterministic=not self.train_encoder)
            else:
                nlp_x = self.encoder(**inputs)
        if not self.train_encoder:
            nlp_x = jax.lax.stop_gradient(nlp_x)

        act, critic = self.nlp_conv_forward(map_x, flat_x, nlp_x)

        return act, critic


class SeqNCA(nn.Module):
    action_dim: Sequence[int]
    act_shape: Tuple[int, int]
    arf_size: int
    vrf_size: int
    hidden_dims: Tuple[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, map_x, flat_x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        h1 = self.hidden_dims[0]

        hid = nn.Conv(
            features=h1, kernel_size=(3, 3), strides=(1, 1), padding="SAME"
        )(map_x)
        hid = activation(hid)

        act, critic = crop_arf_vrf(hid, self.arf_size, self.vrf_size)

        flat_action_dim = self.action_dim * math.prod(self.act_shape)

        act = act.reshape((act.shape[0], -1))
        act = jnp.concatenate((act, flat_x), axis=-1)
        act = nn.Dense(
            h1, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(act)
        act = nn.Dense(
            flat_action_dim, kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(act)

        critic = critic.reshape((critic.shape[0], -1))
        critic = jnp.concatenate((critic, flat_x), axis=-1)
        critic = nn.Dense(
            h1, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class NCA(nn.Module):
    representation: str
    tile_action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, map_x, flat_x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Tile the flat observations to match the map dimensions
        flat_x = jnp.tile(flat_x[:, None, None, :], (1, map_x.shape[1], map_x.shape[2], 1))

        # Concatenate the map and flat observations along the channel dimension
        x = jnp.concatenate((map_x, flat_x), axis=-1)

        x = nn.Conv(features=256, kernel_size=(9, 9), padding="SAME")(x)
        x = activation(x)
        x = nn.Conv(features=256, kernel_size=(5, 5), padding="SAME")(x)
        x = activation(x)
        x = nn.Conv(features=self.tile_action_dim,
                    kernel_size=(3, 3), padding="SAME")(x)

        if self.representation == 'wide':
            act = x.reshape((x.shape[0], -1))

        elif self.representation == 'nca':
            act = x

        else:
            raise NotImplementedError(f"Representation {self.representation} not implemented for NCA model.")

        # Generate random binary mask
        # mask = jax.random.uniform(rng[0], shape=actor_mean.shape) > 0.9
        # Apply mask to logits
        # actor_mean = actor_mean * mask
        # actor_mean = (actor_mean + x) / 2

        # actor_mean *= 10
        # actor_mean = nn.softmax(actor_mean, axis=-1)

        # critic = nn.Conv(features=256, kernel_size=(3,3), padding="SAME")(x)
        # critic = activation(critic)
        # # actor_mean = nn.Conv(
        #       features=256, kernel_size=(3,3), padding="SAME")(actor_mean)
        # # actor_mean = activation(actor_mean)
        # critic = nn.Conv(
        #       features=1, kernel_size=(1,1), padding="SAME")(critic)

        # return act, critic

        critic = activation(x)
        critic = nn.Conv(features=64, kernel_size=(5, 5), strides=(2, 2), padding="SAME")(x)
        critic = activation(critic)
        critic = nn.Conv(features=64, kernel_size=(5, 5), strides=(2, 2), padding="SAME")(x)
        critic = activation(critic)
        critic = critic.reshape((critic.shape[0], -1))
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class AutoEncoder(nn.Module):
    representation: str
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        act = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2),
                      padding="SAME")(x)
        act = activation(act)
        act = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2),
                      padding="SAME")(act)
        act = activation(act)
        act = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2),
                               padding="SAME")(act)
        act = activation(act)
        act = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2),
                               padding="SAME")(act)
        act = activation(act)
        act = nn.Conv(features=self.action_dim,
                      kernel_size=(3, 3), padding="SAME")(act)

        if self.representation == 'wide':
            act = act.reshape((x.shape[0], -1))

        critic = x.reshape((x.shape[0], -1))
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return act, jnp.squeeze(critic, axis=-1)


class ActorCriticPCGRL(nn.Module):
    """Transform the action output into a distribution. Do some pre- and post-processing specific to the 
    PCGRL environments."""
    subnet: nn.Module
    act_shape: Tuple[int, int]
    n_agents: int
    n_ctrl_metrics: int
    nlp_input_dim: int
    model_type: str

    @nn.compact
    def __call__(self, x: PCGRLObs, token=None, attention=None, rng=None):
        map_obs = x.map_obs
        ctrl_obs = x.flat_obs
        nlp_obs = x.nlp_obs

        # Hack. We had to put dummy ctrl obs's here to placate jax tree map during minibatch creation (FIXME?)
        # Now we need to remove them :)
        ctrl_obs = ctrl_obs[:, :self.n_ctrl_metrics]

        if self.model_type in ['nlpconv', 'nlpencconv', 'contconv']:
            act, val = self.subnet(map_obs, ctrl_obs, nlp_obs, rng)
        else:
            act, val = self.subnet(map_obs, ctrl_obs)

        act = act.reshape((act.shape[0], self.n_agents, *self.act_shape, -1))
        # val = val.reshape((n_gpu, n_envs))
        # act = act.reshape((n_gpu, n_envs, self.n_agents, *self.act_shape, -1))

        try:
            import distrax
        except:
            pass
        pi = distrax.Categorical(logits=act)

        return pi, val


class ActorCriticPlayPCGRL(nn.Module):
    """Transform the action output into a distribution."""
    subnet: nn.Module

    @nn.compact
    def __call__(self, x: PCGRLObs):
        map_obs = x.map_obs
        flat_obs = x.flat_obs
        act, val = self.subnet(map_obs, flat_obs)

        import distrax
        pi = distrax.Categorical(logits=act)
        return pi, val


class ActorCritic(nn.Module):
    """Transform the action output into a distribution."""
    subnet: nn.Module

    @nn.compact
    def __call__(self, x: PCGRLObs):
        act, val = self.subnet(x, jnp.zeros((x.shape[0], 0)))
        pi = distrax.Categorical(logits=act)
        return pi, val


if __name__ == '__main__':
    n_trials = 100
    rng = jax.random.PRNGKey(42)
    start_time = timer()
    for _ in range(n_trials):
        rng, _rng = jax.random.split(rng)
        data = jax.random.normal(rng, (4, 256, 2))
        print('data', data)
        import distrax

        dist = distrax.Categorical(data)
        sample = dist.sample(seed=rng)
        print('sample', sample)
        log_prob = dist.log_prob(sample)
        print('log_prob', log_prob)
    time = timer() - start_time
    print(f'Average time per sample: {time / n_trials}')
