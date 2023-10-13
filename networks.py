import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd
import math_functions as mf
import tools
from torch import Tensor, Size
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

TensorTBICHW = Tensor
TensorTBIF = Tensor
TensorTBI = Tensor
TensorTB=Tensor

class RSSM(nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        layers_input=1,
        layers_output=1,
        rec_depth=1,
        shared=False,
        discrete=False,
        act="SiLU",
        norm="LayerNorm",
        mean_act="none",
        std_act="softplus",
        temp_post=True,
        min_std=0.1,
        cell="gru",
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._layers_input = layers_input
        self._layers_output = layers_output
        self._rec_depth = rec_depth
        self._shared = shared
        self._discrete = discrete
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._mean_act = mean_act
        self._std_act = std_act
        self._temp_post = temp_post
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._embed = embed
        self._device = device

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        if self._shared:
            inp_dim += self._embed
        for i in range(self._layers_input):
            inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            inp_layers.append(norm(self._hidden, eps=1e-03))
            inp_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._inp_layers = nn.Sequential(*inp_layers)
        self._inp_layers.apply(tools.weight_init)

        if cell == "gru":
            self._cell = GRUCell(self._hidden, self._deter)
            self._cell.apply(tools.weight_init)
        elif cell == "gru_layer_norm":
            self._cell = GRUCell(self._hidden, self._deter, norm=True)
            self._cell.apply(tools.weight_init)
        else:
            raise NotImplementedError(cell)

        img_out_layers = []
        inp_dim = self._deter
        for i in range(self._layers_output):
            img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            img_out_layers.append(norm(self._hidden, eps=1e-03))
            img_out_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        if self._temp_post:
            inp_dim = self._deter + self._embed
        else:
            inp_dim = self._embed
        for i in range(self._layers_output):
            obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
            obs_out_layers.append(norm(self._hidden, eps=1e-03))
            obs_out_layers.append(act())
            if i == 0:
                inp_dim = self._hidden
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._ims_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._ims_stat_layer.apply(tools.weight_init)
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.weight_init)
        else:
            self._ims_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._ims_stat_layer.apply(tools.weight_init)
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.weight_init)

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter).to(self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                stoch=torch.zeros([batch_size, self._stoch, self._discrete]).to(
                    self._device
                ),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros([batch_size, self._stoch]).to(self._device),
                std=torch.zeros([batch_size, self._stoch]).to(self._device),
                stoch=torch.zeros([batch_size, self._stoch]).to(self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine(self, action, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0])
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # if shared is True, prior and post both use same networks(inp_layers, _img_out_layers, _ims_stat_layer)
        # otherwise, post use different network(_obs_out_layers) with prior[deter] and embed as inputs
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()

        if torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        prior = self.img_step(prev_state, prev_action, None, sample)
        if self._shared:
            post = self.img_step(prev_state, prev_action, embed, sample)
        else:
            if self._temp_post:
                x = torch.cat([prior["deter"], embed], -1)
            else:
                x = embed
            # (batch_size, prior_deter + embed) -> (batch_size, hidden)
            x = self._obs_out_layers(x)
            # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
            stats = self._suff_stats_layer("obs", x)
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mode()
            post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    # this is used for making future image
    def img_step(self, prev_state, prev_action, embed=None, sample=True):
        # (batch, stoch, discrete_num)
        prev_action *= (1.0 / torch.clip(torch.abs(prev_action), min=1.0)).detach()
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        if self._shared:
            if embed is None:
                shape = list(prev_action.shape[:-1]) + [self._embed]
                embed = torch.zeros(shape)
            # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action, embed)
            x = torch.cat([prev_stoch, prev_action, embed], -1)
        else:
            x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._inp_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._ims_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._ims_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        rep_loss = torch.mean(torch.clip(rep_loss, min=free))
        dyn_loss = torch.mean(torch.clip(dyn_loss, min=free))
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        wm_type,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
    ):
        super(MultiEncoder, self).__init__()
        self.wm_type=wm_type
        excluded = ("is_first", "is_last", "is_cont", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, self.wm_type,cnn_depth, act, norm, kernel_size, minres
            )
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
            )
            self.outdim += mlp_units

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs

class MultiDecoder(nn.Module):
    def __init__(
        self,
        shapes,
        features_dim,
        conf,
    ):
        super(MultiDecoder, self).__init__()
        
        self.wm_type=conf.wm_type
        self.image_weight = conf.image_scale
        # self.vecobs_weight = conf.vecobs_weight
        self.reward_weight = conf.reward_scale
        self.cont_weight = conf.cont_scale
        ## image decoder part
        excluded = ("is_first", "is_last", "is_cont")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(conf.decoder["cnn_keys"], k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(conf.decoder["mlp_keys"], k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self.image = ConvDecoder(features_dim,
                                     self.wm_type,
                                     shape,
                                     conf.decoder["cnn_depth"],
                                     conf.decoder["act"], conf.decoder["norm"], conf.decoder["kernel_size"], 
                conf.decoder["minres"],cnn_sigmoid=conf.decoder["cnn_sigmoid"],image_dist=conf.decoder["image_dist"])
        if self.mlp_shapes:
            self.image = MLP(
                features_dim,
                self.mlp_shapes,
                conf.decoder["mlp_layers"],
                conf.decoder["mlp_units"],
                conf.decoder["act"],
                conf.decoder["norm"],
                conf.decoder["vector_dist"],
            )
        if conf.wm_type=='v2':
            if conf.reward_decoder_categorical:
                self.reward = DenseCategoricalSupportDecoder(
                    in_dim=features_dim,
                    support=mf.clip_rewards_np(conf.reward_decoder_categorical, conf.clip_rewards),  # reward_decoder_categorical values are untransformed 
                    hidden_layers=conf.reward_layers,
                    norm=conf.norm)
            else:
                self.reward = DenseNormalDecoder(in_dim=features_dim, hidden_layers=conf.reward_layers, norm=conf.norm)

            self.cont = DenseBernoulliDecoder(in_dim=features_dim, hidden_layers=conf.cont_layers, norm=conf.norm)
        elif conf.wm_type=='v3':
            if conf.reward_head == "symlog_disc":
                self.reward = MLP(
                    features_dim,  # pytorch version
                    (255,),
                    conf.reward_layers,
                    conf.units,
                    conf.act,
                    conf.norm,
                    dist=conf.reward_head,
                    outscale=0.0,
                    device=conf.device,
                )
            else:
                self.reward = MLP(
                    features_dim,  # pytorch version
                    [],
                    conf.reward_layers,
                    conf.units,
                    conf.act,
                    conf.norm,
                    dist=conf.reward_head,
                    outscale=0.0,
                    device=conf.device,
                )
            self.cont = MLP(
                features_dim,  # pytorch version
                [],
                conf.cont_layers,
                conf.units,
                conf.act,
                conf.norm,
                dist="binary",
                device=conf.device,
            )
        self._image_dist = conf.decoder["image_dist"]

    def training_step(self,
                      features,
                      obs,
                      extra_metrics: bool = False
                      ) :
        tensors = {}
        metrics = {}
        loss_reconstr = 0

        if self.image:
            loss_image, image_rec = self.image.training_step(features, obs['image'])
            loss_reconstr += self.image_weight * loss_image
            metrics.update(loss_image=loss_image.detach().mean().cpu())
            tensors.update(loss_image=loss_image.detach(),
                           image_rec=image_rec.detach())

        # if self.vecobs:
        #     loss_vecobs, vecobs_rec = self.vecobs.training_step(features, obs['vecobs'])
        #     loss_reconstr += self.vecobs_weight * loss_vecobs
        #     metrics.update(loss_vecobs=loss_vecobs.detach().mean())
        #     tensors.update(loss_vecobs=loss_vecobs.detach(),
        #                    vecobs_rec=vecobs_rec.detach())

        loss_reward, reward_rec = self.reward.training_step(features, obs['reward'])
        loss_reconstr += self.reward_weight * loss_reward
        metrics.update(loss_reward=loss_reward.detach().mean().cpu())
        tensors.update(loss_reward=loss_reward.detach(),
                       reward_rec=reward_rec.detach())

        loss_cont, cont_rec = self.cont.training_step(features, obs['cont'])
        loss_reconstr += self.cont_weight * loss_cont
        metrics.update(loss_cont=loss_cont.detach().mean().cpu())
        tensors.update(loss_cont=loss_cont.detach(),
                       cont_rec=cont_rec.detach())

        # if extra_metrics:
        #     if isinstance(self.reward, DenseCategoricalSupportDecoder):
        #         # TODO: logic should be moved to appropriate decoder
        #         reward_cat = self.reward.to_categorical(obs['reward'])
        #         for i in range(len(self.reward.support)):
        #             # Logprobs for specific categorical reward values
        #             mask_rewardp = reward_cat == i  # mask where categorical reward has specific value
        #             loss_rewardp = loss_reward * mask_rewardp / mask_rewardp  # set to nan where ~mask
        #             # metrics[f'loss_reward{i}'] = nanmean(loss_rewardp)  # index by support bucket, not by value
        #             tensors[f'loss_reward{i}'] = loss_rewardp
        #     else:
        #         for sig in [-1, 1]:
        #             # Logprobs for positive and negative rewards
        #             mask_rewardp = torch.sign(obs['reward']) == sig  # mask where reward is positive or negative
        #             loss_rewardp = loss_reward * mask_rewardp / mask_rewardp  # set to nan where ~mask
        #             # metrics[f'loss_reward{sig}'] = nanmean(loss_rewardp)
        #             tensors[f'loss_reward{sig}'] = loss_rewardp

        #     mask_cont1 = obs['cont'] > 0  # mask where cont is 1
        #     loss_cont1 = loss_cont * mask_cont1 / mask_cont1  # set to nan where ~mask
        #     # metrics['loss_cont1'] = nanmean(loss_cont1)
        #     tensors['loss_cont1'] = loss_cont1

        return loss_reconstr, metrics, tensors


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        wm_type,
        depth=32,
        act="SiLU",
        norm="LayerNorm",
        kernel_size=4,
        minres=4,
    ):
        super(ConvEncoder, self).__init__()
        self.wm_type=wm_type
        if self.wm_type=='v2':
            act = getattr(torch.nn, act)
            #中间层的channer数
            stride = 2
            h, w, input_ch = input_shape
            self.layers = nn.Sequential(
                nn.Conv2d(input_ch, depth, kernel_size, stride),
                act(),
                nn.Conv2d(depth, depth * 2, kernel_size, stride),
                act(),
                nn.Conv2d(depth * 2, depth * 4, kernel_size, stride),
                act(),
                nn.Conv2d(depth* 4, depth * 8, kernel_size, stride),
                act(),
                # nn.Flatten()
            )
            self.outdim = depth * 32
        elif self.wm_type=='v3':
            act = getattr(torch.nn, act)
            norm = getattr(torch.nn, norm)
            h, w, input_ch = input_shape
            layers = []
            for i in range(int(np.log2(h) - np.log2(minres))):
                if i == 0:
                    in_dim = input_ch
                else:
                    in_dim = 2 ** (i - 1) * depth
                out_dim = 2**i * depth
                layers.append(
                    Conv2dSame(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        kernel_size=kernel_size,
                        stride=2,
                        bias=False,
                    )
                )
                layers.append(ChLayerNorm(out_dim))
                layers.append(act())
                h, w = h // 2, w // 2

            self.outdim = out_dim * h * w
            self.layers = nn.Sequential(*layers)
            self.layers.apply(tools.weight_init)

    def forward(self, obs):
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        wm_type,
        shape=(3, 64, 64),
        cnn_depth=32,
        act=nn.ELU,
        norm=nn.LayerNorm,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
        image_dist='mse',
    ):
        super(ConvDecoder, self).__init__()
        self.wm_type=wm_type
        self._image_dist=image_dist
        try:
            act = getattr(torch.nn, act)
            norm = getattr(torch.nn, norm)
        except TypeError:
            print('already done!')
        self._shape=shape
        self._cnn_sigmoid = cnn_sigmoid
        
        if self.wm_type=='v2':
            kernels = (5, 5, 6, 6)
            stride = 2
            self._embed_size = cnn_depth*32
            # if linear_layers == 0:
            # layers = [
            #     nn.Linear(feat_size, cnn_depth * 32),  # No activation here in DreamerV2
            # ]
            self._linear_layer = nn.Linear(feat_size, self._embed_size)
            # else:
            #     hidden_dim = cnn_depth * 32
            #     layers = [
            #         nn.Linear(feat_size, hidden_dim),
            #         norm(hidden_dim, eps=1e-3),
            #         act()
            #     ]
            #     for _ in range(linear_layers - 1):
            #         layers += [
            #             nn.Linear(hidden_dim, hidden_dim),
            #             norm(hidden_dim, eps=1e-3),
            #             act()]
            
            self.layers = nn.Sequential(
            # FC
            # *layers,
            # nn.Unflatten(-1, (cnn_depth * 32, 1, 1)),
            # Deconv
            nn.ConvTranspose2d(cnn_depth * 32, cnn_depth * 4, kernels[0], stride),
            act(),
            nn.ConvTranspose2d(cnn_depth * 4, cnn_depth * 2, kernels[1], stride),
            act(),
            nn.ConvTranspose2d(cnn_depth * 2, cnn_depth, kernels[2], stride),
            act(),
            nn.ConvTranspose2d(cnn_depth, shape[0], kernels[3], stride))
            
        elif self.wm_type=='v3':
            self._minres = minres
            layer_num = int(np.log2(shape[1]) - np.log2(minres))
            self._embed_size = minres**2 * cnn_depth * 2 ** (layer_num - 1)

            self._linear_layer = nn.Linear(feat_size, self._embed_size)
            self._linear_layer.apply(tools.weight_init)
            in_dim = self._embed_size // (minres**2)

            layers = []
            h, w = minres, minres
            for i in range(layer_num):
                out_dim = self._embed_size // (minres**2) // (2 ** (i + 1))
                bias = False
                initializer = tools.weight_init
                if i == layer_num - 1:
                    out_dim = self._shape[0]
                    act = False
                    bias = True
                    norm = False
                    initializer = tools.uniform_weight_init(outscale)

                if i != 0:
                    in_dim = 2 ** (layer_num - (i - 1) - 2) * cnn_depth
                pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
                pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
                layers.append(
                    nn.ConvTranspose2d(
                        in_dim,
                        out_dim,
                        kernel_size,
                        2,
                        padding=(pad_h, pad_w),
                        output_padding=(outpad_h, outpad_w),
                        bias=bias,
                    )
                )
                if norm:
                    layers.append(ChLayerNorm(out_dim))
                if act:
                    layers.append(act())
                [m.apply(initializer) for m in layers[-3:]]
                h, w = h * 2, w * 2

            self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        if self.wm_type=='v2':
            x=x.reshape(
                 [-1, 1, 1, self._embed_size]
            )
        elif self.wm_type=='v3':
            
            x = x.reshape(
                [-1, self._minres, self._minres, self._embed_size // self._minres**2]
            )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch * time, ch, h, w) necessary???
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch * time, ch, h, w) -> (batch * time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean) - 0.5
        return mean
    
    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)
    
    def loss(self, output, target):
        ##保持64*64*3不变，把后三维给整出来
        output, bd = mf.flatten_batch(output, 3)
        target, _ = mf.flatten_batch(target, 3)
        loss = 0.5 * torch.square(output - target).sum(dim=[-1, -2, -3])  # MSE
        # print(loss[:10])
        return mf.unflatten_batch(loss, bd)

    def training_step(self, features , target) :
        # assert len(features.shape) == 4 and len(target.shape) == 5
        # I = features.shape[2]
        # target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it
        if self.wm_type=='v2':
            decoded = self.forward(features)
            loss_tb = self.loss(decoded, target)
        # loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        # decoded = decoded.mean(dim=2)  # TBICHW => TBCHW

        # assert len(loss_tbi.shape) == 3 and len(decoded.shape) == 5
            return loss_tb, decoded
        elif self.wm_type=='v3':
            # dists = {}
            feat = features
            outputs = self.forward(feat)
            # split_sizes = [v[-1] for v in self._shape.values()]
            # split_sizes
            # outputs = torch.split(outputs, split_sizes, -1)
            # dists.update(
            #     {
            #         key: self._make_image_dist(output)
            #         for key, output in zip(self._shape.keys(), outputs)
            #     }
            # )
            # if self.mlp_shapes:
            #     dists.update(self._mlp(features))
            dist=self._make_image_dist(outputs)
            loss=-dist.log_prob(target)
            decoded=dist.mode()
            
            return loss,decoded
        


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm="LayerNorm",
        dist="normal",
        std=1.0,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
    ):
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        self._layers = layers
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._dist = dist
        self._std = std
        self._symlog_inputs = symlog_inputs
        self._device = device

        layers = []
        for index in range(self._layers):
            layers.append(nn.Linear(inp_dim, units, bias=False))
            layers.append(norm(units, eps=1e-03))
            layers.append(act())
            if index == 0:
                inp_dim = units
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
                
    def loss(self, output: torchd.Distribution, target):
        return -output.log_prob(target)
        
    def training_step(self, features,target):
    # if self._dist=='binary':
        # assert len(features.shape) == 4
        # # I = features.shape[2]
        # # target = insert_dim(target, 2, I)  # Expand target with iwae_samples dim, because features have it
        # if len(target.shape)==3:
        #     target= target.unsqueeze(-1)

        decoded = self.forward(features)
        loss_tb = self.loss(decoded, target)
        # loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        # decoded = decoded.mode().mean(dim=-2)
        decoded=decoded.mode()

        # assert len(loss_tbi.shape) == 3
        if len(decoded.shape)==3:
            decoded=decoded.squeeze(-1)
        if len(loss_tb.shape)==3:
            loss_tb=loss_tb.squeeze(-1)
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == 2
        # Now,
        # len(loss_tb.shape)==2
        #decoded是3？
        return loss_tb, decoded

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(
                    torchd.normal.Normal(mean, std), len(shape)
                )
            )
        if dist == "huber":
            return tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0), len(shape)
                )
            )
        if dist == "binary":
            return tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        if dist == "symlog_disc":
            return tools.DiscDist(logits=mean, device=self._device)
        if dist == "symlog_mse":
            return tools.SymlogDist(mean)
        raise NotImplementedError(dist)
    
class MLP_v2(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, hidden_layers, norm, act='ELU'):
        super().__init__()
        self.out_dim = out_dim
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        layers = []
        dim = in_dim
        for i in range(hidden_layers):
            layers += [
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                act()
            ]
            dim = hidden_dim
        layers += [
            nn.Linear(dim, out_dim),
        ]
        if out_dim == 1:
            layers += [
                nn.Flatten(0),
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x, bd = mf.flatten_batch(x)
        y = self.model(x)
        y = mf.unflatten_batch(y, bd)
        return y
    
    # def forward_actor(self, features: Tensor,dist) -> torchd.Distribution:
    #     # if self.wm_type=='v2':
    #     y = self.forward(features).float()  # .float() to force float32 on AMP
        
    #     if dist == 'onehot':
    #         return torchd.OneHotCategorical(logits=y)
        
    #     if dist == 'normal_tanh':
    #         return mf.normal_tanh(y)

    #     if dist == 'tanh_normal':
    #         return mf.tanh_normal(y)
    #     if dist == 'normal':
    #         return mf.normal(y)
    # # elif self.wm_type=='v3':
    # #     y = self.actor.forward(features)
        
    #     assert False, dist
    


class ActionHead(nn.Module):
    def __init__(
        self,
        inp_dim,
        size,
        layers,
        units,
        act=nn.ELU,
        norm=nn.LayerNorm,
        dist="trunc_normal",
        init_std=0.0,
        min_std=0.1,
        max_std=1.0,
        temp=0.1,
        outscale=1.0,
        unimix_ratio=0.01,
    ):
        super(ActionHead, self).__init__()
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._min_std = min_std
        self._max_std = max_std
        self._init_std = init_std
        self._unimix_ratio = unimix_ratio
        self._temp = temp() if callable(temp) else temp

        pre_layers = []
        for index in range(self._layers):
            pre_layers.append(nn.Linear(inp_dim, self._units, bias=False))
            pre_layers.append(norm(self._units, eps=1e-03))
            pre_layers.append(act())
            if index == 0:
                inp_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)
        self._pre_layers.apply(tools.weight_init)

        if self._dist in ["tanh_normal", "tanh_normal_5", "normal", "trunc_normal"]:
            self._dist_layer = nn.Linear(self._units, 2 * self._size)
            self._dist_layer.apply(tools.uniform_weight_init(outscale))

        elif self._dist in ["normal_1", "onehot", "onehot_gumbel"]:
            self._dist_layer = nn.Linear(self._units, self._size)
            self._dist_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        x = self._pre_layers(x)
        if self._dist == "tanh_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "tanh_normal_5":
            x = self._dist_layer(x)
            mean, std = torch.split(x, 2, -1)
            mean = 5 * torch.tanh(mean / 5)
            std = F.softplus(std + 5) + 5
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == "normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "normal_1":
            mean = self._dist_layer(x)
            dist = torchd.normal.Normal(mean, 1)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "trunc_normal":
            x = self._dist_layer(x)
            mean, std = torch.split(x, [self._size] * 2, -1)
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(torchd.independent.Independent(dist, 1))
        elif self._dist == "onehot":
            x = self._dist_layer(x)
            dist = tools.OneHotDist(x, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            x = self._dist_layer(x)
            temp = self._temp
            dist = tools.ContDist(torchd.gumbel.Gumbel(x, 1 / temp))
        else:
            raise NotImplementedError(self._dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + size, 3 * size, bias=False)
        if norm:
            self._norm = nn.LayerNorm(3 * size, eps=1e-03)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super(ChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DenseBernoulliDecoder(nn.Module):

    def __init__(self, in_dim, hidden_dim=400, hidden_layers=2, norm='LayerNorm'):
        super().__init__()
        self.model = MLP_v2(in_dim, 1, hidden_dim, hidden_layers, norm)

    def forward(self, features: Tensor) -> torchd.Distribution:
        y = self.model.forward(features)
        p = torchd.Bernoulli(logits=y.float())
        return p

    def loss(self, output: torchd.Distribution, target: Tensor) -> Tensor:
        return -output.log_prob(target)

    def training_step(self, features: TensorTBIF, target: Tensor) -> Tuple[TensorTBI, TensorTB, TensorTB]:
        assert len(features.shape) == 3

        decoded = self.forward(features)
        if len(target.shape)==3:
           target=target.squeeze(-1)
        loss_tb = self.loss(decoded, target)
        # loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        # decoded = decoded.mode().mean(dim=-2)
        # decoded=decoded.mode()
        decoded=decoded.mean

        # assert len(loss_tbi.shape) == 3
        if len(decoded.shape)==3:
            decoded=decoded.squeeze(-1)
        if len(loss_tb.shape)==3:
            loss_tb=loss_tb.squeeze(-1)
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == 2
        # Now,
        # len(loss_tb.shape)==2
        #decoded是3？
        return loss_tb, decoded

        # assert len(loss_tbi.shape) == 3
        # assert len(loss_tb.shape) == 2
        # assert len(decoded.shape) == 2
        # return loss_tbi, loss_tb, decoded


class DenseNormalDecoder(nn.Module):

    def __init__(self, in_dim, out_dim=1, hidden_dim=400, hidden_layers=2, norm='LayerNorm', std=0.3989422804):
        super().__init__()
        self.model = MLP_v2(in_dim, out_dim, hidden_dim, hidden_layers, norm)
        self.std = std
        self.out_dim = out_dim

    def forward(self, features: Tensor) -> torchd:
        y = self.model.forward(features)
        p = torchd.Normal(loc=y, scale=torch.ones_like(y) * self.std)
        if self.out_dim > 1:
            p = torchd.independent.Independent(p, 1)  # Makes p.logprob() sum over last dim
        return p

    def loss(self, output: torchd.Distribution, target: Tensor) -> Tensor:
        var = self.std ** 2  # var cancels denominator, which makes loss = 0.5 (target-output)^2
        return -output.log_prob(target) * var

    def training_step(self, features: TensorTBIF, target: Tensor) -> Tuple[TensorTBI, TensorTB, Tensor]:
        
        decoded = self.forward(features)
        loss_tb = self.loss(decoded, target)
        # loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        # decoded = decoded.mode().mean(dim=-2)
        # decoded=decoded.mode()
        decoded=decoded.mean

        # assert len(loss_tbi.shape) == 3
        if len(decoded.shape)==3:
            decoded=decoded.squeeze(-1)
        if len(loss_tb.shape)==3:
            loss_tb=loss_tb.squeeze(-1)
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == 2
        # Now,
        # len(loss_tb.shape)==2
        #decoded是3？
        return loss_tb, decoded

        # assert len(loss_tbi.shape) == 3
        # assert len(loss_tb.shape) == 2
        # assert len(decoded.shape) == (2 if self.out_dim == 1 else 3)
        # return loss_tbi, loss_tb, decoded


class DenseCategoricalSupportDecoder(nn.Module):
    """
    Represent continuous variable distribution by discrete set of support values.
    Useful for reward head, which can be e.g. [-10, 0, 1, 10]
    """

    def __init__(self, in_dim, support=[0.0, 1.0], hidden_dim=400, hidden_layers=2, norm='LayerNorm'):
        assert isinstance(support, (list, np.ndarray))
        super().__init__()
        self.model = MLP_v2(in_dim, len(support), hidden_dim, hidden_layers, norm)
        self.support = np.array(support).astype(float)
        self._support = nn.Parameter(torch.tensor(support).to(torch.float), requires_grad=False)

    def forward(self, features: Tensor) -> torchd:
        y = self.model.forward(features)
        p = CategoricalSupport(logits=y.float(), sup=self._support.data)
        return p

    def loss(self, output: torchd, target: Tensor) -> Tensor:
        target = self.to_categorical(target)
        return -output.log_prob(target)

    def to_categorical(self, target: Tensor) -> Tensor:
        # TODO: should interpolate between adjacent values, like in MuZero
        distances = torch.square(target.unsqueeze(-1) - self._support)
        return distances.argmin(-1)

    def training_step(self, features: TensorTBIF, target: Tensor) -> Tuple[TensorTBI, TensorTB, TensorTB]:
        decoded = self.forward(features)
        loss_tb = self.loss(decoded, target)
        # loss_tb = -logavgexp(-loss_tbi, dim=2)  # TBI => TB
        # decoded = decoded.mode().mean(dim=-2)
        # decoded=decoded.mode()
        decoded=decoded.mean

        # assert len(loss_tbi.shape) == 3
        if len(decoded.shape)==3:
            decoded=decoded.squeeze(-1)
        if len(loss_tb.shape)==3:
            loss_tb=loss_tb.squeeze(-1)
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == 2
        # Now,
        # len(loss_tb.shape)==2
        #decoded是3？
        return loss_tb, decoded

        # assert len(loss_tbi.shape) == 3
        # assert len(loss_tb.shape) == 2
        # assert len(decoded.shape) == 2
        # return loss_tbi, loss_tb, decoded
        
class CategoricalSupport(torchd.Categorical):

    def __init__(self, logits, sup):
        assert logits.shape[-1:] == sup.shape
        super().__init__(logits=logits)
        self.sup = sup

    @property
    def mean(self):
        return torch.einsum('...i,i->...', self.probs, self.sup)
    
class NoNorm(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x
