import copy
import torch
from torch import nn
from torch import Tensor
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()
TensorHM=Tensor
TensorJM=Tensor


class RewardEMA(object):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.values = torch.zeros((2,)).to(device)
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x):
        ## offset:较小的分位数
        ## scale:大分位数与小分位数做差
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        self.values = self.alpha * x_quantile + (1 - self.alpha) * self.values
        scale = torch.clip(self.values[1] - self.values[0], min=1.0)
        offset = self.values[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self.wm_type=config.wm_type
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        
        ## Encoder
        
        self.encoder = networks.MultiEncoder(shapes, self.wm_type,**config.encoder,)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_input_layers,
            config.dyn_output_layers,
            config.dyn_rec_depth,
            config.dyn_shared,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_temp_post,
            config.dyn_min_std,
            config.dyn_cell,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        # self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        #     self.heads["decoder"] = networks.MultiDecoder(
        #     feat_size, self.wm_type,shapes, **config.decoder
        # )
        self.decoder = networks.MultiDecoder(shapes,feat_size, config)
        # if self.wm_type=='v2':
            
        # elif self.wm_type=='v3':
            
        #     self.heads["decoder"] = networks.MultiDecoder(
        #         feat_size, shapes, **config.decoder
        #     )
        #     if config.reward_head == "symlog_disc":
        #         self.heads["reward"] = networks.MLP(
        #             feat_size,  # pytorch version
        #             (255,),
        #             config.reward_layers,
        #             config.units,
        #             config.act,
        #             config.norm,
        #             dist=config.reward_head,
        #             outscale=0.0,
        #             device=config.device,
        #         )
        #     else:
        #         self.heads["reward"] = networks.MLP(
        #             feat_size,  # pytorch version
        #             [],
        #             config.reward_layers,
        #             config.units,
        #             config.act,
        #             config.norm,
        #             dist=config.reward_head,
        #             outscale=0.0,
        #             device=config.device,
        #         )
        #     self.heads["cont"] = networks.MLP(
        #         feat_size,  # pytorch version
        #         [],
        #         config.cont_layers,
        #         config.units,
        #         config.act,
        #         config.norm,
        #         dist="binary",
        #         device=config.device,
        #     )
        # for name in config.grad_heads:
        #     assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        self._scales = dict(reward=config.reward_scale, cont=config.cont_scale)

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                if self.wm_type=='v2':
                    kl_loss, kl_value, dyn_loss, rep_loss= self.dynamics.kl_loss(
                    post, prior, -100000.0,self._config.kl_balance, self._config.kl_balance)
                elif self.wm_type=='v3':
                    kl_free = tools.schedule(self._config.kl_free, self._step)
                    dyn_scale = tools.schedule(self._config.dyn_scale, self._step)
                    rep_scale = tools.schedule(self._config.rep_scale, self._step)
                    kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                        post, prior, kl_free, dyn_scale, rep_scale
                    )
                # preds = {}
                # for name, head in self.heads.items():
                #     grad_head = name in self._config.grad_heads
                #     feat = self.dynamics.get_feat(post)
                #     feat = feat if grad_head else feat.detach()
                #     pred = head(feat)
                #     if type(pred) is dict:
                #         preds.update(pred)
                #     else:
                #         preds[name] = pred
                # losses = {}
                # for name, pred in preds.items():
                #     like = pred.log_prob(data[name])
                #     losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
                features = self.dynamics.get_feat(post)
                loss_reconstr, metrics_dec, tensors = self.decoder.training_step(features, data)
                model_loss = torch.mean(loss_reconstr) + self._config.kl_scale * kl_loss
            metrics = self._model_opt(model_loss, self.parameters())

        # metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics.update(metrics_dec)
        if self.wm_type=='v3':
            metrics["kl_free"] = kl_free
            metrics["dyn_scale"] = dyn_scale
            metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()
        obs["image"] = torch.Tensor(obs["image"]) / 255.0 - 0.5
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        # recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
        #     :6
        # ]
        recon = self.decoder.image._make_image_dist(self.decoder.image(self.dynamics.get_feat(states))).mode()[
            :6
        ]
        # reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        # reward_post = self.decoder.reward(self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine(data["action"][:6, 5:], init)
        # openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        # reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        openl=self.decoder.image._make_image_dist(self.decoder.image(self.dynamics.get_feat(prior))).mode()
        # reward_prior = self.decoder.reward(self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6] + 0.5
        model = model + 0.5
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self.wm_type=config.wm_type
        self._world_model = world_model
        self._stop_grad_actor = stop_grad_actor
        self._reward = reward
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
            
            
        if self.wm_type=='v2':
            # actor_out_dim = config.num_actions if config.actor_dist in ["normal_1", "onehot", "onehot_gumbel"]  else 2 * config.num_actions
            # self.actor = networks.MLP_v2(feat_size, actor_out_dim, config.units,  config.actor_layers, config.layer_norm)
            self.actor = networks.ActionHead(
                feat_size,
                config.num_actions,
                config.actor_layers,
                config.units,
                config.act,
                config.norm,
                config.actor_dist,
                config.actor_init_std,
                config.actor_min_std,
                config.actor_max_std,
                config.actor_temp,
                outscale=1.0,
                unimix_ratio=config.action_unimix_ratio,
            )
            self.value = networks.MLP_v2(feat_size, 1,  config.units,  config.actor_layers, config.norm)
        elif self.wm_type=='v3':
            self.actor = networks.ActionHead(
                feat_size,
                config.num_actions,
                config.actor_layers,
                config.units,
                config.act,
                config.norm,
                config.actor_dist,
                config.actor_init_std,
                config.actor_min_std,
                config.actor_max_std,
                config.actor_temp,
                outscale=1.0,
                unimix_ratio=config.action_unimix_ratio,
            )
            if config.value_head == "symlog_disc":
                self.value = networks.MLP(
                    feat_size,
                    (255,),
                    config.value_layers,
                    config.units,
                    config.act,
                    config.norm,
                    config.value_head,
                    outscale=0.0,
                    device=config.device,
                )
            else:
                self.value = networks.MLP(
                    feat_size,
                    [],
                    config.value_layers,
                    config.units,
                    config.act,
                    config.norm,
                    config.value_head,
                    outscale=0.0,
                    device=config.device,
                )
        if config.slow_value_target:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor_lr,
            config.ac_opt_eps,
            config.actor_grad_clip,
            **kw,
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.value_lr,
            config.ac_opt_eps,
            config.value_grad_clip,
            **kw,
        )
        if self._config.reward_EMA:
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective=None,
        action=None,
        reward=None,
        imagine=None,
        tape=None,
        repeats=None,
    ):
        objective = objective or self._reward
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp): ##训练world_model的整个64*16的向量都会作为它训练a2c时的start
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon, repeats
                )
                reward = objective(imag_feat, imag_state, imag_action)
                if len(reward.shape)==2:
                    reward=reward.unsqueeze(-1)
                if self.wm_type=='v2':
                    # actor_ent = self.actor.forward_actor(imag_feat,self._config.actor_dist).entropy().unsqueeze(-1)
                    actor_ent = self.actor(imag_feat).entropy()
                    
                elif self.wm_type=='v3':
                    actor_ent = self.actor(imag_feat).entropy()
                    
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled
                # slow is flag to indicate whether slow_target is used for lambda-return
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_state,
                    imag_action,
                    target,
                    actor_ent,
                    state_ent,
                    weights,
                    base,
                )
                metrics.update(mets)

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value_loss,val_mets=self._compute_critic_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                )
                metrics.update(val_mets)
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon, repeats=None):
        dynamics = self._world_model.dynamics
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach() if self._stop_grad_actor else feat
            if self.wm_type=='v2':
                # action=policy.forward_actor(inp,self._config.actor_dist).sample()
                action = policy(inp).sample()
            elif self.wm_type=='v3':
                action = policy(inp).sample()
            succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}
        if repeats:
            raise NotImplemented("repeats is not implemented in this version")

        return feats, states, actions

    def _compute_target(
        self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent
    ):  
        if self._world_model.decoder.cont:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.decoder.cont(inp).mean
            if self.wm_type=='v2':
                discount=discount.unsqueeze(-1)
        else:
            discount = self._config.discount * torch.ones_like(reward)
        if self.wm_type=='v2':
            reward1: TensorHM = reward[1:]
            cont0: TensorHM = discount[:-1]
            cont1: TensorHM = discount[1:]
            # if self._conf.wm_type=='v3':
            #     reward1=reward1.squeeze(-1)
            #     terminal0=terminal0.squeeze(-1)
            #     terminal1=terminal1.squeeze(-1)

            # GAE from https://arxiv.org/abs/1506.02438 eq (16)
            #   advantage_gae[t] = advantage[t] + (discount lambda) advantage[t+1] + (discount lambda)^2 advantage[t+2] + ...

            value_t: TensorJM = self._slow_value.forward(imag_feat)
            if len(value_t.shape)==2:
                value_t=value_t.unsqueeze(-1)
            value0t: TensorHM = value_t[:-1]
            value1t: TensorHM = value_t[1:]
            # TD error=r+\discount*V(s')-V(s)
            advantage = - value0t + reward1 +  cont1 * value1t
            advantage_gae = []
            agae = None
            # GAE的累加
            for adv, continu in zip(reversed(advantage.unbind()), reversed(cont1.unbind())):
                if agae is None:
                    agae = adv
                else:
                    agae = adv + self._config.discount_lambda * continu * agae
                advantage_gae.append(agae)
            advantage_gae.reverse()
            advantage_gae = torch.stack(advantage_gae)
            # Note: if lambda=0, then advantage_gae=advantage, then value_target = advantage + value0t = reward + discount * value1t
            value_target = advantage_gae + value0t

            # When calculating losses, should ignore terminal states, or anything after, so:
            #   reality_weight[i] = (1-terminal[0]) (1-terminal[1]) ... (1-terminal[i])
            # Note this takes care of the case when initial state features[0] is terminal - it will get weighted by (1-terminals[0]).
            # Note that this weights didn't consider discounts
            reality_weight = cont0.log().cumsum(dim=0).exp().detach()
            return value_target, reality_weight, value0t
        
        elif self.wm_type=='v3':
            if self._config.future_entropy and self._config.actor_entropy() > 0: #Usually False
                reward += self._config.actor_entropy() * actor_ent
            if self._config.future_entropy and self._config.actor_state_entropy() > 0:
                reward += self._config.actor_state_entropy() * state_ent
            value = self.value(imag_feat).mode()
            target = tools.lambda_return(
                reward[1:],
                value[:-1],
                discount[1:],
                bootstrap=value[-1],
                lambda_=self._config.discount_lambda,
                axis=0,
            )
            weights = torch.cumprod(
                torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
            ).detach()
            return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_state,
        imag_action,
        target,
        actor_ent,
        state_ent,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
        
        if self.wm_type=='v2':
            # policy_distr = self.actor.forward_actor(inp,self._config.actor_dist)
            policy_distr = self.actor(inp)
            actor_ent = policy_distr.entropy()
              # TODO: we could reuse this from dream()
            advantage_gae=target-base
            if self._config.imag_gradient == 'reinforce':
                action_logprob = policy_distr.log_prob(imag_action)[:-1].unsqueeze(-1)
                loss_policy = - action_logprob * advantage_gae.detach()
            elif self._config.imag_gradient == 'dynamics':
                # loss_policy = - value_target
                loss_policy=- advantage_gae
            else:
                assert False, self._config.imag_gradient

            actor_loss = loss_policy -  self._config.actor_entropy() * actor_ent[:-1][:, :, None]
            actor_loss = (actor_loss * weights).mean()
            
            metrics["policy_entropy"] = to_np(torch.mean(actor_ent))
            # metrics["actor_loss"] = actor_loss.detach().cpu().numpy()
            # metrics["actor_loss"] = actor_loss
        elif self.wm_type=='v3':
            target = torch.stack(target, dim=1)
            policy_distr = self.actor(inp)
            actor_ent = policy_distr.entropy()
            # Q-val for actor is not transformed using symlog
            if self._config.reward_EMA:
                offset, scale = self.reward_ema(target)
                normed_target = (target - offset) / scale
                normed_base = (base - offset) / scale
                adv = normed_target - normed_base
                metrics.update(tools.tensorstats(normed_target, "normed_target"))
                values = self.reward_ema.values
                metrics["EMA_005"] = to_np(values[0])
                metrics["EMA_095"] = to_np(values[1])

            if self._config.imag_gradient == "dynamics":
                actor_target = adv
            elif self._config.imag_gradient == "reinforce":
                actor_target = (
                    policy_distr.log_prob(imag_action)[:-1][:, :, None]
                    * (target - self.value(imag_feat[:-1]).mode()).detach()
                )
            elif self._config.imag_gradient == "both":
                actor_target = (
                    policy_distr.log_prob(imag_action)[:-1][:, :, None]
                    * (target - self.value(imag_feat[:-1]).mode()).detach()
                )
                mix = self._config.imag_gradient_mix()
                actor_target = mix * target + (1 - mix) * actor_target
                metrics["imag_gradient_mix"] = mix
            else:
                raise NotImplementedError(self._config.imag_gradient)
            if not self._config.future_entropy and (self._config.actor_entropy() > 0): ## usually False
                actor_entropy = self._config.actor_entropy() * actor_ent[:-1][:, :, None]
                actor_target += actor_entropy
            if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
                state_entropy = self._config.actor_state_entropy() * state_ent[:-1]
                actor_target += state_entropy
                metrics["actor_state_entropy"] = to_np(torch.mean(state_entropy))
            
            actor_loss = -torch.mean(weights[:-1] * actor_target)
            # metrics["actor_loss"] = to_np(torch.mean(state_entropy))
        return actor_loss, metrics
    
    def _compute_critic_loss(
        self,
        features,
        actions,
        value_target,
        weights,
    ): 
        critic_metric={}
        # Critic loss
        if self.wm_type=='v2':
            value: TensorJM = self.value.forward(features.detach()).unsqueeze(-1)
            value: TensorHM = value[:-1]
            value_loss = 0.5 * torch.square(value_target.detach() - value)
            value_loss = (value_loss * weights).mean()
            critic_metric['policy_value']=value[0].mean().detach().cpu(),  # Value of real states
            critic_metric['policy_value_im']=value.mean().detach().cpu(),  # Value of imagined states
        
        elif self.wm_type=='v3':
            value = self.value(features[:-1].detach())
            value_target = torch.stack(value_target, dim=1)
            # (time, batch, 1), (time, batch, 1) -> (time, batch)
            value_loss = -value.log_prob(value_target.detach())
            slow_target = self._slow_value(features[:-1].detach())
            if self._config.slow_value_target:
                value_loss = value_loss - value.log_prob(
                    slow_target.mode().detach()
                )
            if self._config.value_decay:
                value_loss += self._config.value_decay * value.mode()
            # (time, batch, 1), (time, batch, 1) -> (1,)
            value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])


            critic_metric.update(tools.tensorstats(value.mode().detach(), "policy_value"))
            critic_metric.update(tools.tensorstats(value_target, "value_target"))
            # critic_metric.update(tensorstats(rewards, "imag_reward"))
            if self._config.actor_dist in ["onehot"]:
                critic_metric.update(
                    tools.tensorstats(
                        torch.argmax(actions, dim=-1).float(), "image_actions"
                    )
                )
            else:
                critic_metric.update(tools.tensorstats(actions, "imag_actions"))
            
        # with RequiresGrad(self):
            # metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            # metrics.update(self._value_opt(value_loss, self.value.parameters()))
        
        # critic_metric["value_loss"] =value_loss.detach().cpu().numpy()
        # critic_metric["value_loss"] = value_loss.detach().cpu().numpy()
        # tensors = dict(value=value.mode(),
        #                 value_weight=weights.detach(),
        #                 )
        
        return value_loss,critic_metric

    def _update_slow_target(self):
        if self._config.slow_value_target:
            if self._updates % self._config.slow_target_update == 0:
                mix = self._config.slow_target_fraction
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
