# World_model_assemble
A hub for various model-based RL agents
Pytorch implementation of [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)and [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1). More relevent world models will be added later.

## Comparision betwwen dreamer_v2 and dreamer_v3
In [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193), the author lists some main difference bewteen v2 and v3. We summarize our current realization situation here.

|                           | DreamerV2   | DreamerV3     |
|-----------------------------------|-------------|---------------|
| Symlog predictions                | ❌           | ✅             |
| World model regularizer           | ❌           | Free_bits            |
| Policy regularizer                | entropy-only           | entropy+scaling             |
| Activation                        | ELU           | SILU             |
| LayerNorm                         | mlp-only           | mlp+cnn            |
| MLP-layer                         | 4           | 2             |
| RSSM-units                        | 1024           | 512             |
| Critic EMA regularizer            | ❌           | ✅             |
| Replay buffer                     | ✅           | ✅             |
| Hyperparameters                   | env-specific        | general           |
| Exploration - Plan2Explore            | ❌           | ✅             |



## Instructions


Get dependencies with python 3.9:
```
pip install -r requirements.txt
```
Run training on dreamer_v2(Atari Games):
```
python3 dreamer.py --configs atari100k --task atari_pong --logdir ./logdir/atari_pong
```
Run training on dreamer_v3(DMC Vision)
```
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```
Monitor results:
```
tensorboard --logdir ./logdir
```

## Benchmarks
So far, the following benchmarks can be used for testing.
| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [Atari 100k](https://github.com/openai/atari-py) | Image | Discrete |400K| 26 Atari games. |


## Results
<p align="center">
    <img width="20%" src="https://github.com/whatevermybaby/World_model_assemble/blob/main/dreamer_v2_pong.gif">
    <img width="20%" src="https://github.com/whatevermybaby/World_model_assemble/blob/main/dreamer_v3_pong.gif">
    <img width="40%" src="https://github.com/whatevermybaby/World_model_assemble/blob/main/dreamer_v2_walker.gif?raw=true">
    <img width="40%" src="https://github.com/whatevermybaby/World_model_assemble/blob/main/dreamer_v3_walker.gif">
</p>


## Acknowledgments
This code is heavily inspired by the following works:
- danijar's Dreamer-v3 jax implementation: https://github.com/danijar/dreamerv3
- danijar's Dreamer-v2 tensorflow implementation: https://github.com/danijar/dreamerv2
- NM512's Dreamer-v3 pytorch implementation: https://github.com/NM512/dreamerv3-torch
- jurgisp's Pydreamer implementation: https://github.com/jurgisp/pydreamer
