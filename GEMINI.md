# GorgeChase Project Context & Guidelines

## Project Overview
GorgeChase is a reinforcement learning project aimed at training an agent for the "Gorge Chase" (峡谷追猎) environment. The primary objective is to navigate a 128x128 map, collect treasures and buffs, explore the map, and avoid being caught by monsters.

## Core Algorithm: `code/agent_ppo`
The current primary algorithm is a PPO-based implementation with a Mamba-augmented neural network.

### Architecture
- **Model (`code/agent_ppo/model/model.py`)**:
    - **Local View**: 21x21x7 crop around the hero, processed via ResBlocks.
    - **Global View**: 32x32x6 compressed map, processed via strided ResBlocks and Adaptive Average Pooling.
    - **Scalar MLP**: Processes hero stats, monster info, and organ (treasure/buff) stats.
    - **Temporal Memory**: A `HiddenStateMambaCell` (State Space Model) for processing sequence data and maintaining hidden state.
    - **Heads**:
        - Policy: 16-dim (8 Move actions, 8 Flash actions).
        - Value: 1-dim (Critic).
        - Auxiliary: Monster position (2-dim) and distance bucket (6-dim).
- **Algorithm (`code/agent_ppo/algorithm/algorithm.py`)**:
    - PPO with clipped surrogate objective and value clipping.
    - Entropy regularization for exploration.
    - Masked softmax for legal actions.
    - Auxiliary losses for monster tracking to improve feature representation.
- **Feature Processing (`code/agent_ppo/feature/preprocessor.py`)**:
    - **Global Memory**: Maintains persistent maps for passability, exploration, and item availability.
    - **BFS Distance Mapping**: Computes real path distances for both local/global features and reward shaping.
    - **Reward Shaping**:
        - Positive: Step reward (0.08), Exploration (per cell), Explore approach (dense), Treasure collection, Buff collection, Buff maintenance (dense), Flash usage efficiency, Survival milestones.
        - Negative: Proximity to monsters (BFS-based, threshold=4.0, slope=0.03), still/loop penalties, being caught.
        - Strategic Trade-off: Resource approach rewards decay linearly when monsters are near (5.0-10.0 BFS).
        - Strategic Penalty: Heavy penalty (-2.0) for non-dangerous flash usage to discourage waste.

### Key Configurations (`code/agent_ppo/conf/conf.py`)
- `MAP_SIZE = 128`
- `MAMBA_HIDDEN_DIM = 128`
- `ACTION_NUM = 16` (Move: 0-7, Flash: 8-15)
- `GAMMA = 0.99`, `LAMDA = 0.95` (GAE hyperparameters)
- Runtime flash guard and resource override are currently disabled by default.

## Development Workflow
- **Training**: Managed via `code/agent_ppo/workflow/train_workflow.py`. It uses `EpisodeRunner` to interact with the environment and collect samples.
- **Evaluation**: Use `code/agent_ppo/agent.py`'s `exploit` method for deterministic inference.
- **Model Management**: Models are saved/loaded as `.pkl` files (state dicts). "latest" ID is commonly used for the most recent checkpoint.

## Engineering Standards
- **Framework**: Built on the `kaiwudrl` interface.
- **Device**: Torch-based; typically runs on GPU but handles CPU fallbacks.
- **Logging**: Use the provided `logger` and `monitor` for all metrics and debug info.
- **Naming**: Follow existing snake_case for functions/variables and PascalCase for classes.
- **Comments**: Maintain bilingual (CN/EN) comments where appropriate, as seen in the workflow files.

## Guidelines for Changes
1. **Model Changes**: If modifying `Model`, ensure `FEATURE_SPLIT_SHAPE` in `Config` is updated accordingly.
2. **Feature Changes**: Preprocessor changes often require corresponding updates in `Model._split_obs` and `SampleData` definition.
3. **Reward Tuning**: Most rewards are BFS-based. Ensure BFS sentinels are handled correctly to avoid NaN/Inf rewards.
4. **Mamba State**: The `hidden_state` must be correctly reset at the start of each episode and passed through the `observation_process`.
