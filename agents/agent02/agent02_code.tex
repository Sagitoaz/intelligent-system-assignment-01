
"""
AGENT02: Reinforcement-Learning-Based Agent vs Rule-Based Agent
================================================================

Course: Intelligent System Development

Run:
    python agent02_experiment.py

Dependencies:
    numpy
    matplotlib

Outputs:
    agent02_results/
        baseline_episode_rewards.csv
        baseline_evaluation.csv
        baseline_evaluation_episodes.csv
        learning_rate_comparison.csv
        exploration_comparison.csv
        multi_seed_summary.csv
        q_tables.txt
        episode_rewards.png
        learning_rate_comparison.png
        exploration_comparison.png

Agents:
    1. RL-based agent: tabular Q-learning
    2. Rule-based agent: fixed if-else policy

The RL agent is trained against a fixed rule-based opponent.
Evaluation freezes the learned Q-table and disables exploration.
"""

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass
class Config:
    episodes: int = 5000
    steps_per_episode: int = 20

    alpha: float = 0.10
    gamma: float = 0.90

    epsilon_start: float = 0.20
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.995

    price: float = 10.0
    unit_cost: float = 5.0
    discount_rate: float = 0.20

    low_demand_customers: int = 10
    high_demand_customers: int = 20

    seed: int = 42
    evaluation_episodes: int = 500

    output_dir: str = "agent02_results"


State = tuple[int, int]


# ============================================================
# 2. E-COMMERCE ENVIRONMENT
# ============================================================

class ECommerceEnvironment:
    """
    Two agents compete in a simplified e-commerce market.

    State:
        (demand, competitor_previous_action)

        demand:
            0 = low demand
            1 = high demand

        competitor_previous_action:
            0 = no discount
            1 = discount

    Action:
        0 = no discount
        1 = discount

    Reward:
        Profit = sales * profit_per_sale
    """

    def __init__(self, config, seed=None):
        self.config = config
        actual_seed = config.seed if seed is None else seed
        self.rng = np.random.default_rng(actual_seed)

        self.demand = 0
        self.previous_actions = [0, 0]

    def reset(self):
        self.demand = int(self.rng.integers(0, 2))
        self.previous_actions = [0, 0]
        return self.get_states()

    def get_states(self):
        return [
            (self.demand, self.previous_actions[1]),
            (self.demand, self.previous_actions[0])
        ]

    def step(self, actions):
        if len(actions) != 2:
            raise ValueError("Exactly two actions are required.")

        if any(action not in (0, 1) for action in actions):
            raise ValueError("Actions must be 0 or 1.")

        rewards = []
        sales_list = []
        conversion_list = []

        if self.demand == 0:
            customers = self.config.low_demand_customers
        else:
            customers = self.config.high_demand_customers

        # Both actions are selected before this function is called.
        # Neither agent observes the other's current action beforehand.
        for i in range(2):
            own_action = actions[i]
            competitor_action = actions[1 - i]

            # Illustrative conversion probability.
            conversion = (
                0.30
                + 0.25 * self.demand
                + 0.15 * own_action
                - 0.10 * competitor_action
            )

            conversion = float(np.clip(
                conversion, 0.05, 0.95
            ))

            # Stochastic number of sales.
            sales = int(self.rng.binomial(
                customers, conversion
            ))

            # Apply a discount only when action = 1.
            effective_price = self.config.price * (
                1.0 - self.config.discount_rate * own_action
            )

            profit_per_sale = (
                effective_price - self.config.unit_cost
            )

            profit = sales * profit_per_sale

            rewards.append(float(profit))
            sales_list.append(sales)
            conversion_list.append(conversion)

        # Store current actions for the next observation.
        self.previous_actions = list(actions)

        # Demand evolves independently of current actions.
        self.demand = int(self.rng.integers(0, 2))

        next_states = self.get_states()

        info = {
            "sales": sales_list,
            "conversion": conversion_list,
            "customers": customers
        }

        return next_states, rewards, info


# ============================================================
# 3. REINFORCEMENT-LEARNING-BASED AGENT
# ============================================================

class QLearningAgent:
    """
    Tabular Q-learning agent.

    Q[demand, competitor_previous_action, action]

    There are:
        2 demand states
        2 competitor-action states
        2 possible actions

    Total Q-values = 2 * 2 * 2 = 8
    """

    def __init__(
        self,
        alpha=0.10,
        gamma=0.90,
        epsilon=0.20,
        seed=123
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.rng = np.random.default_rng(seed)

        self.q = np.zeros((2, 2, 2), dtype=float)

    def act(self, state, explore=True):
        demand, competitor_previous_action = state

        # Exploration.
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, 2))

        # Exploitation.
        values = self.q[demand, competitor_previous_action]

        # Random tie-breaking when action values are equal.
        best_actions = np.flatnonzero(
            np.isclose(values, np.max(values))
        )

        return int(self.rng.choice(best_actions))

    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        terminal=False
    ):
        demand, competitor_previous_action = state
        next_demand, next_competitor_action = next_state

        old_value = self.q[
            demand, competitor_previous_action, action
        ]

        if terminal:
            target = reward
        else:
            best_future_value = np.max(
                self.q[next_demand, next_competitor_action]
            )

            target = (
                reward + self.gamma * best_future_value
            )

        # Q-learning update.
        self.q[
            demand, competitor_previous_action, action
        ] = old_value + self.alpha * (
            target - old_value
        )

    def greedy_policy(self):
        policy = {}

        for demand in range(2):
            for competitor_action in range(2):
                state = (demand, competitor_action)

                policy[state] = int(
                    np.argmax(self.q[demand, competitor_action])
                )

        return policy


# ============================================================
# 4. RULE-BASED AGENT
# ============================================================

class RuleBasedAgent:
    """
    Fixed policy:

    Rule 1:
        Low demand -> no discount.

    Rule 2:
        High demand and competitor previously discounted
        -> offer a discount.

    Rule 3:
        High demand and competitor did not previously discount
        -> no discount.

    This agent does not learn from rewards.
    """

    def act(self, state):
        demand, competitor_previous_action = state

        if demand == 0:
            return 0

        elif demand == 1 and competitor_previous_action == 1:
            return 1

        else:
            return 0


# ============================================================
# 5. TRAIN THE RL AGENT
# ============================================================

def train_rl_against_rule(
    config,
    seed,
    alpha=None,
    epsilon_start=None,
    epsilon_min=None,
    epsilon_decay=None
):
    """
    Train one RL agent against a fixed rule-based opponent.

    The opponent's policy remains unchanged during training.
    """

    alpha = config.alpha if alpha is None else alpha

    epsilon_start = (
        config.epsilon_start
        if epsilon_start is None
        else epsilon_start
    )

    epsilon_min = (
        config.epsilon_min
        if epsilon_min is None
        else epsilon_min
    )

    epsilon_decay = (
        config.epsilon_decay
        if epsilon_decay is None
        else epsilon_decay
    )

    env = ECommerceEnvironment(config, seed=seed)

    rl_agent = QLearningAgent(
        alpha=alpha,
        gamma=config.gamma,
        epsilon=epsilon_start,
        seed=seed + 10_000
    )

    rule_agent = RuleBasedAgent()

    episode_results = []

    for episode in range(config.episodes):

        # Decrease exploration as training progresses.
        rl_agent.epsilon = max(
            epsilon_min,
            epsilon_start * epsilon_decay ** episode
        )

        states = env.reset()

        episode_rewards = [0.0, 0.0]

        for step in range(config.steps_per_episode):

            # Both agents choose actions before the environment changes.
            rl_action = rl_agent.act(
                states[0], explore=True
            )

            rule_action = rule_agent.act(states[1])

            actions = [rl_action, rule_action]

            next_states, rewards, info = env.step(actions)

            terminal = (
                step == config.steps_per_episode - 1
            )

            rl_agent.learn(
                state=states[0],
                action=rl_action,
                reward=rewards[0],
                next_state=next_states[0],
                terminal=terminal
            )

            episode_rewards[0] += rewards[0]
            episode_rewards[1] += rewards[1]

            states = next_states

        episode_results.append({
            "episode": episode + 1,
            "rl_episode_reward": episode_rewards[0],
            "rule_episode_reward": episode_rewards[1],
            "rl_mean_reward_per_step": (
                episode_rewards[0]
                / config.steps_per_episode
            ),
            "rule_mean_reward_per_step": (
                episode_rewards[1]
                / config.steps_per_episode
            ),
            "epsilon": rl_agent.epsilon,
            "alpha": alpha
        })

    return rl_agent, episode_results


# ============================================================
# 6. SEPARATE POLICY EVALUATION
# ============================================================

def evaluate_pair(
    config,
    rl_agent,
    episodes=None,
    seed=900_000
):
    """
    Evaluate a frozen RL policy against the fixed rule-based agent.

    Important:
        - No Q-table updates occur.
        - RL exploration is disabled.
        - Evaluation uses a separate environment random seed.
    """

    if episodes is None:
        episodes = config.evaluation_episodes

    env = ECommerceEnvironment(config, seed=seed)

    rule_agent = RuleBasedAgent()

    total_rewards = np.zeros(2, dtype=float)
    action_counts = np.zeros((2, 2), dtype=int)

    episode_results = []

    for episode in range(episodes):

        states = env.reset()
        rewards_this_episode = [0.0, 0.0]

        for _ in range(config.steps_per_episode):

            # Greedy RL policy: no exploration.
            rl_action = rl_agent.act(
                states[0], explore=False
            )

            rule_action = rule_agent.act(states[1])

            actions = [rl_action, rule_action]

            next_states, rewards, info = env.step(actions)

            for i in range(2):
                total_rewards[i] += rewards[i]
                rewards_this_episode[i] += rewards[i]
                action_counts[i, actions[i]] += 1

            states = next_states

        episode_results.append({
            "episode": episode + 1,
            "rl_reward": rewards_this_episode[0],
            "rule_reward": rewards_this_episode[1],
            "rl_mean_reward_per_step": (
                rewards_this_episode[0]
                / config.steps_per_episode
            ),
            "rule_mean_reward_per_step": (
                rewards_this_episode[1]
                / config.steps_per_episode
            )
        })

    total_steps = episodes * config.steps_per_episode

    summary = []

    for i, name in enumerate(["RL", "Rule-based"]):
        summary.append({
            "agent": name,
            "total_reward": total_rewards[i],
            "mean_reward_per_step": (
                total_rewards[i] / total_steps
            ),
            "no_discount_actions": int(action_counts[i, 0]),
            "discount_actions": int(action_counts[i, 1]),
            "discount_action_rate": (
                action_counts[i, 1] / total_steps
            )
        })

    return summary, episode_results


# ============================================================
# 7. CSV AND Q-TABLE OUTPUT
# ============================================================

def write_csv(filepath, rows):
    if not rows:
        return

    with open(
        filepath, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def save_q_table(filepath, agent):
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(
            "Q-table: Q[demand, competitor_previous_action, action]\n"
        )
        file.write(
            "Action 0 = no discount; Action 1 = discount\n\n"
        )

        for demand in range(2):
            for competitor_action in range(2):

                q0 = agent.q[demand, competitor_action, 0]
                q1 = agent.q[demand, competitor_action, 1]

                best_action = int(
                    np.argmax(
                        agent.q[demand, competitor_action]
                    )
                )

                file.write(
                    f"State=({demand}, {competitor_action}) | "
                    f"Q0={q0:.6f} | Q1={q1:.6f} | "
                    f"Greedy action={best_action}\n"
                )


# ============================================================
# 8. PLOTS
# ============================================================

def moving_average(values, window=100):
    values = np.asarray(values, dtype=float)

    if len(values) < window:
        return values

    kernel = np.ones(window) / window

    return np.convolve(
        values, kernel, mode="valid"
    )


def plot_training_rewards(rows, filepath):
    episodes = [
        row["episode"] for row in rows
    ]

    rl_rewards = [
        row["rl_mean_reward_per_step"] for row in rows
    ]

    rule_rewards = [
        row["rule_mean_reward_per_step"] for row in rows
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        episodes, rl_rewards,
        alpha=0.25, label="RL training reward"
    )

    plt.plot(
        episodes, rule_rewards,
        alpha=0.25, label="Rule-based reward"
    )

    window = min(100, max(10, len(rows) // 20))

    if len(rows) >= window:
        plt.plot(
            episodes[window - 1:],
            moving_average(rl_rewards, window),
            linewidth=2,
            label=f"RL moving average ({window})"
        )

        plt.plot(
            episodes[window - 1:],
            moving_average(rule_rewards, window),
            linewidth=2,
            label=f"Rule moving average ({window})"
        )

    plt.xlabel("Training episode")
    plt.ylabel("Mean reward per step")
    plt.title("RL Training Against a Rule-Based Agent")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close()


def plot_bar_comparison(
    rows, label_key, value_key, filepath, title
):
    labels = [str(row[label_key]) for row in rows]
    values = [row[value_key] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)

    plt.xlabel(label_key.replace("_", " ").title())
    plt.ylabel(value_key.replace("_", " ").title())
    plt.title(title)

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=160)
    plt.close()


# ============================================================
# 9. BASELINE EXPERIMENT
# ============================================================

def run_baseline(config, output_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: BASELINE")
    print("=" * 60)

    agent, training_rows = train_rl_against_rule(
        config, seed=config.seed
    )

    write_csv(
        output_dir / "baseline_episode_rewards.csv",
        training_rows
    )

    save_q_table(
        output_dir / "q_tables.txt", agent
    )

    plot_training_rewards(
        training_rows,
        output_dir / "episode_rewards.png"
    )

    summary, eval_rows = evaluate_pair(
        config,
        agent,
        episodes=config.evaluation_episodes,
        seed=config.seed + 500_000
    )

    write_csv(
        output_dir / "baseline_evaluation.csv",
        summary
    )

    write_csv(
        output_dir / "baseline_evaluation_episodes.csv",
        eval_rows
    )

    print("\nSeparate evaluation results:")

    for row in summary:
        print(
            f"{row['agent']:12s} | "
            f"mean reward/step = "
            f"{row['mean_reward_per_step']:.4f} | "
            f"discount rate = "
            f"{row['discount_action_rate']:.3f}"
        )

    print("\nLearned greedy policy:")

    for state, action in agent.greedy_policy().items():
        action_name = (
            "DISCOUNT" if action == 1
            else "NO DISCOUNT"
        )
        print(f"State {state}: {action_name}")

    return agent


# ============================================================
# 10. LEARNING-RATE EXPERIMENT
# ============================================================

def run_learning_rate_experiment(config, output_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: LEARNING RATE")
    print("=" * 60)

    results = []

    for alpha in [0.01, 0.10, 0.50]:

        agent, _ = train_rl_against_rule(
            config,
            seed=config.seed + 1000,
            alpha=alpha
        )

        summary, _ = evaluate_pair(
            config,
            agent,
            episodes=config.evaluation_episodes,
            seed=config.seed + 600_000
        )

        rl_result = next(
            row for row in summary
            if row["agent"] == "RL"
        )

        results.append({
            "alpha": alpha,
            "rl_mean_reward_per_step":
                rl_result["mean_reward_per_step"],
            "rl_discount_action_rate":
                rl_result["discount_action_rate"]
        })

        print(
            f"alpha={alpha:.2f} | "
            f"evaluation reward/step="
            f"{rl_result['mean_reward_per_step']:.4f}"
        )

    write_csv(
        output_dir / "learning_rate_comparison.csv",
        results
    )

    plot_bar_comparison(
        results,
        "alpha",
        "rl_mean_reward_per_step",
        output_dir / "learning_rate_comparison.png",
        "Effect of Learning Rate"
    )

    return results


# ============================================================
# 11. EXPLORATION EXPERIMENT
# ============================================================

def run_exploration_experiment(config, output_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: EXPLORATION")
    print("=" * 60)

    settings = [
        ("low", 0.05, 0.02, 0.999),
        ("baseline", 0.20, 0.02, 0.995),
        ("high", 0.50, 0.05, 0.995)
    ]

    results = []

    for name, eps_start, eps_min, eps_decay in settings:

        agent, _ = train_rl_against_rule(
            config,
            seed=config.seed + 2000,
            epsilon_start=eps_start,
            epsilon_min=eps_min,
            epsilon_decay=eps_decay
        )

        summary, _ = evaluate_pair(
            config,
            agent,
            episodes=config.evaluation_episodes,
            seed=config.seed + 700_000
        )

        rl_result = next(
            row for row in summary
            if row["agent"] == "RL"
        )

        results.append({
            "exploration_setting": name,
            "epsilon_start": eps_start,
            "epsilon_min": eps_min,
            "epsilon_decay": eps_decay,
            "rl_mean_reward_per_step":
                rl_result["mean_reward_per_step"],
            "rl_discount_action_rate":
                rl_result["discount_action_rate"]
        })

        print(
            f"{name:8s} | "
            f"evaluation reward/step="
            f"{rl_result['mean_reward_per_step']:.4f}"
        )

    write_csv(
        output_dir / "exploration_comparison.csv",
        results
    )

    plot_bar_comparison(
        results,
        "exploration_setting",
        "rl_mean_reward_per_step",
        output_dir / "exploration_comparison.png",
        "Effect of Exploration Schedule"
    )

    return results


# ============================================================
# 12. MULTIPLE-SEED EXPERIMENT
# ============================================================

def run_multi_seed_experiment(config, output_dir):
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: MULTIPLE RANDOM SEEDS")
    print("=" * 60)

    seeds = [11, 22, 33, 44, 55]
    results = []

    for seed in seeds:

        agent, _ = train_rl_against_rule(
            config, seed=seed
        )

        summary, _ = evaluate_pair(
            config,
            agent,
            episodes=config.evaluation_episodes,
            seed=seed + 800_000
        )

        for row in summary:
            results.append({
                "seed": seed,
                "agent": row["agent"],
                "mean_reward_per_step":
                    row["mean_reward_per_step"],
                "discount_action_rate":
                    row["discount_action_rate"]
            })

    write_csv(
        output_dir / "multi_seed_summary.csv",
        results
    )

    print("\nMean and standard deviation across seeds:")

    for name in ["RL", "Rule-based"]:

        values = [
            row["mean_reward_per_step"]
            for row in results
            if row["agent"] == name
        ]

        print(
            f"{name:12s} | "
            f"mean={np.mean(values):.4f} | "
            f"std={np.std(values, ddof=1):.4f} | "
            f"n={len(values)}"
        )

    return results


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

def main():
    config = Config()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AGENT02: RL AGENT VS RULE-BASED AGENT")
    print("=" * 60)

    print(f"Episodes: {config.episodes}")
    print(f"Steps per episode: {config.steps_per_episode}")
    print(f"Evaluation episodes: {config.evaluation_episodes}")
    print(f"Random seed: {config.seed}")
    print(f"Output directory: {output_dir.resolve()}")

    run_baseline(config, output_dir)

    run_learning_rate_experiment(
        config, output_dir
    )

    run_exploration_experiment(
        config, output_dir
    )

    run_multi_seed_experiment(
        config, output_dir
    )

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)

    print(f"Results saved to: {output_dir.resolve()}")

    print("\nUse the CSV files and PNG figures in the AGENT02 report.")


if __name__ == "__main__":
    main()
