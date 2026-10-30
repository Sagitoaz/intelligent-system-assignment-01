"""Lõi mô phỏng cho hai bài tập agent: Agent 01 (mèo đuổi chuột) và Agent 02 (hai người bán).

Chỉ dùng thư viện chuẩn của Python để chạy được trên mọi máy mà không cần cài thêm gói.
"""
from __future__ import annotations

import random
from collections import deque

ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # lên, phải, xuống, trái
ACTION_NAMES = ("lên", "phải", "xuống", "trái")


# --------------------------------------------------------------------------
# Agent 01: bản đồ, môi trường mèo đuổi chuột và Q-learning
# --------------------------------------------------------------------------
MAP_SPECS = {
    "garden": dict(title="Khu vườn (18×12)", w=18, h=12, blocks=10, seed=3, smell=18),
    "maze": dict(title="Mê cung (26×16)", w=26, h=16, blocks=30, seed=11, smell=24),
    "city": dict(title="Thành phố (34×20)", w=34, h=20, blocks=55, seed=7, smell=60),
}


def _generate_grid(w, h, blocks, seed):
    rng = random.Random(seed)
    grid = [["#" if r in (0, h - 1) or c in (0, w - 1) else "." for c in range(w)] for r in range(h)]
    for _ in range(blocks):
        bh, bw = (rng.randint(1, 2), rng.randint(2, 6)) if rng.random() < 0.5 else (rng.randint(2, 5), rng.randint(1, 2))
        r0, c0 = rng.randint(1, h - 2), rng.randint(1, w - 2)
        for r in range(r0, min(h - 1, r0 + bh)):
            for c in range(c0, min(w - 1, c0 + bw)):
                grid[r][c] = "#"
    # giữ lại vùng thông nhau lớn nhất để mèo và chuột luôn có đường gặp nhau
    best, seen = [], set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] == "." and (r, c) not in seen:
                comp, queue = [], deque([(r, c)])
                seen.add((r, c))
                while queue:
                    cr, cc = queue.popleft()
                    comp.append((cr, cc))
                    for dr, dc in ACTIONS:
                        n = (cr + dr, cc + dc)
                        if grid[n[0]][n[1]] == "." and n not in seen:
                            seen.add(n)
                            queue.append(n)
                if len(comp) > len(best):
                    best = comp
    keep = set(best)
    return ["".join("." if (r, c) in keep else "#" for c in range(w)) for r in range(h)]


class ChaseMap:
    """Bản đồ lưới với bảng khoảng cách BFS giữa mọi cặp ô (tính một lần)."""

    def __init__(self, key):
        spec = MAP_SPECS[key]
        self.key, self.title, self.smell = key, spec["title"], spec["smell"]
        self.grid = _generate_grid(spec["w"], spec["h"], spec["blocks"], spec["seed"])
        self.h, self.w = len(self.grid), len(self.grid[0])
        self.cells = [(r, c) for r in range(self.h) for c in range(self.w) if self.grid[r][c] == "."]
        self.index = {cell: i for i, cell in enumerate(self.cells)}
        self.nbr = []
        for r, c in self.cells:
            row = []
            for dr, dc in ACTIONS:
                n = (r + dr, c + dc)
                row.append(self.index.get(n, -1))
            self.nbr.append(row)
        self.dist = [self._bfs(i) for i in range(len(self.cells))]

    def _bfs(self, src):
        dist = [-1] * len(self.cells)
        dist[src] = 0
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for n in self.nbr[cur]:
                if n >= 0 and dist[n] < 0:
                    dist[n] = dist[cur] + 1
                    queue.append(n)
        return dist

    def public(self):
        return dict(key=self.key, title=self.title, w=self.w, h=self.h, grid=self.grid, open_cells=len(self.cells))


class ChaseEnv:
    """Mèo (agent) đuổi chuột. Chuột chỉ đi 3 trong 4 nhịp và chạy trốn mèo theo khoảng cách BFS."""

    def __init__(self, cmap, rng, max_steps=250, mouse_speed=0.75, flee_prob=0.85):
        self.m, self.rng = cmap, rng
        self.max_steps, self.mouse_speed, self.flee_prob = max_steps, mouse_speed, flee_prob

    def reset(self):
        n = len(self.m.cells)
        self.cat = self.rng.randrange(n)
        # chuột xuất phát cách mèo ít nhất 1/3 chiều dài lớn nhất để mỗi ván đều phải đuổi
        far = [i for i in range(n) if self.m.dist[self.cat][i] >= max(6, max(self.m.dist[self.cat]) // 3)]
        self.mouse = self.rng.choice(far or list(range(n)))
        self.steps, self.last_action = 0, 4
        return self.state()

    def state(self):
        """Quan sát của mèo: tường kề, khoảng cách thô, và (nếu chuột trong tầm đánh hơi của bản đồ) hướng đánh hơi."""
        m = self.m
        (cr, cc), (mr, mc) = m.cells[self.cat], m.cells[self.mouse]
        sign = lambda v: (v > 0) - (v < 0) + 1
        d = m.dist[self.cat][self.mouse]
        bucket = 0 if d <= 1 else 1 if d <= 3 else 2 if d <= 7 else 3 if d <= m.smell else 4
        walls = sum((m.nbr[self.cat][a] < 0) << a for a in range(4))
        if d <= m.smell:
            code = 9 + bfs_action(m, self.cat, self.mouse)  # hướng đánh hơi: 9..12
        else:
            code = sign(mr - cr) * 3 + sign(mc - cc)  # chỉ biết chuột nằm ở phía nào: 0..8
        return (code * 5 + bucket) * 16 + walls

    def _mouse_move(self, forced=None):
        m = self.m
        if forced is not None:
            n = m.nbr[self.mouse][forced]
            if n >= 0:
                self.mouse = n
            return
        if self.rng.random() > self.mouse_speed:
            return
        options = [n for n in m.nbr[self.mouse] if n >= 0] + [self.mouse]
        if self.rng.random() < self.flee_prob:
            best = max(m.dist[self.cat][n] for n in options)
            options = [n for n in options if m.dist[self.cat][n] == best]
        self.mouse = self.rng.choice(options)

    def step(self, action, mouse_forced=None):
        m = self.m
        before = m.dist[self.cat][self.mouse]
        n = m.nbr[self.cat][action]
        bump = n < 0
        if not bump:
            self.cat = n
        self.last_action = action
        caught = self.cat == self.mouse
        if not caught:
            self._mouse_move(mouse_forced)
            caught = self.cat == self.mouse
        self.steps += 1
        after = m.dist[self.cat][self.mouse]
        reward = -0.1 + 0.2 * (before - after) - (0.3 if bump else 0.0)
        if caught:
            reward += 20.0
        return self.state(), reward, caught, (not caught and self.steps >= self.max_steps)


def greedy_index(values, rng):
    best = max(values)
    return rng.choice([a for a, v in enumerate(values) if v == best])


def bfs_action(cmap, cat, mouse, rng=None):
    """Chính sách oracle: đi vào ô láng giềng gần chuột nhất theo BFS (biết toàn bộ bản đồ)."""
    best, choices = None, []
    for a, n in enumerate(cmap.nbr[cat]):
        if n < 0:
            continue
        d = cmap.dist[n][mouse]
        if best is None or d < best:
            best, choices = d, [a]
        elif d == best:
            choices.append(a)
    return choices[0] if rng is None else rng.choice(choices)


class CatQAgent:
    """Q-learning dạng bảng; trạng thái chỉ là quan sát cục bộ (hướng tương đối, khoảng cách thô, tường kề)."""

    def __init__(self, alpha=0.2, gamma=0.95, seed=0):
        self.alpha, self.gamma = alpha, gamma
        self.q = {}
        self.rng = random.Random(seed)

    def values(self, state):
        return self.q.setdefault(state, [0.0] * 4)

    def act(self, state, epsilon=0.0):
        if self.rng.random() < epsilon:
            return self.rng.randrange(4)
        return greedy_index(self.values(state), self.rng)

    def learn(self, state, action, reward, nxt, terminal):
        target = reward if terminal else reward + self.gamma * max(self.values(nxt))
        v = self.values(state)
        v[action] += self.alpha * (target - v[action])


def train_cat(cmap, episodes=6000, alpha=0.2, gamma=0.95, seed=0):
    agent = CatQAgent(alpha, gamma, seed)
    env = ChaseEnv(cmap, random.Random(seed + 1))
    history = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - 0.95 * ep / (0.6 * episodes))
        state, total = env.reset(), 0.0
        while True:
            a = agent.act(state, eps)
            nxt, r, caught, timeout = env.step(a)
            agent.learn(state, a, r, nxt, caught)
            state, total = nxt, total + r
            if caught or timeout:
                break
        history.append(dict(episode=ep + 1, caught=int(caught), steps=env.steps, total_return=round(total, 3), epsilon=round(eps, 3)))
    return agent, history


def policy_action(kind, cmap, env, agent, rng):
    if kind == "bfs":
        return bfs_action(cmap, env.cat, env.mouse, rng)
    if kind == "random":
        return rng.randrange(4)
    return agent.act(env.state(), 0.05)


def play_episode(kind, cmap, agent, seed):
    rng = random.Random(seed)
    env = ChaseEnv(cmap, rng)
    env.reset()
    frames = [[cmap.cells[env.cat], cmap.cells[env.mouse]]]
    while True:
        a = policy_action(kind, cmap, env, agent, rng)
        _, _, caught, timeout = env.step(a)
        frames.append([cmap.cells[env.cat], cmap.cells[env.mouse]])
        if caught or timeout:
            return dict(frames=frames, caught=caught, steps=env.steps)


def evaluate_cat(cmap, agent, episodes=200, base_seed=900000):
    rows = []
    for kind in ("random", "q", "bfs"):
        results = [play_episode(kind, cmap, agent, base_seed + i) for i in range(episodes)]
        won = [r for r in results if r["caught"]]
        rows.append(dict(map=cmap.key, policy=kind, episodes=episodes,
                         catch_rate=round(len(won) / episodes, 4),
                         mean_steps_when_caught=round(sum(r["steps"] for r in won) / len(won), 2) if won else None,
                         mean_steps_all=round(sum(r["steps"] for r in results) / episodes, 2)))
    return rows


def interactive_step(kind, cmap, agent, cat, mouse, mouse_dir, seed):
    """Người chơi điều khiển chuột; mèo đi theo chính sách đã chọn. Máy chủ không lưu trạng thái."""
    rng = random.Random(seed)
    env = ChaseEnv(cmap, rng)
    env.cat, env.mouse, env.steps, env.last_action = cat, mouse, 0, 4
    n = cmap.nbr[mouse][mouse_dir] if 0 <= mouse_dir < 4 else -1
    if n >= 0:
        env.mouse = n
    if env.cat == env.mouse:
        return dict(cat=cmap.cells[env.cat], mouse=cmap.cells[env.mouse], caught=True)
    a = policy_action(kind, cmap, env, agent, rng)
    nc = cmap.nbr[env.cat][a]
    if nc >= 0:
        env.cat = nc
    return dict(cat=cmap.cells[env.cat], mouse=cmap.cells[env.mouse], caught=env.cat == env.mouse)


# --------------------------------------------------------------------------
# Agent 02: hai người bán cạnh tranh giá (đúng mô hình trong agent02_experiment.py)
# --------------------------------------------------------------------------
class SellerConfig:
    price, unit_cost, discount_rate = 10.0, 5.0, 0.20
    low_customers, high_customers = 10, 20
    steps = 20


class MarketEnv:
    def __init__(self, rng, cfg=SellerConfig):
        self.rng, self.cfg = rng, cfg

    def reset(self):
        self.demand = self.rng.randrange(2)
        self.prev = [0, 0]
        return self.states()

    def states(self):
        return [(self.demand, self.prev[1]), (self.demand, self.prev[0])]

    def step(self, actions):
        cfg = self.cfg
        customers = cfg.high_customers if self.demand else cfg.low_customers
        rewards, sales_list = [], []
        for i in range(2):
            own, other = actions[i], actions[1 - i]
            conv = min(0.95, max(0.05, 0.30 + 0.25 * self.demand + 0.15 * own - 0.10 * other))
            sales = sum(self.rng.random() < conv for _ in range(customers))
            profit = sales * (cfg.price * (1 - cfg.discount_rate * own) - cfg.unit_cost)
            rewards.append(profit)
            sales_list.append(sales)
        info = dict(demand=self.demand, sales=sales_list)
        self.prev = list(actions)
        self.demand = self.rng.randrange(2)
        return self.states(), rewards, info


class SellerQAgent:
    def __init__(self, alpha, gamma, epsilon, seed):
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q = [[[0.0, 0.0] for _ in range(2)] for _ in range(2)]
        self.rng = random.Random(seed)

    def act(self, state, explore=True):
        d, p = state
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(2)
        return greedy_index(self.q[d][p], self.rng)

    def learn(self, state, action, reward, nxt, terminal):
        d, p = state
        target = reward if terminal else reward + self.gamma * max(self.q[nxt[0]][nxt[1]])
        self.q[d][p][action] += self.alpha * (target - self.q[d][p][action])


def rule_action(state):
    demand, comp_prev = state
    return 1 if demand == 1 and comp_prev == 1 else 0


OPPONENTS = {
    "rule": ("Luật cố định (giảm giá khi cầu cao và đối thủ vừa giảm)", rule_action),
    "always": ("Luôn giảm giá", lambda s: 1),
    "never": ("Không bao giờ giảm giá", lambda s: 0),
}


def run_agent02(alpha=0.10, gamma=0.90, epsilon_start=0.20, epsilon_min=0.02, decay=0.995,
                episodes=5000, opponent="rule", seed=42, eval_episodes=500):
    cfg = SellerConfig
    opp = OPPONENTS[opponent][1]
    env = MarketEnv(random.Random(seed))
    agent = SellerQAgent(alpha, gamma, epsilon_start, seed + 10000)
    curve = []
    for ep in range(episodes):
        agent.epsilon = max(epsilon_min, epsilon_start * decay ** ep)
        states, total = env.reset(), 0.0
        for t in range(cfg.steps):
            a = agent.act(states[0])
            b = opp(states[1])
            nxt, rewards, _ = env.step([a, b])
            agent.learn(states[0], a, rewards[0], nxt[0], t == cfg.steps - 1)
            states, total = nxt, total + rewards[0]
        curve.append(round(total / cfg.steps, 3))

    def evaluate(pick):
        e = MarketEnv(random.Random(seed + 500000))
        rl_r = op_r = disc = 0.0
        for _ in range(eval_episodes):
            st = e.reset()
            for _ in range(cfg.steps):
                a = pick(st[0])
                nxt, rw, _ = e.step([a, opp(st[1])])
                rl_r, op_r, disc = rl_r + rw[0], op_r + rw[1], disc + a
                st = nxt
        n = eval_episodes * cfg.steps
        return dict(profit_per_step=round(rl_r / n, 4), opponent_profit_per_step=round(op_r / n, 4),
                    discount_rate=round(disc / n, 4))

    summary = [dict(agent="Q-learning", **evaluate(lambda s: agent.act(s, explore=False))),
               dict(agent="Luật cố định", **evaluate(rule_action)),
               dict(agent="Luôn giảm giá", **evaluate(lambda s: 1)),
               dict(agent="Không giảm giá", **evaluate(lambda s: 0))]
    # một ván minh hoạ 20 bước với chính sách tham lam
    e = MarketEnv(random.Random(seed + 7))
    st, trace = e.reset(), []
    for t in range(cfg.steps):
        a, b = agent.act(st[0], explore=False), opp(st[1])
        nxt, rw, info = e.step([a, b])
        trace.append(dict(step=t + 1, demand=info["demand"], rl_action=a, opp_action=b,
                          rl_sales=info["sales"][0], opp_sales=info["sales"][1],
                          rl_profit=round(rw[0], 2), opp_profit=round(rw[1], 2)))
        st = nxt
    policy = {f"cầu={'cao' if d else 'thấp'}, đối thủ trước={'giảm' if p else 'không'}":
              ("giảm giá" if agent.q[d][p][1] > agent.q[d][p][0] else "giữ giá") for d in range(2) for p in range(2)}
    return dict(opponent=opponent, opponent_title=OPPONENTS[opponent][0], curve=curve, q=agent.q,
                summary=summary, trace=trace, policy=policy,
                params=dict(alpha=alpha, gamma=gamma, epsilon_start=epsilon_start, episodes=episodes, seed=seed))
