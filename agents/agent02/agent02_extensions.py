"""Controlled extensions for AGENT02; model oracle is explicitly not model-free."""
from dataclasses import replace
from itertools import product
import json, platform, sys, time
from agent02_experiment import Config, ECommerceEnvironment, QLearningAgent, RuleBasedAgent, train_rl_against_rule, evaluate_pair, write_csv, moving_average
import numpy as np
import matplotlib.pyplot as plt

SEEDS = [11, 22, 33, 44, 55]
STATES = list(product(range(2), repeat=3))  # demand, own previous, opponent previous

def validate_config(c):
    """Reject invalid experiment parameters before spending time on simulation."""
    if c.episodes < 1 or c.steps_per_episode < 1 or c.evaluation_episodes < 2:
        raise ValueError('Episode counts must be positive; evaluation >= 2.')
    if not 0 < c.alpha <= 1 or not 0 <= c.gamma <= 1:
        raise ValueError('Require 0 < alpha <= 1 and 0 <= gamma <= 1.')
    if not 0 <= c.epsilon_min <= c.epsilon_start <= 1:
        raise ValueError('Invalid exploration range.')
    if not 0 < c.epsilon_decay <= 1 or not 0 <= c.discount_rate < 1:
        raise ValueError('Invalid decay or discount.')
    if c.price <= 0 or c.unit_cost < 0:
        raise ValueError('Invalid price or cost.')

def expected_profit(c, demand, own_action, opponent_action):
    """Exact one-step mean, using E[Binomial(N,p)] = Np."""
    n = c.low_demand_customers if demand == 0 else c.high_demand_customers
    p = np.clip(.30 + .25*demand + .15*own_action - .10*opponent_action, .05, .95)
    return float(n*p*(c.price*(1-c.discount_rate*own_action)-c.unit_cost))

def oracle_dp(c, gamma=1.0):
    """Finite-horizon full-state DP against the known fixed rule opponent."""
    q = np.zeros((c.steps_per_episode+1, 2, 2, 2, 2))
    v = np.zeros(q.shape[:-1])
    for h in range(1, c.steps_per_episode+1):
        for d, own_prev, opp_prev in STATES:
            b = RuleBasedAgent().act((d, own_prev))
            for a in (0, 1):
                future = .5*(v[h-1, 0, a, b]+v[h-1, 1, a, b])
                q[h, d, own_prev, opp_prev, a] = expected_profit(c,d,a,b)+gamma*future
            v[h,d,own_prev,opp_prev] = q[h,d,own_prev,opp_prev].max()
    return q, v

def action_probabilities(agent, kind, h, state):
    """Replicate greedy random tie-breaking; oracle uses deterministic argmax."""
    d, own_prev, opp_prev = state
    if kind == 'oracle':
        p = np.zeros(2); p[np.argmax(agent[h,d,own_prev,opp_prev])] = 1
        return p
    if kind == 'horizon': values = agent.q[h,d,own_prev,opp_prev]
    else: values = agent.q[d,opp_prev]
    best = np.isclose(values, values.max())
    return best.astype(float)/best.sum()

def exact_evaluation(c, agent, kind):
    """Integrate demand and binomial means exactly; report undiscounted profit/step."""
    value = np.zeros((c.steps_per_episode+1,2,2,2,2))
    for h in range(1,c.steps_per_episode+1):
        for state in STATES:
            d, own_prev, opp_prev = state
            b = RuleBasedAgent().act((d,own_prev))
            for a,p in enumerate(action_probabilities(agent,kind,h,state)):
                reward = np.array([expected_profit(c,d,a,b),expected_profit(c,d,b,a)])
                future = .5*(value[h-1,0,a,b]+value[h-1,1,a,b])
                value[(h,)+state] += p*(reward+future)
    return .5*(value[c.steps_per_episode,0,0,0]+value[c.steps_per_episode,1,0,0])/c.steps_per_episode

class HorizonQLearningAgent:
    """Model-free finite-horizon Q-learning with sufficient observable history."""
    def __init__(self, c, seed):
        self.q = np.zeros((c.steps_per_episode+1,2,2,2,2))
        self.visits = np.zeros_like(self.q,dtype=int)
        self.rng = np.random.default_rng(seed)
        self.epsilon = c.epsilon_start
    def act(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(2))
        values = self.q[state]
        best = np.flatnonzero(np.isclose(values,values.max()))
        return int(self.rng.choice(best))
    def learn(self, state, action, reward, next_state, terminal=False):
        key = state+(action,)
        self.visits[key] += 1
        rate = self.visits[key]**(-.6)
        future = 0.0 if terminal else float(self.q[next_state].max())
        # gamma = 1 matches the finite-horizon cumulative profit objective.
        self.q[key] += rate*(reward+future-self.q[key])

def train_horizon(c, seed):
    """Train with (remaining steps, demand, own previous, opponent previous)."""
    validate_config(c)
    env = ECommerceEnvironment(c,seed=seed)
    agent = HorizonQLearningAgent(c,seed+10_000)
    rule = RuleBasedAgent()
    rows = []
    for episode in range(c.episodes):
        obs = env.reset(); total = 0.0
        agent.epsilon = max(c.epsilon_min,c.epsilon_start*c.epsilon_decay**episode)
        for t in range(c.steps_per_episode):
            h = c.steps_per_episode-t
            state = (h,obs[0][0],env.previous_actions[0],obs[0][1])
            a = agent.act(state); b = rule.act(obs[1])
            nxt, rewards, _ = env.step([a,b])
            next_state = (h-1,nxt[0][0],a,b)
            agent.learn(state,a,rewards[0],next_state,terminal=h==1)
            total += rewards[0]; obs = nxt
        rows.append(total/c.steps_per_episode)
    return agent, rows

def evaluate_extended(c, agent, kind, seed):
    """Frozen Monte Carlo evaluation; distinct RNG and no learning calls."""
    env = ECommerceEnvironment(c,seed=seed)
    rng = np.random.default_rng(seed+123_456)
    rewards = np.zeros(2); counts = np.zeros((2,2),int); episodes = []
    before = agent.q.copy() if hasattr(agent,'q') else agent.copy()
    for ep in range(c.evaluation_episodes):
        obs = env.reset(); ep_r = np.zeros(2)
        for t in range(c.steps_per_episode):
            state = (obs[0][0],env.previous_actions[0],obs[0][1])
            p = action_probabilities(agent,kind,c.steps_per_episode-t,state)
            a = int(rng.choice(2,p=p)); b = RuleBasedAgent().act(obs[1])
            obs, r, _ = env.step([a,b]); rewards += r; ep_r += r
            counts[0,a] += 1; counts[1,b] += 1
        episodes.append({'episode':ep+1,'rl_reward':ep_r[0],'rule_reward':ep_r[1]})
    after = agent.q if hasattr(agent,'q') else agent
    assert np.array_equal(before,after), 'Evaluation changed Q values'
    steps = c.evaluation_episodes*c.steps_per_episode
    return {'mean_reward':rewards[0]/steps,'rule_mean':rewards[1]/steps,
            'discount_rate':counts[0,1]/steps,'rule_discount_rate':counts[1,1]/steps}, episodes

def summarize(rows, keys, metric='mean_reward'):
    """Mean, sample SD and two-sided Student t CI across five training seeds."""
    groups = {}
    for row in rows: groups.setdefault(tuple(row[k] for k in keys),[]).append(row[metric])
    result = []
    for key, values in groups.items():
        x = np.array(values); n = len(x); sd = float(x.std(ddof=1))
        # t(.975,4) is fixed because this protocol uses exactly five seeds.
        assert n == 5
        half = 2.7764451051977987*sd/np.sqrt(n)
        result.append(dict(zip(keys,key),mean=float(x.mean()),sd=sd,n=n,
                           ci_low=float(x.mean()-half),ci_high=float(x.mean()+half)))
    return result

def run_controlled_sweep(c, out):
    """A-D with one factor changed, five seeds each, raw data and 100-episode windows."""
    validate_config(c); out.mkdir(exist_ok=True,parents=True)
    settings = [
        ('baseline',{}), ('alpha_001',{'alpha':.01}), ('alpha_050',{'alpha':.50}),
        ('explore_low',{'epsilon_start':.05,'epsilon_min':.02,'epsilon_decay':.999}),
        ('explore_high',{'epsilon_start':.50,'epsilon_min':.05,'epsilon_decay':.995}),
        ('fixed_005',{'epsilon_start':.05,'epsilon_min':.05,'epsilon_decay':1.0}),
        ('fixed_030',{'epsilon_start':.30,'epsilon_min':.30,'epsilon_decay':1.0})]
    all_rows = []; windows = []; tables = {}
    for name, changes in settings:
        cfg = replace(c,**changes)
        for seed in SEEDS:
            start = time.perf_counter()
            agent, train = train_rl_against_rule(cfg,seed=seed)
            summary, ev = evaluate_pair(cfg,agent,seed=seed+800_000)
            r, b = summary; exact = exact_evaluation(cfg,agent,'local')
            row = {'setting':name,'seed':seed,'alpha':cfg.alpha,
                   'epsilon_start':cfg.epsilon_start,'epsilon_min':cfg.epsilon_min,
                   'epsilon_decay':cfg.epsilon_decay,'gamma':cfg.gamma,
                   'mean_reward':r['mean_reward_per_step'],'rule_mean':b['mean_reward_per_step'],
                   'discount_rate':r['discount_action_rate'],'rule_discount_rate':b['discount_action_rate'],
                   'training_last100':float(np.mean([x['rl_mean_reward_per_step'] for x in train[-100:]])),
                   'exact_mean':exact[0],'exact_rule_mean':exact[1]}
            all_rows.append(row); tables[f'{name}_{seed}']=agent.q
            write_csv(out/f'{name}_{seed}_training.csv',train)
            write_csv(out/f'{name}_{seed}_evaluation.csv',ev)
            for first in range(0,len(train),100):
                chunk = train[first:first+100]
                windows.append({'setting':name,'seed':seed,'episode':first+len(chunk),
                    'reward':float(np.mean([x['rl_mean_reward_per_step'] for x in chunk])),
                    'rule_reward':float(np.mean([x['rule_mean_reward_per_step'] for x in chunk]))})
            print(f'Sweep {name} seed={seed}: eval={row["mean_reward"]:.3f}, exact={exact[0]:.3f}, {time.perf_counter()-start:.1f}s',flush=True)
            write_csv(out/'controlled_runs.csv',all_rows)
    write_csv(out/'training_windows.csv',windows)
    write_csv(out/'controlled_summary.csv',summarize(all_rows,['setting']))
    np.savez_compressed(out/'controlled_qtables.npz',**tables)
    return all_rows, windows

def run_robustness(c, out):
    """Discount stress test: frozen transfer, retraining, richer state, model oracle."""
    tables = np.load(out/'controlled_qtables.npz'); rows = []
    for discount in [.20,.10,.05]:
        cfg = replace(c,discount_rate=discount)
        oracle,_ = oracle_dp(cfg)
        for seed in SEEDS:
            frozen = QLearningAgent(); frozen.q = tables[f'baseline_{seed}'].copy()
            if discount == .20: retrained = frozen
            else: retrained,_ = train_rl_against_rule(cfg,seed=seed)
            improved, curve = train_horizon(cfg,seed)
            candidates = [('reference_transfer',frozen,'local'),('reference_retrained',retrained,'local'),
                          ('horizon_q',improved,'horizon'),('oracle_dp',oracle,'oracle')]
            if discount == .10:
                ablated,_ = train_rl_against_rule(replace(cfg,gamma=1.0),seed=seed)
                candidates.append(('local_gamma1',ablated,'local'))
            np.savez_compressed(out/f'robust_q_{discount:.2f}_{seed}.npz',
                reference=retrained.q,horizon=improved.q,visits=improved.visits,oracle=oracle)
            write_csv(out/f'horizon_{discount:.2f}_{seed}_training.csv',
                      [{'episode':i+1,'reward':x} for i,x in enumerate(curve)])
            optimal = exact_evaluation(cfg,oracle,'oracle')[0]
            for name, agent, kind in candidates:
                metrics, episodes = evaluate_extended(cfg,agent,kind,seed+900_000)
                exact = exact_evaluation(cfg,agent,kind)
                rows.append(dict(discount=discount,seed=seed,agent=name,**metrics,
                                 exact_mean=exact[0],exact_rule_mean=exact[1],
                                 oracle_mean=optimal,regret=optimal-exact[0]))
                write_csv(out/f'robust_{discount:.2f}_{seed}_{name}_evaluation.csv',episodes)
            print(f'Robustness discount={discount:.2f} seed={seed}: completed',flush=True)
            write_csv(out/'robustness_runs.csv',rows)
    write_csv(out/'robustness_summary.csv',summarize(rows,['discount','agent']))
    return rows

def make_figures(c, out, runs, windows, robust):
    """Save analysis figures; error bars represent seed variability, not individual steps."""
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    figs = []
    def save(name):
        path=out/(name+'.png'); plt.tight_layout(); plt.savefig(path,dpi=190); plt.close(); figs.append(str(path))
    for names,title,name in [(['baseline','alpha_001','alpha_050'],'Learning rate: training windows','alpha_curves'),
        (['baseline','explore_low','explore_high','fixed_005','fixed_030'],'Exploration: training windows','exploration_curves')]:
        plt.figure(figsize=(9,4.8))
        for label in names:
            selected = [r for r in windows if r['setting']==label]
            x = sorted({r['episode'] for r in selected})
            y = np.array([[r['reward'] for r in selected if r['seed']==s] for s in SEEDS])
            mean,sd=y.mean(0),y.std(0,ddof=1)
            plt.plot(x,mean,label=label); plt.fill_between(x,mean-sd,mean+sd,alpha=.13)
        plt.title(title); plt.xlabel('Episode (non-overlapping windows of 100)');plt.ylabel('Profit / step');plt.legend(ncol=2);plt.grid(alpha=.2);save(name)
    sums=summarize(runs,['setting']); x=np.arange(len(sums))
    plt.figure(figsize=(10,4.8)); means=np.array([r['mean'] for r in sums]); err=np.array([r['ci_high']-r['mean'] for r in sums])
    plt.bar(x,means,yerr=err,capsize=4,color='#3274A1');plt.xticks(x,[r['setting'] for r in sums],rotation=25,ha='right');plt.axhline(35,color='black',ls='--',label='Always no discount: exact 35');plt.ylabel('Frozen evaluation profit / step');plt.title('Five training seeds: mean and 95% t intervals');plt.legend();save('evaluation_ci')
    plt.figure(figsize=(8.5,4.6))
    for label in ['baseline','alpha_001','alpha_050']:
        rs=[r for r in runs if r['setting']==label]
        plt.plot(SEEDS,[r['mean_reward'] for r in rs],'o-',label=label)
    plt.xlabel('Training seed');plt.ylabel('Evaluation profit / step');plt.title('Variation between independently trained policies');plt.legend();save('seed_variation')
    plt.figure(figsize=(8.5,4.8))
    for name in ['reference_transfer','reference_retrained','horizon_q','oracle_dp']:
        xs=[.05,.10,.20]; ys=[np.mean([r['exact_mean'] for r in robust if r['agent']==name and r['discount']==d]) for d in xs]
        plt.plot(np.array(xs)*100,ys,'o-',label=name)
    plt.xlabel('Discount (%)');plt.ylabel('Exact expected profit / step');plt.title('Discount robustness: transfer versus retraining');plt.legend();plt.grid(alpha=.2);save('robustness')
    plt.figure(figsize=(9,4.5)); labels=['reference_retrained','local_gamma1','horizon_q']
    vals=[[r['regret'] for r in robust if r['discount']==.1 and r['agent']==name] for name in labels]
    plt.boxplot(vals,tick_labels=labels,showmeans=True);plt.ylabel('Oracle gap (expected profit / step)');plt.title('10% discount: limited ablation and remaining gap');save('oracle_gap')
    q=np.load(out/'controlled_qtables.npz')['baseline_11'].reshape(4,2)
    plt.figure(figsize=(7,4));plt.imshow(q,aspect='auto',cmap='Blues');plt.colorbar(label='Q value');plt.xticks([0,1],['No discount','Discount']);plt.yticks(range(4),['(0,0)','(0,1)','(1,0)','(1,1)']);plt.xlabel('Action');plt.ylabel('(demand, opponent previous)');plt.title('Learned reference Q table: seed 11')
    for i in range(4):
        for j in range(2):plt.text(j,i,f'{q[i,j]:.2f}',ha='center',va='center',color='white' if (q[i,j]-q.min())/max(float(q.max()-q.min()),1e-9)>.60 else 'black')
    save('q_heatmap')
    plt.figure(figsize=(8.5,4.6));t=np.arange(c.episodes)
    for name,start,minimum,decay in [('baseline',.2,.02,.995),('low',.05,.02,.999),('high',.5,.05,.995),('fixed .05',.05,.05,1),('fixed .30',.3,.3,1)]:
        plt.plot(t+1,np.maximum(minimum,start*decay**t),label=name)
    plt.xlabel('Training episode');plt.ylabel('Exploration probability');plt.title('Exploration schedules');plt.legend();save('epsilon_schedules')
    return figs
