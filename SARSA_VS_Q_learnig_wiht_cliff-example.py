# Cliff Walking: SARSA vs Q-learning with epsilon-greedy
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Environment --------------------
class CliffEnv:
    """
    4x12 Cliff Walking (Sutton & Barto)
    Start=(3,0), Goal=(3,11)
    Reward: -1 per step; stepping into the cliff (row=3, col=1..10) gives -100 and resets to Start.
    Episode ends only when reaching Goal.
    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    """
    def __init__(self, H=4, W=12, seed=None):
        self.H, self.W = H, W
        self.start = (H-1, 0)
        self.goal = (H-1, W-1)
        self.cliff = {(H-1, c) for c in range(1, W-1)}
        self.rng = np.random.default_rng(seed)
        self.reset()

    def to_state(self, pos):
        r, c = pos
        return r * self.W + c

    def reset(self):
        self.pos = self.start
        return self.to_state(self.pos)

    def step(self, a):
        r, c = self.pos
        if a == 0: r -= 1
        elif a == 1: c += 1
        elif a == 2: r += 1
        elif a == 3: c -= 1
        # stay within grid
        r = max(0, min(self.H-1, r))
        c = max(0, min(self.W-1, c))
        nxt = (r, c)

        if nxt in self.cliff:
            reward, done = -100, False
            self.pos = self.start  # fall and reset
        elif nxt == self.goal:
            reward, done = -1, True
            self.pos = nxt
        else:
            reward, done = -1, False
            self.pos = nxt

        return self.to_state(self.pos), reward, done

# -------------------- Policy --------------------
def epsilon_greedy(Q, s, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(Q.shape[1])
    q = Q[s]
    m = q.max()
    # break ties uniformly
    idx = np.flatnonzero(np.isclose(q, m))
    return rng.choice(idx)

# -------------------- Learners --------------------
def run_sarsa(env, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1,
              seed=0, max_steps=10000):
    rng = np.random.default_rng(seed)
    nS, nA = env.H * env.W, 4
    Q = np.zeros((nS, nA))
    ep_returns = []

    for _ in range(episodes):
        s = env.reset()
        a = epsilon_greedy(Q, s, epsilon, rng)
        total = 0.0
        for _ in range(max_steps):
            s2, r, done = env.step(a)
            total += r
            if done:
                Q[s, a] += alpha * (r - Q[s, a])
                break
            a2 = epsilon_greedy(Q, s2, epsilon, rng)
            target = r + gamma * Q[s2, a2]
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s2, a2
        ep_returns.append(total)
    return np.array(ep_returns), Q

def run_q_learning(env, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1,
                   seed=1, max_steps=10000):
    rng = np.random.default_rng(seed)
    nS, nA = env.H * env.W, 4
    Q = np.zeros((nS, nA))
    ep_returns = []

    for _ in range(episodes):
        s = env.reset()
        total = 0.0
        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, epsilon, rng)
            s2, r, done = env.step(a)
            total += r
            if done:
                Q[s, a] += alpha * (r - Q[s, a])
                break
            target = r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
        ep_returns.append(total)
    return np.array(ep_returns), Q

# -------------------- Utils --------------------
def smooth(x, k=10):
    """Simple moving average for nicer curves."""
    if k <= 1:
        return x
    c = np.convolve(x, np.ones(k) / k, mode="valid")
    return np.concatenate([np.full(k - 1, c[0]), c])

# -------------------- Run & Plot --------------------
if __name__ == "__main__":
    episodes = 500
    epsilon = 0.5
    alpha = 0.5
    gamma = 1.0

    env = CliffEnv(seed=42)
    sarsa_ret, _ = run_sarsa(env, episodes, alpha, gamma, epsilon, seed=0)

    # 새 시드로 동일 환경 새로 생성(내부 상태 공유 방지)
    env = CliffEnv(seed=43)
    q_ret, _ = run_q_learning(env, episodes, alpha, gamma, epsilon, seed=0)

    plt.figure(figsize=(7, 4))
    plt.plot(smooth(sarsa_ret, 10), label="SARSA")
    plt.plot(smooth(q_ret, 10), label="Q-learning")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episode")
    plt.title(r"Cliff Walking ($\varepsilon$-greedy, $\varepsilon=0.5$)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig("cliff_sarsa_qlearning.png", dpi=150)  # 저장하고 싶으면 사용
