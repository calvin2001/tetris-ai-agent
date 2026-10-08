"""GA 가중치를 prior로 두고, 가치함수 V(afterstate)를 강화학습으로 추가 학습한다.

반복(= generalized policy iteration):
  1) 현재 정책(GA 점수 + beta*V, ε-greedy)으로 대결해 경험 수집
  2) 각 배치 이후의 실제 할인 누적보상 G 계산 (공격 줄 수 + 승패)
  3) V(특징) ≈ G 를 ridge 회귀로 맞춤  → 정책 평가(evaluation)
  4) 새 V로 다시 행동 선택                → 정책 개선(improvement)

예) python src/rl_finetune.py --init results/ga/best_model.json --rounds 5 --games 40 --beta 20
"""
from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from arena import ROOT, load_rule_agent, match
from agent_core import FEATURES, SHAPES, W, PlacementAgent, drop, features

GAMMA, ATTACK_REWARD, WIN_REWARD = 0.98, 0.05, 1.0
OPPONENTS = ("dellacherie", "codemyroad_genetic")


class ExploringAgent(PlacementAgent):
    """학습용: ε 확률로 무작위 배치 + 선택한 afterstate 특징을 기록."""

    def __init__(self, *a, epsilon=0.05, rng_seed=0, **kw):
        self.epsilon, self.rng = epsilon, random.Random(rng_seed)
        super().__init__(*a, **kw)

    def reset(self, seed, player):
        super().reset(seed, player)
        self.trace = []                       # [(특징 벡터, 계획 시점 누적 공격)]

    def plan(self, me):
        cmds = super().plan(me)
        if self.rng.random() < self.epsilon:  # 탐험: 현재 조각의 무작위 합법 배치
            rows = [sum(1 << x for x, c in enumerate(r) if c) for r in me["board"]] + [0] * 4
            piece = me["piece"]
            options = [(rot, tx) for rot in range(4) for tx in range(-3, W)
                       if drop(rows, SHAPES[piece][rot], tx) is not None]
            rot, tx = self.rng.choice(options)
            cmds = {0: [], 1: [5], 2: [7], 3: [6]}[(rot - int(me["rotation"])) % 4]
            cmds += ([2] if tx > me["x"] else [1]) * abs(tx - int(me["x"])) + [4]
        self._record(me, cmds)
        return cmds

    def _record(self, me, cmds):
        # 실제로 실행할 배치의 afterstate 특징을 다시 계산해서 기록
        rows = [sum(1 << x for x, c in enumerate(r) if c) for r in me["board"]] + [0] * 4
        piece = (me["hold"] or me["next"][0]) if cmds and cmds[0] == 8 else me["piece"]
        rot = (int(me["rotation"]) if cmds[0] != 8 else 0)
        x = int(me["x"]) if cmds[0] != 8 else 3
        for c in cmds:
            if c in (5, 6, 7):
                rot = (rot + {5: 1, 6: 3, 7: 2}[c]) % 4
            elif c in (1, 2):
                x += 1 if c == 2 else -1
        r = drop(rows, SHAPES[piece][rot], x)
        if r is not None:
            f = features(r[0], r[1], r[2], r[3])
            self.trace.append(([f[k] for k in FEATURES], me["attack"]))


def _job(args):
    model, opp, seed, seat, eps = args
    me = ExploringAgent(model["weights"], model.get("use_hold", True), model.get("value"),
                        model.get("beta", 0.0), epsilon=eps, rng_seed=seed * 2 + seat)
    if opp.endswith(".json"):                 # model.json 경로면 내 Agent 구조로 상대 생성
        rival = PlacementAgent.from_json(opp)
    else:                                     # 이름이면 키트의 룰베이스
        rival = load_rule_agent(opp)
    result = match(me, rival, seed, seat)
    # 보상: 다음 배치 계획 시점까지 늘어난 공격량, 마지막에 승패
    xs, attacks = [t[0] for t in me.trace], [t[1] for t in me.trace]
    rewards = [ATTACK_REWARD * (attacks[i + 1] - attacks[i]) for i in range(len(xs) - 1)] + [0.0]
    if rewards:
        rewards[-1] += WIN_REWARD if result["win"] else (0.0 if result["draw"] else -WIN_REWARD)
    returns, g = [], 0.0
    for r in reversed(rewards):
        g = r + GAMMA * g
        returns.append(g)
    return xs, returns[::-1], result["win"]


def fit_value(X, y, l2=1.0):
    X, y = np.asarray(X, float), np.asarray(y, float)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / std
    theta = np.linalg.solve(Z.T @ Z + l2 * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return {"bias": float(y.mean()), "theta": theta.tolist(),
            "mean": mean.tolist(), "std": std.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="GA 결과 model.json")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--games", type=int, default=40, help="라운드당 경기 수")
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "rl"))
    ap.add_argument("--vs", nargs="+", default=list(OPPONENTS),
                    help="상대 목록: 룰베이스 이름 또는 model.json 경로 (예: agents/v1_hold/model.json)")
    args = ap.parse_args()

    model = json.loads(Path(args.init).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1)
    with Pool(args.workers) as pool:
        for rnd in range(args.rounds):
            jobs = [(model, args.vs[i % len(args.vs)], rng.randrange(10**6),
                     (i // len(args.vs)) % 2, args.epsilon)
                    for i in range(args.games)]
            X, y, wins = [], [], 0
            for xs, gs, win in pool.map(_job, jobs):
                X += xs
                y += gs
                wins += win
            model = dict(model, value=fit_value(X, y), beta=args.beta)
            path = out / f"round{rnd:02d}_model.json"
            path.write_text(json.dumps(model, indent=2), encoding="utf-8")
            print(f"[round {rnd}] samples={len(X)} train_win_rate={wins / len(jobs):.2f} "
                  f"-> {path.name}", flush=True)


if __name__ == "__main__":
    main()
