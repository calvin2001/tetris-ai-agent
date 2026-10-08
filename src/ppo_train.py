"""PPO로 배치 정책을 미세 조정한다 (GA 가중치에서 출발).

정책(actor):  후보 배치마다 logit = (가중치 · 특징) / T  →  softmax로 확률을 만들고 샘플링
가치(critic): V(s) = c + u · E_π[특징]   (현재 정책이 고를 배치들의 평균 특징으로 상태를 요약)
학습:         GAE(λ)로 advantage를 구하고, PPO 클립 목적함수로 가중치를 업데이트

numpy만 사용한다 (PyTorch 불필요). 결과 model.json은 기존 Agent가 그대로 읽는다.

예) python src/ppo_train.py --init agents/my_agent/model.json \
        --vs agents/my_agent/model.json agents/v5_d2/model.json --iters 30 --games 40 --workers 10
"""
from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from arena import ROOT, load_rule_agent, match
from agent_core import FEATURES, SHAPES, W, PlacementAgent, drop, features, to_rows

GAMMA, LAM = 0.98, 0.95
ATTACK_REWARD, WIN_REWARD = 0.05, 1.0
F = len(FEATURES)


# ---------------------------------------------------------------- 경험 수집용 Agent
class PPOAgent(PlacementAgent):
    """softmax 정책으로 배치를 고르고, 학습에 필요한 정보를 기록한다."""

    def __init__(self, weights, temp, sample=True, rng_seed=0):
        super().__init__(weights, True)
        self.w = np.array([self.weights[k] for k in FEATURES], dtype=np.float64)
        self.temp, self.sample = temp, sample
        self.rng = np.random.default_rng(rng_seed)

    def reset(self, seed, player):
        super().reset(seed, player)
        self.trace = []   # (Φ, 고른 번호, 고른 확률의 log, 평균 특징, 누적 공격)

    def plan(self, me):
        rows = to_rows(me["board"])
        options = [(False, me["piece"], int(me["x"]), int(me["rotation"]))]
        hp = me["hold"] or (me["next"][0] if me["next"] else None)
        if hp:
            options.append((True, hp, 3, 0))
        phis, moves, safe = [], [], []
        for hold, piece, sx, sr in options:
            for rot in ((0,) if piece == "O" else range(4)):
                cells = SHAPES[piece][rot]
                for tx in range(-min(c[0] for c in cells), W - max(c[0] for c in cells)):
                    r = drop(rows, cells, tx)
                    if r is None:
                        continue
                    f = features(r[0], r[1], r[2], r[3], self.ctx)
                    phis.append([f[k] for k in FEATURES])
                    moves.append((hold, rot, tx, sx, sr))
                    safe.append(not r[4])
        if not phis:
            return [4]
        if any(safe):                       # 천장을 넘는 배치는 후보에서 아예 뺀다 (학습 계산과 일치시키기 위해)
            keep = [k for k, ok in enumerate(safe) if ok]
            phis = [phis[k] for k in keep]
            moves = [moves[k] for k in keep]
        phi = np.asarray(phis, dtype=np.float64)
        logits = phi @ self.w / self.temp
        p = np.exp(logits - logits.max())
        p /= p.sum()
        a = int(self.rng.choice(len(p), p=p)) if self.sample else int(np.argmax(logits))
        self.trace.append((phi.astype(np.float32), a, float(np.log(p[a] + 1e-12)),
                           (p[:, None] * phi).sum(0).astype(np.float32), me["attack"]))
        hold, rot, tx, sx, sr = moves[a]
        return self.to_commands(hold, rot, tx, sx, sr)


def make_rival(spec):
    if spec.endswith(".json"):
        rival = PlacementAgent.from_json(spec)
        rival.depth = 1                     # 학습 속도를 위해 상대도 1수 탐색
        return rival
    return load_rule_agent(spec)


def _job(args):
    w, temp, opp, seed, seat = args
    me = PPOAgent(dict(zip(FEATURES, w)), temp, True, rng_seed=seed * 2 + seat)
    res = match(me, make_rival(opp), seed, seat)
    tr = me.trace
    if not tr:
        return None
    att = [t[4] for t in tr]
    rew = [ATTACK_REWARD * (att[i + 1] - att[i]) for i in range(len(tr) - 1)] + [0.0]
    rew[-1] += WIN_REWARD if res["win"] else (0.0 if res["draw"] else -WIN_REWARD)
    return {"phi": [t[0] for t in tr], "act": [t[1] for t in tr], "logp": [t[2] for t in tr],
            "mean_phi": np.stack([t[3] for t in tr]), "rew": np.asarray(rew), "win": res["win"]}


# ---------------------------------------------------------------- 학습
def discounted(x, g):
    out, run = np.zeros_like(x), 0.0
    for i in range(len(x) - 1, -1, -1):
        run = x[i] + g * run
        out[i] = run
    return out


def fit_critic(X, y, l2=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    u = np.linalg.solve(Z.T @ Z + l2 * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda A: y.mean() + ((A - mu) / sd) @ u


def pad(phis):
    n, cmax = len(phis), max(p.shape[0] for p in phis)
    out = np.zeros((n, cmax, F), dtype=np.float32)
    mask = np.zeros((n, cmax), dtype=bool)
    for i, p in enumerate(phis):
        out[i, :len(p)] = p
        mask[i, :len(p)] = True
    return out, mask


def policy_terms(w, temp, phi, mask, act):
    logits = np.where(mask, phi @ w / temp, -1e30)
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits) * mask
    p /= p.sum(1, keepdims=True)
    logp_all = np.log(p + 1e-12)
    idx = np.arange(len(act))
    mean_phi = (p[:, :, None] * phi).sum(1)
    grad_logp = (phi[idx, act] - mean_phi) / temp                 # ∇_w log π(a|s)
    ent = -(p * logp_all).sum(1)
    grad_ent = -((p * logp_all)[:, :, None] * (phi - mean_phi[:, None, :])).sum(1) / temp
    return logp_all[idx, act], grad_logp, ent, grad_ent


def ppo_update(w, temp, data, args):
    phi, mask = pad(data["phi"])
    act, logp_old, adv = data["act"], data["logp"], data["adv"]
    m, v = np.zeros_like(w), np.zeros_like(w)
    b1, b2, step = 0.9, 0.999, 0
    n = len(act)
    stats = {}
    for _ in range(args.epochs):
        if stats.get("approx_kl", 0.0) > args.target_kl:   # 정책이 너무 많이 바뀌면 이번 반복은 여기서 멈춤
            break
        order = np.random.permutation(n)
        for s in range(0, n, args.batch):
            i = order[s:s + args.batch]
            logp, glogp, ent, gent = policy_terms(w, temp, phi[i], mask[i], act[i])
            ratio = np.exp(logp - logp_old[i])
            a = adv[i]
            clipped = ((a > 0) & (ratio > 1 + args.clip)) | ((a < 0) & (ratio < 1 - args.clip))
            coef = np.where(clipped, 0.0, a * ratio)
            # 최대화할 목표: mean(min(r·A, clip(r)·A)) + c_ent · mean(H)
            grad = (coef[:, None] * glogp).mean(0) + args.ent * gent.mean(0)
            step += 1
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad ** 2
            w = w + args.lr * (m / (1 - b1 ** step)) / (np.sqrt(v / (1 - b2 ** step)) + 1e-8)
            stats = {"clip_frac": float(clipped.mean()), "entropy": float(ent.mean()),
                     "approx_kl": float((logp_old[i] - logp).mean())}
            if stats["approx_kl"] > 1.5 * args.target_kl:
                break
    return w, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="출발 model.json (GA 가중치)")
    ap.add_argument("--vs", nargs="+", default=["dellacherie"],
                    help="상대: 룰베이스 이름 또는 model.json 경로 (상대는 1수 탐색으로 실행)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--games", type=int, default=40, help="반복당 경기 수")
    ap.add_argument("--temp", type=float, default=1.0, help="softmax 온도 (클수록 탐험 많음)")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ent", type=float, default=0.001, help="엔트로피 보너스 계수")
    ap.add_argument("--target-kl", type=float, default=0.02, help="반복당 정책 변화 상한 (넘으면 조기 종료)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "ppo"))
    args = ap.parse_args()

    init = json.loads(Path(args.init).read_text(encoding="utf-8"))
    w = np.array([float(init["weights"].get(k, 0.0)) for k in FEATURES])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    with Pool(args.workers) as pool:
        for it in range(args.iters):
            t0 = time.time()
            jobs = [(w.tolist(), args.temp, args.vs[g % len(args.vs)], rng.randrange(10**6),
                     (g // len(args.vs)) % 2) for g in range(args.games)]
            games = [r for r in pool.map(_job, jobs) if r]
            # critic: 평균 특징 → 몬테카를로 리턴
            X = np.concatenate([g["mean_phi"] for g in games])
            G = np.concatenate([discounted(g["rew"], GAMMA) for g in games])
            V = fit_critic(X, G)
            advs = []
            for g in games:
                v = V(g["mean_phi"])
                v_next = np.append(v[1:], 0.0)
                delta = g["rew"] + GAMMA * v_next - v
                advs.append(discounted(delta, GAMMA * LAM))   # GAE(λ)
            adv = np.concatenate(advs)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            data = {"phi": [p for g in games for p in g["phi"]],
                    "act": np.concatenate([g["act"] for g in games]),
                    "logp": np.concatenate([g["logp"] for g in games]),
                    "adv": adv}
            w, st = ppo_update(w, args.temp, data, args)
            wins = sum(g["win"] for g in games) / len(games)
            model = {"weights": dict(zip(FEATURES, map(float, w))), "use_hold": True,
                     "search_depth": init.get("search_depth", 1), "beam": init.get("beam", 3),
                     "ply1_weight": init.get("ply1_weight", 1.0), "ppo_temp": args.temp}
            path = out / f"iter{it:02d}_model.json"
            path.write_text(json.dumps(model, indent=2), encoding="utf-8")
            print(f"[iter {it:02d}] decisions={len(adv)} train_win={wins:.2f} "
                  f"entropy={st['entropy']:.2f} kl={st['approx_kl']:.4f} "
                  f"clip={st['clip_frac']:.2f} {time.time() - t0:.0f}s -> {path.name}", flush=True)


if __name__ == "__main__":
    main()
