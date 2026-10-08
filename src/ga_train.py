"""유전 알고리즘으로 PlacementAgent의 특징 가중치를 튜닝한다.

예) python src/ga_train.py --pop 20 --gens 15 --games 4 --workers 8
"""
from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

from arena import ROOT, load_rule_agent, match
from agent_core import DELLACHERIE, FEATURES, PlacementAgent

OPPONENTS = ("dellacherie", "codemyroad_genetic")
_cache = {}


def _opponent(name):
    if name not in _cache:
        _cache[name] = load_rule_agent(name)
    return _cache[name]


def _job(args):
    weights, opp, seed, seat = args
    if isinstance(opp, dict):
        rival = PlacementAgent(opp)                 # 챔피언 (가중치 사전)
    elif opp.endswith(".json"):
        rival = PlacementAgent.from_json(opp)       # 보관해 둔 이전 버전 Agent
    else:
        rival = _opponent(opp)                      # 키트의 룰베이스
    return match(PlacementAgent(weights), rival, seed, seat)


def fitness(results) -> float:
    n = len(results)
    win_rate = sum(r["win"] for r in results) / n
    attack_edge = sum(r["my_attack"] - r["opp_attack"] for r in results) / n
    return win_rate + 0.001 * attack_edge          # 승률 우선, 공격량 차이는 동점 처리용


def mutate(w, sigma, rng):
    return {k: w[k] + rng.gauss(0, sigma) if rng.random() < 0.5 else w[k] for k in FEATURES}


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in FEATURES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--games", type=int, default=4, help="상대 1명당 경기 수(짝수, 좌우 교대)")
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "ga"))
    ap.add_argument("--init", default=None, help="출발 가중치 model.json (없으면 Dellacherie)")
    ap.add_argument("--rivals", nargs="+", default=list(OPPONENTS),
                    help="상대 목록 (챔피언은 자동 추가): 룰베이스 이름 또는 model.json 경로")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.init:
        init_w = json.loads(Path(args.init).read_text(encoding="utf-8"))["weights"]
        start = {k: float(init_w.get(k, 0.0)) for k in FEATURES}
    else:
        start = {k: DELLACHERIE.get(k, 0.0) for k in FEATURES}
    population = [start] + [mutate(start, args.sigma, rng) for _ in range(args.pop - 1)]
    champion = start   # 직전 세대 1등도 상대로 넣어 '룰베이스는 다 이겨서 점수 차이가 안 나는' 문제를 막는다

    with Pool(args.workers) as pool:
        for gen in range(args.gens):
            t0 = time.time()
            # 세대마다 새 seed (같은 세대 안에서는 모두 같은 seed → 공정 비교)
            seeds = [rng.randrange(10**6) for _ in range(args.games // 2)]
            rivals = tuple(args.rivals) + (champion,)
            jobs = [(w, opp, s, seat) for w in population for opp in rivals
                    for s in seeds for seat in (0, 1)]
            flat = pool.map(_job, jobs)
            per = len(rivals) * len(seeds) * 2
            scored = sorted(((fitness(flat[i * per:(i + 1) * per]), w)
                             for i, w in enumerate(population)),
                            key=lambda t: t[0], reverse=True)
            best_fit, best_w = scored[0]
            print(f"[gen {gen:02d}] best={best_fit:.3f} "
                  f"mean={sum(f for f, _ in scored) / len(scored):.3f} {time.time() - t0:.0f}s",
                  flush=True)
            with open(out / "log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"gen": gen, "fitness": best_fit, "weights": best_w}) + "\n")
            (out / "best_model.json").write_text(
                json.dumps({"weights": best_w, "use_hold": True}, indent=2), encoding="utf-8")
            champion = best_w
            elites = [w for _, w in scored[:args.elite]]
            sigma = args.sigma * (0.9 ** gen)
            population = elites + [
                mutate(crossover(rng.choice(elites), rng.choice(elites), rng), sigma, rng)
                for _ in range(args.pop - args.elite)]


if __name__ == "__main__":   # Windows multiprocessing에 필수
    main()
