"""서브프로세스 없이 한 프로세스 안에서 대결 → 공식 evaluate.py보다 빠름(학습용).

최종 성능 확인은 반드시 공식 tools/evaluate.py로 다시 한다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "kits" / "tetris_competition_kit_v2"
RULE_MODELS = ROOT / "kits" / "local_tetris_battle_kit_v2" / "rule_based_tetris_models"
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(ROOT / "agents" / "my_agent"))

from tetrio_env import Action, DuelEngine, competition_config, competition_observation  # noqa: E402


def load_rule_agent(name: str):
    """'dellacherie' 또는 'codemyroad_genetic' 룰베이스 Agent를 그대로 불러온다."""
    spec = importlib.util.spec_from_file_location("classic_core", RULE_MODELS / "classic_core.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ClassicPlacementAgent(RULE_MODELS / name / "model.json")


def play(agent0, agent1, seed: int) -> dict:
    engine = DuelEngine(config=competition_config(), seed=seed)
    agents = (agent0, agent1)
    for i, agent in enumerate(agents):
        agent.reset(seed, i)
    while not engine.done:
        engine.step(tuple(Action(int(a.act(competition_observation(engine, i))))
                          for i, a in enumerate(agents)))
    return {"winner": engine.winner, "ticks": engine.tick,
            "attack": [p.attack_sent for p in engine.players]}


def match(me, opponent, seed: int, seat: int) -> dict:
    """seat=0이면 내가 player_0. 결과를 '내 기준'으로 돌려준다."""
    r = play(me, opponent, seed) if seat == 0 else play(opponent, me, seed)
    return {"win": r["winner"] == seat, "draw": r["winner"] is None, "ticks": r["ticks"],
            "my_attack": r["attack"][seat], "opp_attack": r["attack"][1 - seat]}
