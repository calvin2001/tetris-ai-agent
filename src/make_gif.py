"""경기 한 판을 GIF로 저장한다 (README용).

예) python src/make_gif.py --opp dellacherie --seed 5000001
    python src/make_gif.py --opp agents/v1_hold/model.json --seed 9950000 --out docs/vs_v1.gif
기본값: 최대 120초, 6 tick마다 한 장, 2배속 재생 → 보통 1~3MB

필요: pip install pillow   (제출물에는 필요 없음, GIF 만들 때만)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from arena import Action, DuelEngine, competition_config, competition_observation, load_rule_agent
from agent_core import SHAPES, PlacementAgent

COLORS = {0: (16, 20, 30), 1: (49, 199, 239), 2: (62, 82, 229), 3: (239, 151, 43),
          4: (246, 211, 47), 5: (82, 209, 84), 6: (177, 76, 229), 7: (235, 65, 74),
          8: (105, 112, 128)}
PIECE_ID = {p: i + 1 for i, p in enumerate("IJLOSTZ")}
BG, GRID, TEXT, DIM, RED = (8, 10, 18), (34, 40, 56), (235, 238, 245), (140, 146, 160), (230, 60, 60)
CELL, MINI = 16, 8
BOARD_W, BOARD_H = 10 * CELL, 20 * CELL
PANEL_W = BOARD_W + 2 * 56
WIDTH, HEIGHT = PANEL_W * 2 + 24, BOARD_H + 92


def load_agent(spec: str):
    """룰베이스 이름 또는 model.json 경로 → (Agent, 표시 이름)."""
    if spec.endswith(".json"):
        path = Path(spec)
        name = path.parent.name
        manifest = path.parent / "manifest.json"
        if manifest.exists():
            name = json.loads(manifest.read_text(encoding="utf-8")).get("name", name)
        return PlacementAgent.from_json(path), name
    return load_rule_agent(spec), spec


def mini_piece(draw, piece, x0, y0):
    if not piece:
        return
    cells = SHAPES[piece][0]
    for dx, dy in cells:
        x, y = x0 + dx * MINI, y0 + (1 - dy) * MINI
        draw.rectangle([x, y, x + MINI - 2, y + MINI - 2], fill=COLORS[PIECE_ID[piece]])


def draw_panel(draw, font, engine, i, obs, name, ox, result):
    me = obs["self"]
    top = 40
    bx = ox + 56
    draw.text((bx, 12), name[:22], fill=TEXT, font=font)
    # 보드 테두리와 칸
    draw.rectangle([bx - 2, top - 2, bx + BOARD_W + 1, top + BOARD_H + 1], outline=GRID)
    for y in range(20):
        for x in range(10):
            v = int(me["board"][y][x])
            px, py = bx + x * CELL, top + (19 - y) * CELL
            draw.rectangle([px, py, px + CELL - 2, py + CELL - 2], fill=COLORS[v] if v else COLORS[0])
    # 떨어지는 중인 조각
    p = engine.players[i]
    if p.alive:
        for x, y in engine.cells(p):
            if 0 <= y < 20:
                px, py = bx + x * CELL, top + (19 - y) * CELL
                draw.rectangle([px, py, px + CELL - 2, py + CELL - 2], fill=COLORS[PIECE_ID[p.piece]])
    # 들어올 garbage 막대 (보드 왼쪽 빨간 막대)
    inc = min(int(me["incoming"]), 20)
    if inc:
        draw.rectangle([bx - 9, top + BOARD_H - inc * CELL, bx - 5, top + BOARD_H], fill=RED)
    # HOLD와 NEXT
    draw.text((ox + 6, top), "HOLD", fill=DIM, font=font)
    mini_piece(draw, me["hold"], ox + 10, top + 16)
    draw.text((bx + BOARD_W + 8, top), "NEXT", fill=DIM, font=font)
    for k, piece in enumerate(me["next"][:4]):
        mini_piece(draw, piece, bx + BOARD_W + 12, top + 16 + k * 26)
    # 아래쪽 수치
    draw.text((bx, top + BOARD_H + 8), f"lines {me['lines']}   attack {me['attack']}", fill=TEXT, font=font)
    if result:
        color = (90, 220, 120) if result == "WIN" else RED
        draw.rectangle([bx + 30, top + 130, bx + BOARD_W - 30, top + 170], fill=BG, outline=color)
        draw.text((bx + BOARD_W // 2 - 12, top + 144), result, fill=color, font=font)


def frame(engine, names, font, result=None):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    for i in range(2):
        obs = competition_observation(engine, i)
        res = None if result is None else ("WIN" if result == i else "LOSE")
        draw_panel(draw, font, engine, i, obs, names[i], 8 + i * (PANEL_W + 8), res)
    draw.text((WIDTH // 2 - 22, HEIGHT - 22), f"{engine.tick / 60:5.1f}s", fill=DIM, font=font)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", default="agents/my_agent/model.json")
    ap.add_argument("--opp", default="dellacherie", help="룰베이스 이름 또는 model.json 경로")
    ap.add_argument("--seed", type=int, default=5000001)
    ap.add_argument("--seconds", type=float, default=120, help="GIF에 담을 최대 게임 시간")
    ap.add_argument("--every", type=int, default=6, help="몇 tick마다 한 장 찍을지")
    ap.add_argument("--speed", type=float, default=2.0, help="재생 배속")
    ap.add_argument("--out", default="docs/gameplay.gif")
    args = ap.parse_args()

    agents, names = zip(*(load_agent(s) for s in (args.me, args.opp)))
    engine = DuelEngine(config=competition_config(), seed=args.seed)
    for i, agent in enumerate(agents):
        agent.reset(args.seed, i)
    font = ImageFont.load_default()
    frames = []
    max_ticks = int(args.seconds * 60)
    while not engine.done and engine.tick < max_ticks:
        if engine.tick % args.every == 0:
            frames.append(frame(engine, names, font))
        engine.step(tuple(Action(int(a.act(competition_observation(engine, i))))
                          for i, a in enumerate(agents)))
    final = frame(engine, names, font, engine.winner if engine.done else None)
    frames += [final] * 1                                   # 마지막 장면
    ms = max(20, int(1000 * args.every / 60 / args.speed))
    durations = [ms] * (len(frames) - 1) + [2500]           # 마지막 장면은 2.5초 멈춤
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)
    winner = "끝나지 않음(시간 제한)" if not engine.done else (names[engine.winner] if engine.winner is not None else "무승부")
    print(f"{out}  frames={len(frames)}  size={out.stat().st_size / 1e6:.1f}MB  "
          f"game={engine.tick / 60:.1f}s  winner={winner}")


if __name__ == "__main__":
    main()
