"""배치 탐색 Agent 코어: 순수 Python, 외부 라이브러리 없음.

1) 현재 조각(그리고 hold 조각)의 모든 (회전, x) 배치를 시뮬레이션
2) 배치 후 보드(afterstate)의 특징을 가중합해서 점수화
3) 최고 점수 배치를 [HOLD?] + 회전 + 좌우 이동 + HARD_DROP 프레임 행동으로 변환
"""
from __future__ import annotations

import json
from pathlib import Path

W, H = 10, 24          # 가시 20행 + 스폰 여유 4행
FULL = (1 << W) - 1

# tetrio_env/pieces.py 와 같은 좌표계 (x 오른쪽, y 위쪽, 0번 행 = 바닥)
SHAPES = {
    "I": (((0, 1), (1, 1), (2, 1), (3, 1)), ((2, 0), (2, 1), (2, 2), (2, 3)),
          ((0, 2), (1, 2), (2, 2), (3, 2)), ((1, 0), (1, 1), (1, 2), (1, 3))),
    "J": (((0, 1), (0, 0), (1, 0), (2, 0)), ((1, 2), (2, 2), (1, 1), (1, 0)),
          ((0, 1), (1, 1), (2, 1), (2, 0)), ((1, 2), (1, 1), (0, 0), (1, 0))),
    "L": (((2, 1), (0, 0), (1, 0), (2, 0)), ((1, 2), (1, 1), (1, 0), (2, 0)),
          ((0, 1), (1, 1), (2, 1), (0, 0)), ((0, 2), (1, 2), (1, 1), (1, 0))),
    "O": (((1, 1), (2, 1), (1, 0), (2, 0)),) * 4,
    "S": (((1, 1), (2, 1), (0, 0), (1, 0)), ((1, 2), (1, 1), (2, 1), (2, 0)),
          ((1, 2), (2, 2), (0, 1), (1, 1)), ((0, 2), (0, 1), (1, 1), (1, 0))),
    "T": (((1, 1), (0, 0), (1, 0), (2, 0)), ((1, 2), (1, 1), (2, 1), (1, 0)),
          ((0, 1), (1, 1), (2, 1), (1, 0)), ((1, 2), (0, 1), (1, 1), (1, 0))),
    "Z": (((0, 1), (1, 1), (1, 0), (2, 0)), ((2, 2), (1, 1), (2, 1), (1, 0)),
          ((0, 2), (1, 2), (1, 1), (2, 1)), ((1, 2), (0, 1), (1, 1), (0, 0))),
}

FEATURES = ("landing_height", "eroded_cells", "row_transitions", "col_transitions",
            "holes", "wells", "aggregate_height", "max_height", "bumpiness",
            "tetris", "lines",
            # 대전 특징 (v6에서 추가, 반드시 맨 뒤에 둔다: 기존 model.json의 V 순서 보호)
            "garbage_risk", "danger", "cancel", "b2b_break", "attack")

LINE_ATTACK = (0, 0, 1, 2, 4)   # 지운 줄 수 → 기본 공격량 (T-spin, combo, B2B 보너스 제외)

DELLACHERIE = {"landing_height": -4.500, "eroded_cells": 3.418, "row_transitions": -3.218,
               "col_transitions": -9.349, "holes": -7.899, "wells": -3.386}


def to_rows(board) -> list[int]:
    """관측 board(20x10, 0=빈칸) → 행별 비트마스크 리스트(길이 H)."""
    rows = [sum(1 << x for x, cell in enumerate(row) if cell) for row in board]
    return rows + [0] * (H - len(rows))


def drop(rows, cells, tx):
    """(회전된 cells, 왼쪽 기준 x=tx)로 하드드롭. 불가능하면 None."""
    def ok(oy):
        for dx, dy in cells:
            x, y = tx + dx, oy + dy
            if x < 0 or x >= W or y < 0 or y >= H or (rows[y] >> x) & 1:
                return False
        return True

    oy = H - 4
    if not ok(oy):
        return None
    while ok(oy - 1):
        oy -= 1
    new = rows[:]
    placed = [(tx + dx, oy + dy) for dx, dy in cells]
    for x, y in placed:
        new[y] |= 1 << x
    full = {y for y, r in enumerate(new) if r == FULL}
    eroded = sum(y in full for _, y in placed)
    new = [r for y, r in enumerate(new) if y not in full] + [0] * len(full)
    topout = any(y >= 20 for _, y in placed)
    return new, len(full), placed, eroded, topout


def features(rows, cleared, placed, eroded, ctx=None) -> dict[str, float]:
    """ctx: 대전 상황 {"incoming", "cap", "b2b", "mult"}. 없으면 대전 특징은 모두 0."""
    heights = []
    for x in range(W):
        h = 0
        for y in range(19, -1, -1):
            if (rows[y] >> x) & 1:
                h = y + 1
                break
        heights.append(h)
    holes = sum(1 for x in range(W) for y in range(heights[x]) if not (rows[y] >> x) & 1)
    row_t = 0
    for y in range(20):
        prev = 1
        for x in range(W):
            cur = (rows[y] >> x) & 1
            row_t += cur != prev
            prev = cur
        row_t += not prev
    col_t = wells = 0
    for x in range(W):
        prev, depth = 1, 0
        for y in range(20):
            cur = (rows[y] >> x) & 1
            col_t += cur != prev
            prev = cur
            left = x == 0 or (rows[y] >> (x - 1)) & 1
            right = x == W - 1 or (rows[y] >> (x + 1)) & 1
            if not cur and left and right:
                depth += 1
                wells += depth
            else:
                depth = 0
        col_t += not prev
    return {
        "landing_height": sum(y + 1 for _, y in placed) / 4.0,
        "eroded_cells": float(cleared * eroded),
        "row_transitions": float(row_t),
        "col_transitions": float(col_t),
        "holes": float(holes),
        "wells": float(wells),
        "aggregate_height": float(sum(heights)),
        "max_height": float(max(heights)),
        "bumpiness": float(sum(abs(a - b) for a, b in zip(heights, heights[1:]))),
        "tetris": float(cleared == 4),
        "lines": float(cleared),
        **battle_features(heights, cleared, ctx),
    }


def battle_features(heights, cleared, ctx) -> dict[str, float]:
    """들어올 garbage, 상쇄, B2B, 공격량을 반영한 특징."""
    if not ctx:
        return {"garbage_risk": 0.0, "danger": 0.0, "cancel": 0.0, "b2b_break": 0.0, "attack": 0.0}
    attack = LINE_ATTACK[cleared] * ctx["mult"]
    incoming = ctx["incoming"]
    cancel = min(incoming, attack)                    # 이번 공격으로 막는 garbage 줄 수
    remaining = incoming - cancel
    # 줄을 못 지우고 고정하면 남은 garbage가 최대 cap줄까지 올라올 수 있음
    risk = float(min(remaining, ctx["cap"])) if cleared == 0 else 0.0
    return {
        "garbage_risk": risk,
        "danger": float(max(heights)) + risk,         # garbage까지 올라왔을 때의 최고 높이
        "cancel": float(cancel),
        "b2b_break": 1.0 if (ctx["b2b"] > 0 and 1 <= cleared <= 3) else 0.0,
        "attack": float(attack - cancel),             # 상대에게 실제로 넘어가는 공격
    }


class PlacementAgent:
    def __init__(self, weights: dict[str, float], use_hold: bool = True,
                 value: dict | None = None, beta: float = 0.0,
                 depth: int = 1, beam: int = 3, ply1_weight: float = 1.0):
        self.weights = {k: float(weights.get(k, 0.0)) for k in FEATURES}
        self.use_hold = use_hold
        self.value = value          # RL 단계에서 학습한 선형 가치함수 V(afterstate)
        self.beta = float(beta)     # 점수 = GA 휴리스틱 + beta * V
        self.depth = int(depth)     # 1 = 현재 조각만, 2 = 다음 조각까지 앞보기
        self.beam = int(beam)       # 2수 앞보기에서 1수 상위 몇 개만 더 깊이 볼지
        self.ply1_weight = float(ply1_weight)  # 2수 점수 = 2수째 점수 + ply1_weight * 1수째 점수
        self.reset(0, 0)

    @classmethod
    def from_json(cls, path: str | Path) -> "PlacementAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["weights"], data.get("use_hold", True),
                   data.get("value"), data.get("beta", 0.0),
                   data.get("search_depth", 1), data.get("beam", 3),
                   data.get("ply1_weight", 1.0))

    def reset(self, seed: int, player: int) -> None:
        self.commands: list[int] = []
        self.last_pieces = -1
        self.ctx = None

    def set_context(self, obs: dict) -> None:
        sd = obs.get("sudden_death") or {}
        me = obs["self"]
        self.ctx = {"incoming": int(me.get("incoming", 0)), "b2b": int(me.get("b2b", 0)),
                    "cap": int(sd.get("garbage_cap", 8)),
                    "mult": float(sd.get("attack_multiplier", 1.0))}

    def act(self, obs: dict) -> int:
        me = obs["self"]
        if not me["alive"]:
            return 0
        if me["pieces"] != self.last_pieces:          # 새 조각이 나오면 다시 계획
            self.last_pieces = me["pieces"]
            self.set_context(obs)
            self.commands = self.plan(me)
        return self.commands.pop(0) if self.commands else 0

    def value_of(self, f: dict[str, float]) -> float:
        v = self.value
        return v["bias"] + sum(t * (f[k] - m) / s for k, t, m, s
                               in zip(FEATURES, v["theta"], v["mean"], v["std"]))

    def score(self, f: dict[str, float]) -> float:
        s = sum(self.weights[k] * f[k] for k in FEATURES)
        if self.value is not None and self.beta:
            s += self.beta * self.value_of(f)
        return s

    def best_for(self, rows, piece, start_x):
        best = None
        for rot in ((0,) if piece == "O" else range(4)):
            cells = SHAPES[piece][rot]
            for tx in range(-min(c[0] for c in cells), W - max(c[0] for c in cells)):
                r = drop(rows, cells, tx)
                if r is None:
                    continue
                new, cleared, placed, eroded, topout = r
                s = self.score(features(new, cleared, placed, eroded, self.ctx)) - (1e9 if topout else 0)
                key = (s, -abs(tx - start_x), -rot)
                if best is None or key > best[0]:
                    best = (key, rot, tx)
        return best

    def candidates(self, rows, piece):
        """한 조각의 모든 배치: [(점수, 회전, x, 놓은 뒤 보드)]"""
        out = []
        for rot in ((0,) if piece == "O" else range(4)):
            cells = SHAPES[piece][rot]
            for tx in range(-min(c[0] for c in cells), W - max(c[0] for c in cells)):
                r = drop(rows, cells, tx)
                if r is None:
                    continue
                new, cleared, placed, eroded, topout = r
                s = self.score(features(new, cleared, placed, eroded, self.ctx)) - (1e9 if topout else 0)
                out.append((s, rot, tx, new))
        return out

    @staticmethod
    def to_commands(use_hold, rot, tx, sx, sr):
        cmds = [8] if use_hold else []
        if use_hold:
            sx, sr = 3, 0
        cmds += {0: [], 1: [5], 2: [7], 3: [6]}[(rot - sr) % 4]
        cmds += ([2] if tx > sx else [1]) * abs(tx - sx)
        return cmds + [4]

    def plan_deep(self, me: dict) -> list[int]:
        """2수 앞보기: 1수 상위 beam개만 골라, 다음 조각의 최고 배치까지 보고 고른다."""
        rows = to_rows(me["board"])
        nxt = me["next"] or []
        sx, sr = int(me["x"]), int(me["rotation"])
        follow = nxt[0] if nxt else None
        opts = [(s, False, rot, tx, new, follow)
                for s, rot, tx, new in self.candidates(rows, me["piece"])]
        if self.use_hold:
            if me["hold"]:
                hp, hf = me["hold"], follow
            elif nxt:
                hp, hf = nxt[0], (nxt[1] if len(nxt) > 1 else None)
            else:
                hp = hf = None
            if hp:
                opts += [(s, True, rot, tx, new, hf)
                         for s, rot, tx, new in self.candidates(rows, hp)]
        if not opts:
            return [4]
        opts.sort(key=lambda o: o[0], reverse=True)
        best = None
        for s1, hold, rot, tx, new, f in opts[:self.beam]:
            if s1 <= -1e8 or f is None:
                total = s1
            else:
                s2 = max((c[0] for c in self.candidates(new, f)), default=-1e9)
                total = s2 + self.ply1_weight * s1
            key = (total, not hold, -abs(tx - (3 if hold else sx)), -rot)
            if best is None or key > best[0]:
                best = (key, hold, rot, tx)
        _, hold, rot, tx = best
        return self.to_commands(hold, rot, tx, sx, sr)

    def plan(self, me: dict) -> list[int]:
        if self.depth >= 2:
            return self.plan_deep(me)
        rows = to_rows(me["board"])
        cur = self.best_for(rows, me["piece"], int(me["x"]))
        alt_piece = me["hold"] or (me["next"][0] if me["next"] else None)
        alt = self.best_for(rows, alt_piece, 3) if (self.use_hold and alt_piece) else None
        use_hold = alt is not None and (cur is None or alt[0][0] > cur[0][0])
        best = alt if use_hold else cur
        if best is None:
            return [4]
        _, rot, tx = best
        cmds = [8] if use_hold else []
        sx, sr = (3, 0) if use_hold else (int(me["x"]), int(me["rotation"]))
        cmds += {0: [], 1: [5], 2: [7], 3: [6]}[(rot - sr) % 4]
        cmds += ([2] if tx > sx else [1]) * abs(tx - sx)
        return cmds + [4]
