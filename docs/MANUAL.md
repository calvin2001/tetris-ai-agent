# 테트리스 1:1 AI 대회 참가 매뉴얼 (Step-by-Step)

> 기준: 2026-10-02 · 배포 키트 v2 · 대전 규칙 `local-sudden-death-v1`
> 이 문서의 코드와 수치는 배포된 키트로 **직접 실행해서 확인한 것**이다 (Linux, Python 3.12, CPU 1코어 환경).
> Windows CMD 명령을 기본으로 쓰고, macOS/Linux/Colab 차이는 따로 적었다.

---

## 0. 한눈에 보기

**남은 일정**

| 날짜 | 일정 |
|---|---|
| 9/24 (목) | 강화학습 세미나 (완료) |
| 추석 연휴 | DQN 자료 배포 (완료) |
| 10/1 (목) | PPO 세미나 (완료) |
| **10/8 (목)** | **모델 발표회 및 시상식 ← 최종 목표** |

**전략 한 줄 요약**

> 프레임 단위(매 tick 9개 버튼) 강화학습을 처음부터 하지 않는다.
> **① 배치 탐색 Agent → ② 유전 알고리즘(GA)으로 가중치 튜닝 → ③ (선택) 강화학습으로 가치함수 보정** 순서로 쌓는다.

이렇게 하는 이유는 세 가지다.

1. 운영진이 직접 밝혔듯이 실습 패키지(DQN, Offline DQN)는 실제 `CompetitionEnv`와 차이가 커서 **랜덤 Agent를 겨우 이기는 수준**이다.
2. 운영진 팁: **유전 알고리즘이 안정적인 출발점**이고, **GA + RL 계열이 성능이 좋았다.**
3. 실제로 확인해 보니, 룰베이스 Dellacherie에 **hold 판단 하나만 추가해도** 두 룰베이스를 모두 이긴다 (Step 3 결과 참고). 즉 "일단 이기는 Agent"는 첫날에 만들 수 있고, 남은 시간은 전부 개선에 쓸 수 있다.

**이 매뉴얼을 끝까지 따라가면 얻는 것**

- 두 룰베이스(dellacherie, codemyroad)를 이기는 제출 가능한 Agent
- GA 학습 스크립트, RL 보정 스크립트, 빠른 실험용 대전기
- 그대로 GitHub에 올릴 수 있는 저장소 구조

---

## 1. 받은 자료 정리: 무엇을 쓰고 무엇을 건너뛸지

| 자료 | 역할 | 이 매뉴얼에서 |
|---|---|---|
| `tetris_competition_kit_v2.zip` | 공식 대전 엔진(`CompetitionEnv`), 제출 템플릿, 검증·평가 도구 | **필수** — 학습·검증의 기준 |
| `local_tetris_battle_kit_v2.zip` | 브라우저 대전 화면, 룰베이스 모델 2개(dellacherie, codemyroad) | **필수** — 목표 상대 + 눈으로 관전 |
| `dqn_tetris_lab_v2.zip` | 단일 보드 DQN 입문 실습 | 선택 — 개념 이해용. 시간 없으면 건너뜀 |
| `tetris_offline_hybrid_extension_v1.zip` | 리플레이 기반 Offline DQN / CQL / BC | 선택 — 발표용 추가 실험 거리 |
| `tetris_replay_dataset_2024.zip` (약 132MB) | TETR.IO 상위권 리플레이 약 772만 배치 | 선택 — **GitHub에 올리면 안 됨** (§11) |
| `강화학습_세미나.pdf`, `924_DQN_실습.pdf` | 개념 정리, 운영진 팁 | 부록 A에서 개념과 코드를 연결 |

> 두 키트의 엔진 파일(`tetrio_env/engine.py` 등)은 서로 내용이 같다는 것을 비교해서 확인했다. 그래서 학습은 competition kit 엔진으로, 관전은 local battle kit 화면으로 해도 결과가 같다.

---

## 2. 꼭 알아야 할 규칙 (엔진 코드에서 직접 확인)

### 2-1. 행동과 관측

- 행동은 `0 NOOP, 1 LEFT, 2 RIGHT, 3 SOFT_DROP, 4 HARD_DROP, 5 CW, 6 CCW, 7 180, 8 HOLD`.
- 1 tick(1/60초)마다 행동 1개. **LEFT/RIGHT는 tick당 정확히 1칸** 움직인다(엔진에 DAS 처리가 없음). 그래서 "회전 → 좌우 n칸 → 하드드롭"은 약 n+2 tick이 걸린다.
- `observation["self"]["board"]`는 20×10, **0번 행이 바닥**이다. 0이 아니면 블록이 있다.
- `x, y`는 조각 로컬 박스의 왼쪽 아래 기준. 새 조각은 항상 `x=3, rotation=0`에서 나온다.
- `pieces`(놓은 조각 수)는 **조각이 고정될 때만** 증가한다. HOLD로는 바뀌지 않는다 → "pieces가 바뀌면 새 계획" 방식이 hold와도 잘 맞는다.
- `next`는 다음 5개, `hold`는 보관 조각(없으면 `None`), `incoming`은 들어올 garbage 줄 수 합계다.

### 2-2. 시간 압박 (서든 데스)

| 시간 | 공격 배율 | garbage 지연 | 1회 최대 적용 |
|---|---:|---:|---:|
| 0\~30초 | ×1.00 | 20 tick | 8줄 |
| 30\~60초 | ×1.25 | 15 tick | 10줄 |
| 60\~90초 | ×1.50 | 10 tick | 12줄 |
| 90\~120초 | ×1.75 | 5 tick | 14줄 |
| 120초\~ | ×2.00 | 0 tick | 16줄 |

중력은 20초부터 매초 0.035G씩 빨라진다(최대 20G). 경기는 최대 10분이다.

> ⚠️ `dqn_tetris_lab/RULES.md`에는 "60초 뒤 0.0035G, 최대 18,000 tick"이라고 적혀 있는데, 이는 **현재 키트와 다르다.** 숫자가 다르면 항상 `tetris_competition_kit_v2/RULES.json`을 믿는다.

### 2-3. 실행 제한 (`tetrio_env/agent_runtime.py`에서 확인)

| 항목 | 제한 | 어기면 |
|---|---|---|
| Agent 시작(`create_agent()` 포함) | **2초** | 시작 실패 = 패배 |
| `reset()` | 1초 | 패배 |
| `act()` 1회 | **50ms** (JSON 왕복 포함) | 패배 |
| 메모리 | 2GiB (Linux에서만 강제) | 프로세스 종료 = 패배 |

---

## 3. Step 1 — 저장소 뼈대 만들기

처음부터 GitHub에 올릴 구조로 만든다. 배포 키트는 `kits/` 안에 풀고, **내 코드만** `agents/`, `src/`에 둔다.

```text
tetris-ai-agent/
├── README.md                    # 프로젝트 소개 + 결과표 (§11 템플릿)
├── MANUAL.md                    # 이 문서
├── .gitignore
├── requirements.txt
├── kits/                        # ← 배포 ZIP 압축 해제 위치 (git에 올리지 않음)
│   ├── tetris_competition_kit_v2/
│   └── local_tetris_battle_kit_v2/
├── agents/
│   └── my_agent/                # ← 이 폴더가 그대로 "제출물"
│       ├── agent.py
│       ├── agent_core.py
│       ├── manifest.json
│       └── model.json
├── src/
│   ├── init_model.py            # model.json 초기화
│   ├── arena.py                 # 빠른 대결기 (학습용)
│   ├── ga_train.py              # 유전 알고리즘
│   └── rl_finetune.py           # 강화학습 보정 (선택)
├── results/                     # 실험 결과 (요약만 커밋)
└── dist/                        # 제출 ZIP (git에 올리지 않음)
```

**Windows CMD**

```bat
mkdir tetris-ai-agent
cd tetris-ai-agent
mkdir kits agents\my_agent src results dist
tar -xf ..\tetris_competition_kit_v2.zip -C kits
tar -xf ..\local_tetris_battle_kit_v2.zip -C kits
```

ZIP 파일 위치에 맞게 `..\` 부분은 바꾼다. `tar`가 없으면 파일 탐색기로 `kits\` 안에 압축을 푼다. 풀고 나면 `kits\tetris_competition_kit_v2\README.md`가 보여야 한다.

**macOS / Linux**

```bash
mkdir -p tetris-ai-agent/{kits,agents/my_agent,src,results,dist} && cd tetris-ai-agent
unzip -q ../tetris_competition_kit_v2.zip -d kits
unzip -q ../local_tetris_battle_kit_v2.zip -d kits
```

`.gitignore`를 지금 바로 만든다 (나중에 큰 파일이 실수로 올라가는 것을 막는다).

```gitignore
# 배포 키트·데이터: 용량과 재배포 문제로 커밋하지 않는다
kits/
data/
dist/
*.zip
*.xz

# 가상환경·캐시
.venv*/
__pycache__/
*.pyc

# 실험 산출물: 요약(README.md)만 커밋
results/*
!results/README.md
*.pt

# 비교용 임시 후보 Agent
agents/cand_*/
```

`requirements.txt`:

```text
numpy>=1.24
```

---

## 4. Step 2 — 설치하고 동작 확인하기

Python 3.10 이상이 필요하다. 저장소 최상위 폴더에서 실행한다.

**Windows CMD**

```bat
py -3 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e kits\tetris_competition_kit_v2
python kits\tetris_competition_kit_v2\training\env_smoke.py --episodes 3 --seed 1
```

PowerShell이라면 두 번째 줄 대신 `.venv\Scripts\Activate.ps1`을 쓴다. `py`가 없으면 `python`으로 바꾼다.
macOS/Linux는 `python3 -m venv .venv && source .venv/bin/activate` 후 경로의 `\`를 `/`로 바꾸면 된다.

마지막 줄에서 `[complete] episodes=3 ...`가 나오면 성공이다.

**목표 상대끼리 붙여 보기**

```bat
python kits\tetris_competition_kit_v2\tools\evaluate.py kits\local_tetris_battle_kit_v2\rule_based_tetris_models\codemyroad_genetic kits\local_tetris_battle_kit_v2\rule_based_tetris_models\dellacherie --games 6 --seed 20000
```

직접 돌린 결과는 `{"codemyroad_genetic": 0, "dellacherie": 6}`이었다. 한 경기는 대략 30\~40초(1,800\~2,300 tick)에 끝났다. **dellacherie가 더 강한 상대**라는 뜻이다.

**브라우저로 관전하기**

```bat
cd kits\local_tetris_battle_kit_v2
python main.py
```

브라우저가 열리면 두 룰베이스를 골라 경기를 본다. 종료는 `Ctrl-C`, 끝나면 `cd ..\..`로 저장소 최상위로 돌아온다.

> 시간이 남으면 `dqn_tetris_lab`의 실습 1(1,000 step 스모크 테스트, PyTorch 필요) 정도만 돌려 보며 replay buffer·target network·ε-greedy 개념을 확인한다. 성능 목적이라면 건너뛰어도 된다.

---

## 5. Step 3 — 첫 Agent: 배치 탐색 + hold

### 5-1. 아이디어

두 룰베이스는 "현재 조각을 놓을 수 있는 모든 (회전, x) 위치를 시뮬레이션하고, 놓은 뒤 보드(afterstate)를 특징 점수로 평가해서 최고 위치를 고른 다음, 회전·이동·하드드롭으로 실행"한다. 다만 **hold를 쓰지 않는다.**

그래서 첫 Agent는 같은 구조에 다음 한 가지만 더한다.

> 현재 조각의 최고 점수 vs hold 조각(비었으면 next[0])의 최고 점수를 비교해서, 더 좋으면 HOLD를 먼저 누른다.

이 점수 함수가 곧 세미나에서 배운 **V(s): 이 상태가 얼마나 좋은가**의 역할이고, "최고 점수 배치를 고른다"는 것이 **π → greedy(V)** 이다. 나중에 GA는 이 V의 가중치를 찾고, RL은 V를 경험으로 보정한다.

### 5-2. 코드

`agents/my_agent/agent_core.py` (순수 Python, 외부 라이브러리 없음 — 이유는 §5-4)

<details>
<summary>agent_core.py 전체 코드 펼치기</summary>

```python
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
            "tetris", "lines")

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


def features(rows, cleared, placed, eroded) -> dict[str, float]:
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
    }


class PlacementAgent:
    def __init__(self, weights: dict[str, float], use_hold: bool = True,
                 value: dict | None = None, beta: float = 0.0):
        self.weights = {k: float(weights.get(k, 0.0)) for k in FEATURES}
        self.use_hold = use_hold
        self.value = value          # RL 단계에서 학습한 선형 가치함수 V(afterstate)
        self.beta = float(beta)     # 점수 = GA 휴리스틱 + beta * V
        self.reset(0, 0)

    @classmethod
    def from_json(cls, path: str | Path) -> "PlacementAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["weights"], data.get("use_hold", True),
                   data.get("value"), data.get("beta", 0.0))

    def reset(self, seed: int, player: int) -> None:
        self.commands: list[int] = []
        self.last_pieces = -1

    def act(self, obs: dict) -> int:
        me = obs["self"]
        if not me["alive"]:
            return 0
        if me["pieces"] != self.last_pieces:          # 새 조각이 나오면 다시 계획
            self.last_pieces = me["pieces"]
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
                s = self.score(features(new, cleared, placed, eroded)) - (1e9 if topout else 0)
                key = (s, -abs(tx - start_x), -rot)
                if best is None or key > best[0]:
                    best = (key, rot, tx)
        return best

    def plan(self, me: dict) -> list[int]:
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
```

</details>

`agents/my_agent/agent.py`

```python
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # 대전 worker는 python -I 로 실행 → 같은 폴더 import에 필수

from agent_core import PlacementAgent  # noqa: E402


def create_agent():
    return PlacementAgent.from_json(HERE / "model.json")
```

`agents/my_agent/manifest.json` — `id`는 **영문 소문자·숫자·`_`·`-`만** 쓰고 팀 고유 값으로 바꾼다.

```json
{
  "id": "myteam_ga_v1",
  "name": "MyTeam GA v1",
  "kind": "rule",
  "description": "Placement search + hold, GA-tuned linear afterstate evaluation",
  "entry": "agent.py"
}
```

`src/init_model.py` — Dellacherie 가중치로 `model.json`을 만든다.

```python
"""agents/my_agent/model.json 을 Dellacherie 가중치 + hold 사용으로 초기화한다."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agents" / "my_agent"))
from agent_core import DELLACHERIE  # noqa: E402

target = ROOT / "agents" / "my_agent" / "model.json"
target.write_text(json.dumps({"weights": DELLACHERIE, "use_hold": True}, indent=2), encoding="utf-8")
print(target)
```

### 5-3. 실행·검증·평가

```bat
python src\init_model.py
python kits\tetris_competition_kit_v2\tools\validate_submission.py agents\my_agent
python kits\tetris_competition_kit_v2\tools\evaluate.py agents\my_agent kits\local_tetris_battle_kit_v2\rule_based_tetris_models\dellacherie --games 20 --seed 20000
python kits\tetris_competition_kit_v2\tools\evaluate.py agents\my_agent kits\local_tetris_battle_kit_v2\rule_based_tetris_models\codemyroad_genetic --games 20 --seed 20000
```

**직접 돌린 결과** (공식 `evaluate.py`, 같은 seed에서 좌우 자리 교대):

| 상대 | seed | 결과 |
|---|---|---|
| dellacherie | 20000 | **17승 3패** |
| dellacherie | 90000 | **18승 2패** |
| codemyroad_genetic | 20000 | **20승 0패** |

`act()` 응답 시간은 한 경기 전체에서 최대 약 6ms였다(제한 50ms). 여기까지 되면 **10/8 최소 목표는 달성**이고, 이 상태로 GitHub에 첫 커밋을 한다(§11).

### 5-4. 제출물 함정 정리 (직접 확인한 것)

1. **같은 폴더의 모듈 import가 그냥은 안 된다.** 대전 worker가 `python -I`(격리 모드)로 실행돼서, `agent.py` 옆의 `agent_core.py`를 `import`하면 `ModuleNotFoundError`가 난다. 위 `agent.py`처럼 `sys.path.insert(0, 폴더)`를 먼저 해야 한다.
2. **룰베이스 폴더를 그대로 복사해 제출하면 안 된다.** 그 `agent.py`는 상위 폴더의 `classic_core.py`를 읽기 때문에 단독으로는 깨진다.
3. **PyTorch는 추론에 쓰지 않는다.** 공식 서버 환경에 torch가 설치돼 있다는 보장이 없고(키트 의존성은 numpy뿐), 시작 제한이 2초다. 실습 키트의 export 도구들도 일부러 순수 Python 추론으로 만들어져 있다. 학습은 torch로 해도 되지만, 제출할 때는 가중치를 JSON으로 내보내 순수 Python으로 계산한다.
4. **`print()`는 보이지 않는다.** worker가 stdout을 가로채고 stderr는 버린다. 디버깅은 §6의 `arena.py`로 같은 프로세스 안에서 한다.
5. **모델 파일은 `Path(__file__).parent` 기준으로 읽는다.** 제출 폴더 밖 파일은 대전 서버에 없다.

---

## 6. Step 4 — 빠른 실험용 대결기 `arena.py`

공식 `evaluate.py`는 Agent마다 별도 프로세스를 띄우고 매 tick JSON을 주고받아서 느리다. 학습(수백\~수천 경기)에는 같은 프로세스 안에서 바로 엔진을 돌리는 대결기를 쓴다. 룰베이스 상대도 키트 원본을 그대로 불러온다.

`src/arena.py`

```python
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
```

내 환경(1코어)에서 한 경기는 약 2\~3.5초였다. 경기 시간의 대부분은 엔진이 아니라 Agent의 배치 탐색이다. **학습은 arena로, 최종 판정은 반드시 공식 `evaluate.py`로** 한다.

---

## 7. Step 5 — 유전 알고리즘으로 가중치 튜닝

### 7-1. 방법

- **개체** = 특징 11개의 가중치 묶음. 첫 개체는 Dellacherie 가중치, 나머지는 거기에 노이즈를 더해 만든다.
- **적합도** = 상대(dellacherie, codemyroad, **직전 세대 1등**)와 좌우 교대로 붙은 승률 + 공격량 차이(동점 처리용 아주 작은 값).
- **선택·교배·변이**: 상위 `elite`개를 남기고, 둘을 섞은 뒤(교배) 가우시안 노이즈(변이)를 더해 다음 세대를 만든다. 노이즈는 세대마다 조금씩 줄인다.

> "직전 세대 1등"을 상대로 넣는 이유: Step 3 Agent가 이미 룰베이스를 거의 다 이기기 때문에, 룰베이스만 상대하면 모든 개체가 승률 100% 근처에 몰려서 **누가 더 나은지 구분이 안 된다.** 실제로 작은 테스트에서 개체 대부분이 적합도 1.0 근처로 붙었다. 더 강한 상대가 있어야 선택이 의미를 가진다.

`src/ga_train.py`

<details>
<summary>ga_train.py 전체 코드 펼치기</summary>

```python
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
    rival = PlacementAgent(opp) if isinstance(opp, dict) else _opponent(opp)
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
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = {k: DELLACHERIE.get(k, 0.0) for k in FEATURES}
    population = [start] + [mutate(start, args.sigma, rng) for _ in range(args.pop - 1)]
    champion = start   # 직전 세대 1등도 상대로 넣어 '룰베이스는 다 이겨서 점수 차이가 안 나는' 문제를 막는다

    with Pool(args.workers) as pool:
        for gen in range(args.gens):
            t0 = time.time()
            # 세대마다 새 seed (같은 세대 안에서는 모두 같은 seed → 공정 비교)
            seeds = [rng.randrange(10**6) for _ in range(args.games // 2)]
            rivals = OPPONENTS + (champion,)
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
```

</details>

### 7-2. 실행

먼저 아주 작게 돌려서 에러가 없는지 본다.

```bat
python src\ga_train.py --pop 4 --gens 1 --games 2 --elite 2 --workers 2 --out results\ga_smoke
```

그다음 본 학습. `--workers`는 CPU 코어 수에 맞춘다(작업 관리자 → 성능 → CPU에서 "논리 프로세서" 확인).

```bat
python src\ga_train.py --pop 20 --gens 15 --games 4 --workers 8
```

**걸리는 시간 계산법**: 한 세대 경기 수 = `pop × 3(상대 수) × games`. 위 설정이면 세대당 240경기다. 한 경기 3초로 잡으면 1코어 약 12분, 8코어 약 1.5\~2분이므로 15세대는 8코어 기준 30분 정도다. 밤새 돌릴 때는 `--pop 30 --gens 40 --games 6`처럼 키운다.

> Colab: GA는 CPU 작업이라 GPU가 도움이 안 되고, 무료 Colab은 보통 2코어라 개인 노트북보다 느릴 수 있다. 쓴다면 `--workers 2`.

### 7-3. 결과 채택

학습이 끝나면 `results\ga\best_model.json`이 생긴다. **학습에 안 쓴 seed**로 공식 평가를 해서 Step 3보다 좋을 때만 바꾼다.

```bat
xcopy /E /I agents\my_agent agents\cand_ga
copy /Y results\ga\best_model.json agents\cand_ga\model.json
notepad agents\cand_ga\manifest.json
python kits\tetris_competition_kit_v2\tools\evaluate.py agents\cand_ga agents\my_agent --games 40 --seed 5000000
```

`notepad`에서 `id`를 `myteam_ga_cand`처럼 **다르게** 바꾼다. `evaluate.py`는 점수를 `id`로 집계하므로 두 Agent의 id가 같으면 점수가 한 칸에 합쳐져서 비교가 안 된다.

새 후보가 기존 Agent와의 맞대결에서 이기고, 두 룰베이스 상대 성적도 떨어지지 않았으면 채택한다.

```bat
copy /Y agents\cand_ga\model.json agents\my_agent\model.json
```

---

## 8. Step 6 — (선택) 강화학습으로 가치함수 보정 = "GA + RL"

### 8-1. 방법 (세미나의 evaluation → improvement 그대로)

1. **경험 수집**: 현재 정책(GA 점수 + β·V)으로 룰베이스와 대결한다. ε 확률(기본 5%)로 무작위 배치를 섞어 탐험한다 (ε-greedy).
2. **보상 계산**: 매 배치 사이에 보낸 공격 줄 수 × 0.05, 경기 끝에 승리 +1 / 패배 −1. 각 배치 이후의 할인 누적보상 G를 계산한다 (γ = 0.98).
3. **정책 평가**: 배치 후 보드 특징 → G를 예측하는 선형 가치함수 V를 ridge 회귀로 맞춘다.
4. **정책 개선**: 새 점수 = GA 점수 + β·V 로 다시 행동을 고른다. 1로 돌아가 반복한다.

β는 "학습한 V를 얼마나 믿을지"이다. β = 0이면 GA 그대로이고, 클수록 RL 쪽 판단이 커진다. GA 점수는 부호와 크기가 V와 달라서 β는 5\~50 범위에서 실험으로 정한다.

`src/rl_finetune.py` (학습 쪽만 numpy 사용, 제출물은 여전히 순수 Python)

<details>
<summary>rl_finetune.py 전체 코드 펼치기</summary>

```python
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
    result = match(me, load_rule_agent(opp), seed, seat)
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
    args = ap.parse_args()

    model = json.loads(Path(args.init).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1)
    with Pool(args.workers) as pool:
        for rnd in range(args.rounds):
            jobs = [(model, OPPONENTS[i % 2], rng.randrange(10**6), (i // 2) % 2, args.epsilon)
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
```

</details>

### 8-2. 실행과 채택 규칙

```bat
python src\rl_finetune.py --init agents\my_agent\model.json --rounds 5 --games 40 --beta 20 --workers 8 --out results\rl_b20
python src\rl_finetune.py --init agents\my_agent\model.json --rounds 5 --games 40 --beta 5 --workers 8 --out results\rl_b5
```

라운드마다 `round00_model.json`, `round01_model.json`…이 생긴다. §7-3과 같은 방법으로 후보 폴더를 만들어 **GA 버전과 40경기 이상 맞대결**하고, 이길 때만 채택한다.

> 솔직한 주의: 이 단계는 스크립트가 끝까지 돌아가는 것과, RL 결과 모델이 제출 검증을 통과하는 것까지만 확인했다. **성능이 좋아진다는 보장은 없다.** 그래서 반드시 GA 버전과 비교해서 채택 여부를 정한다. 좋아지지 않았더라도 "β별 결과표"는 발표 자료로 충분히 의미가 있다.

**다음 단계로 키우고 싶다면**: 선형 V 대신 작은 MLP(torch로 학습 → 가중치를 JSON으로 저장 → 순수 Python 행렬곱으로 추론), 몬테카를로 대신 TD 타깃 + replay buffer + target network(= DQN 슬라이드 내용)로 바꿀 수 있다. 오프라인 리플레이 데이터로 V를 미리 학습한 뒤 위 루프로 미세조정하는 것도 가능하다.

---

## 9. Step 7 — 평가 규칙 (운에 속지 않기)

- **항상 공식 `evaluate.py`로 최종 판정**한다. arena는 학습용이다.
- **좌우 교대 + 고정 seed**: `--games 40`이면 seed 20개 × 자리 2개다. 비교할 때는 두 후보를 같은 seed로 평가한다.
- **학습 seed와 평가 seed를 분리**한다. GA·RL은 0\~999,999 사이 seed를 쓰므로 평가는 `--seed 5000000`처럼 멀리 떨어진 값으로 한다.
- **경기 수가 적으면 승률은 크게 흔들린다** (95% 신뢰구간, Wilson):

| 결과 | 승률 | 실제로 가능한 범위 |
|---|---:|---|
| 17/20 | 85% | 약 64% \~ 95% |
| 34/40 | 85% | 약 71% \~ 93% |
| 85/100 | 85% | 약 77% \~ 91% |
| 11/20 | 55% | 약 34% \~ 74% |

"20경기에서 2\~3승 차이"는 운일 수 있다. 중요한 결정(최종 제출 모델 선택)은 100경기 이상으로 확인한다.

- **결과는 표로 남긴다** (`results/README.md`). 발표 자료와 GitHub README에 그대로 쓴다.

| 버전 | 바꾼 점 | vs dellacherie | vs codemyroad | vs 이전 버전 | 채택 |
|---|---|---:|---:|---:|---|
| v1 | Dellacherie + hold | 17/20 | 20/20 | — | ✅ |
| v2 | GA 15세대 | | | | |
| v3 | GA + RL β=20 | | | | |

---

## 10. Step 8 — 제출 ZIP 만들기 (형식이 두 가지다!)

**두 키트가 요구하는 ZIP 구조가 서로 다르다.** 직접 확인해 보니 로컬 배틀 키트는 최상단에 파일이 바로 있는 ZIP을 `ZIP must contain exactly one top-level Agent folder` 오류로 거부했다.

| 용도 | ZIP 안 구조 | 만드는 방법 |
|---|---|---|
| 로컬 배틀 키트 (`submissions/`에 넣기) | `myteam_ga_v1/agent.py` … **폴더 하나** | `make_submission_zip.py` |
| competition kit README 방식 | `agent.py`, `manifest.json` … **최상단에 바로** | `Compress-Archive` |

```bat
rem (1) 로컬 배틀 키트용: 폴더 구조, __pycache__ 자동 제외
python kits\local_tetris_battle_kit_v2\tetris\tools\make_submission_zip.py agents\my_agent dist\myteam_ga_v1.zip

rem (2) competition kit 방식: 최상단 파일 구조 (캐시 폴더를 먼저 지운다)
if exist agents\my_agent\__pycache__ rmdir /S /Q agents\my_agent\__pycache__
powershell -NoProfile -Command "Compress-Archive -Path .\agents\my_agent\* -DestinationPath .\dist\myteam_submission_flat.zip -Force"
```

**10/8 발표회에서 어떤 형식을 받는지 운영진에게 미리 물어보고**, 헷갈리면 두 개 다 준비한다.

로컬 배틀 화면에서 내 Agent와 붙여 보려면:

```bat
if not exist kits\local_tetris_battle_kit_v2\submissions mkdir kits\local_tetris_battle_kit_v2\submissions
copy dist\myteam_ga_v1.zip kits\local_tetris_battle_kit_v2\submissions\
cd kits\local_tetris_battle_kit_v2
python main.py
```

실행 로그에 `[registered] myteam_ga_v1.zip`이 나오고 모델 목록에 이름이 보이면 성공이다. **모델을 고친 뒤 다시 등록할 때**는 같은 id가 이미 있으면 덮어쓰지 않고 `[skipped]` 처리되므로, `kits\local_tetris_battle_kit_v2\tetris\submissions\myteam_ga_v1` 폴더를 지우거나 id를 `myteam_ga_v2`처럼 올린다.

제출 직전 체크리스트:

- [ ] `validate_submission.py`가 `[valid]`
- [ ] 두 룰베이스 상대로 학습에 안 쓴 seed 40경기 이상 평가
- [ ] `manifest.json`의 `id`가 소문자 고유값이고 `name`, `description` 수정
- [ ] 제출 폴더 밖 파일을 읽는 코드가 없음
- [ ] torch 등 무거운 import 없음

---

## 11. Step 9 — GitHub에 올리기

### 11-1. 올리면 안 되는 것

- **리플레이 데이터**: `data.csv.xz` 하나가 약 131.8MB라서 GitHub의 파일당 100MiB 제한을 넘어 **push가 거부된다.** 출처(Kaggle, `n3koasakura/tetr-io-top-players-replays`) 링크만 README에 적는다. 키트 안내에 따르면 라이선스는 MIT로 표기돼 있지만, 사용 전 Kaggle 페이지에서 최신 조건을 확인하라고 돼 있다.
- **배포 키트 원본**: 세미나 운영진이 만든 자료다. 공개 저장소에 올려도 되는지 **먼저 운영진에게 확인**하고, 그 전까지는 `kits/`를 `.gitignore`에 둔 채 "어디서 받아서 어디에 풀면 되는지"만 README에 적는다.
- 가상환경, 캐시, 대용량 체크포인트.

`agents/my_agent/` 안의 코드와 `model.json`(몇 KB)은 올려도 된다. 이 매뉴얼의 코드는 키트 엔진을 import해서 쓰고, 룰베이스 코드는 복사하지 않고 실행 시점에 키트에서 불러오도록 짜 두었다(`agent_core.py`의 조각 모양 표는 표준 SRS 좌표다).

### 11-2. 첫 업로드

GitHub 웹에서 빈 저장소(`tetris-ai-agent`, README 체크 해제)를 만든 뒤:

```bat
git init
git add .
git status
git commit -m "v1: placement search + hold agent, beats both rule-based baselines"
git branch -M main
git remote add origin https://github.com/<내아이디>/tetris-ai-agent.git
git push -u origin main
```

`git status`에서 `kits/`, `dist/`, `.venv/`가 **목록에 없는지** 꼭 확인한다.

버전마다 태그를 붙이면 "발표 때 낸 모델"을 정확히 되돌릴 수 있다.

```bat
git tag v1-hold
git push --tags
```

### 11-3. README.md 템플릿

```markdown
# Tetris 1v1 AI Agent

강화학습 세미나 테트리스 대회 출전 Agent.

## 방법
1. 배치 탐색: 현재/hold 조각의 모든 (회전, x) 배치를 시뮬레이션하고 afterstate 특징으로 평가
2. 유전 알고리즘: 특징 가중치 11개를 룰베이스 + 직전 세대 챔피언 상대로 진화
3. (선택) 강화학습: 몬테카를로 가치함수 V(afterstate)로 GA 점수를 보정 (GA + βV)

## 결과 (공식 evaluate.py, 좌우 교대, 학습에 쓰지 않은 seed)
| 버전 | vs dellacherie | vs codemyroad |
|---|---:|---:|
| v1 hold | 17/20 | 20/20 |

## 재현
1. 세미나 배포 키트를 `kits/`에 압축 해제 (MANUAL.md §3)
2. `pip install -r requirements.txt` 후 `pip install -e kits/tetris_competition_kit_v2`
3. `python src/init_model.py` → `python src/ga_train.py ...`

## 참고
- Dellacherie 특징: Thiery & Scherrer (2009), Building Controllers for Tetris
- 대전 엔진·룰베이스 Agent: 세미나 배포 키트 (저장소에 포함하지 않음)
```

---

## 12. 다음 개선 아이디어 (효과 대비 난이도 순)

1. **GA 더 오래, 더 많은 경기로** — 가장 쉽고 확실하다. `--games`를 늘리면 노이즈가 줄어든다.
2. **공격·garbage 인식 특징 추가** — 지금 점수는 보드 모양만 본다. `features()`에 "incoming × 지운 줄"(줄을 지우면 들어올 garbage를 상쇄), "B2B 유지 여부", "상대 높이" 같은 항목을 추가하고 GA로 가중치를 찾게 한다. 서든 데스로 시간이 갈수록 공격 배율이 커지므로 공격 효율의 가치가 크다.
3. **2수 앞보기 (next 조각까지)** — 단순하게 모든 조합을 보면 1수(약 6ms)의 30배 이상이라 50ms를 넘는다. 1수에서 상위 3\~5개만 남겨 다음 조각을 보는 beam 방식으로 하고, 반드시 최악 응답 시간을 측정한다.
4. **빠른 중력 대비** — 경기가 길어지면 이동 도중 조각이 쌓인 블록에 걸릴 수 있다. 매 tick `x`, `rotation`이 계획과 다르면 다시 계획하도록 `act()`를 고친다.
5. **RL 보정 고도화** — §8의 "다음 단계" 참고.
6. **T-spin** — 공격력은 크지만 회전 경로 탐색이 필요해서 6일 안에는 우선순위가 낮다.

한 번에 하나만 바꾸고, 바꿀 때마다 §9 표에 기록한다.

---

## 13. 추천 일정 (10/2 금 → 10/8 목)

| 날짜 | 할 일 | 끝났다는 기준 |
|---|---|---|
| 10/2 (금) | Step 1\~3: 저장소, 설치, 첫 Agent | 두 룰베이스 상대 승률 표, 첫 GitHub push |
| 10/3 (토) | Step 4\~5: GA 스모크 → 본 학습(밤새) | `results\ga\best_model.json` |
| 10/4 (일) | GA 결과 평가·채택, 개선 아이디어 2번 추가 후 GA 재실행 | v2 채택 여부 결정 |
| 10/5 (월) | (선택) Step 6: β = 5, 20, 50 비교 | β별 결과표 |
| 10/6 (화) | 최종 후보 100경기 평가, 응답 시간 점검, 제출 ZIP 2종 | 체크리스트 전부 ✅ |
| 10/7 (수) | 발표 자료(방법 + 결과표 + 실패한 실험), README 정리, `git tag` | 발표 리허설 |
| 10/8 (목) | 발표회 | 🎉 |

막히면 "지금 버전은 이미 제출 가능하다"는 점을 기억하고, 개선은 항상 별도 후보 폴더(`agents\cand_*`)에서 한다.

---

## 14. 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `[invalid] ... ModuleNotFoundError: No module named 'agent_core'` | worker가 `python -I`로 실행돼 같은 폴더가 import 경로에 없음 | `agent.py` 맨 위에서 `sys.path.insert(0, 폴더)` |
| `ZIP must contain exactly one top-level Agent folder` | 로컬 키트에 최상단 파일 구조 ZIP을 넣음 | `make_submission_zip.py`로 다시 만들기 (§10) |
| `[skipped] ...`, `Agent id already exists` | 같은 id가 이미 등록됨 | `tetris\submissions\<id>` 삭제 또는 id 변경 |
| `manifest id must use lowercase ...` | id에 대문자·공백 | 소문자·숫자·`_`·`-`만 |
| 브라우저에서 `MATCH FAILED` | 시작/추론 오류 (무승부 아님) | `validate_submission.py`로 먼저 확인, `kits\local_tetris_battle_kit_v2\tetris\artifacts\matches\<match-id>\summary.json`의 `error` |
| `agent ... timed out` | `act()` 50ms 또는 시작 2초 초과 | 무거운 import 제거, 탐색 폭 줄이기, arena로 최악 응답 시간 측정 |
| 맞대결 점수가 한 칸에 몰림 | 두 Agent의 `id`가 같음 | 후보의 `manifest.json` id 변경 |
| `print()`가 안 보임 | worker가 stdout을 가로챔 | `src\arena.py`에서 같은 프로세스로 실행해 디버깅 |
| Windows에서 GA 실행 시 `RuntimeError: ... bootstrapping phase` | multiprocessing의 main 보호 누락 | `if __name__ == "__main__":` 유지 (스크립트에 포함됨) |
| `'py'은(는) 내부 또는 외부 명령...이 아닙니다` | Python 런처 없음 | `py -3` → `python` |

---

## 부록 A. 세미나 개념 ↔ 이 매뉴얼의 코드

| 세미나 개념 | 이 매뉴얼에서 |
|---|---|
| 상태(state) | `observation`: 보드, 현재/hold/next 조각, incoming, combo, B2B |
| 행동(action) | 엔진은 9개 프레임 행동. Agent는 그 위에 "배치(회전 × x, hold 여부)"라는 큰 행동을 고르고 프레임 행동으로 풀어서 실행 |
| 보상(reward) | 엔진 기본: 공격 1줄당 +0.02, 승 +1, 패 −1. RL 단계: 공격 × 0.05 + 승패 ±1 |
| V(s) | afterstate 점수 (GA 가중합 + β·학습한 V) |
| 정책 π | V가 가장 큰 배치를 고르는 greedy 정책 |
| Evaluation → Improvement | RL 단계의 "V를 G에 맞춤 → 새 V로 행동 선택" 반복 |
| ε-greedy | RL 수집 단계의 5% 무작위 배치 |
| Replay buffer / Target network | 선형 몬테카를로 버전에는 필요 없음. MLP + TD로 키울 때 도입 (§8 다음 단계) |
| Offline RL, BC | 리플레이 데이터로 V를 사전학습하는 확장 아이디어 (선택) |
| 유전 알고리즘 (운영진 팁) | Step 5 `ga_train.py` |

## 부록 B. 빠른 명령 모음 (저장소 최상위, Windows CMD)

```bat
call .venv\Scripts\activate.bat
python kits\tetris_competition_kit_v2\tools\validate_submission.py agents\my_agent
python kits\tetris_competition_kit_v2\tools\evaluate.py agents\my_agent kits\local_tetris_battle_kit_v2\rule_based_tetris_models\dellacherie --games 40 --seed 5000000
python src\ga_train.py --pop 20 --gens 15 --games 4 --workers 8
python src\rl_finetune.py --init agents\my_agent\model.json --rounds 5 --games 40 --beta 20 --workers 8
python kits\local_tetris_battle_kit_v2\tetris\tools\make_submission_zip.py agents\my_agent dist\myteam_ga_v1.zip
```
