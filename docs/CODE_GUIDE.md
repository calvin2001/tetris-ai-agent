# 코드 설명서 (최종 v6 기준)

> 대상: `agents/my_agent/`(manifest.json, model.json, agent.py, agent_core.py), `src/ga_train.py`, `src/rl_finetune.py`
> 목표: 각 파일이 무슨 일을 하는지, 왜 그렇게 짰는지 남에게 설명할 수 있게 되는 것.
> RL 단계의 원리는 [`STEP6_RL_GUIDE.md`](STEP6_RL_GUIDE.md)에 따로 자세히 정리했다.

---

## 0. 전체 그림

```text
           ┌────────────────────┐
           │  manifest.json     │  "나는 누구고, 시작 파일은 agent.py야"  ← 대전 서버가 제일 먼저 읽음
           └─────────┬──────────┘
                     ▼
           ┌────────────────────┐
           │  agent.py          │  model.json을 읽어 Agent를 만들어 줌 (연결 파일)
           └─────────┬──────────┘
                     ▼
┌──────────────┐   ┌────────────────────┐
│ model.json   │──▶│  agent_core.py     │  두뇌: 놓을 자리를 찾고 버튼으로 바꿈
│ 가중치·설정   │   └────────────────────┘
└──────▲───────┘
       │ 더 좋은 숫자를 찾아서 덮어씀 (학습할 때만, 경기 중엔 안 쓰임)
┌──────┴──────────────────────────────┐
│ src/ga_train.py   가중치 진화        │
│ src/rl_finetune.py 가치함수 V 학습   │
└─────────────────────────────────────┘
```

한 문장 요약:

> **`agent_core.py`는 놓을 수 있는 모든 자리를 시뮬레이션하고(다음 조각까지 2수), 놓은 뒤 보드의 특징 16개에 가중치를 곱한 점수(+ 학습한 V)로 최고 자리를 고른다. 가중치는 `ga_train.py`가, V는 `rl_finetune.py`가 학습했다.**

---

## 1. manifest.json — 이름표

```json
{
  "id": "myteam_v6",
  "name": "MyTeam v6 Final",
  "kind": "rule",
  "description": "Placement search + hold, GA-tuned linear afterstate evaluation",
  "entry": "agent.py"
}
```

| 항목 | 의미 | 규칙 |
|---|---|---|
| `id` | Agent 고유 이름 | 영문 **소문자**·숫자·`_`·`-`. 다른 Agent와 겹치면 안 됨 |
| `name` | 화면 표시 이름 | 자유 |
| `kind` | 종류 표시 | 동작에 영향 없음 |
| `entry` | 서버가 실행할 파일 | 제출 폴더 **안**에 있어야 함 |

- `evaluate.py`는 점수를 `id`로 집계한다. 두 Agent의 `id`가 같으면 점수가 합쳐져서 비교가 안 된다. 그래서 후보와 기준선 폴더마다 `id`를 다르게 둔다.
- 로컬 배틀 키트는 같은 `id`가 이미 등록돼 있으면 새로 등록하지 않는다.

---

## 2. model.json — 점수 공식과 설정

```json
{
  "weights": { "landing_height": -8.165, "eroded_cells": 3.724, "...": "16개" },
  "use_hold": true,
  "value": { "bias": 0.3, "theta": ["11개"], "mean": ["11개"], "std": ["11개"] },
  "beta": 50.0,
  "search_depth": 2,
  "beam": 3,
  "ply1_weight": 1.0
}
```

| 항목 | 의미 | 어디서 왔나 |
|---|---|---|
| `weights` | 특징 16개의 가중치 | GA (v6: 대전 특징 GA) |
| `use_hold` | hold 사용 여부 | v1부터 `true` |
| `value`, `beta` | 학습한 가치함수 V와 반영 세기 | RL (v4에서 학습, β = 50) |
| `search_depth` | 1 = 현재 조각만, 2 = 다음 조각까지 | v5부터 2 |
| `beam` | 2수 앞보기에서 더 깊이 볼 1수 후보 수 | 3 (5는 나아지지 않음) |
| `ply1_weight` | 2수 점수에서 1수째 점수에 곱하는 값 | 1.0 (0이면 무너짐) |

**적혀 있지 않은 항목은 기본값을 쓴다.** 가중치가 없는 특징은 0, `search_depth`가 없으면 1이다. 그래서 오래된 `model.json`(v1\~v4)도 새 `agent_core.py`에서 예전과 **완전히 같게** 동작한다. 같은 seed에서 tick 수까지 같은 것을 확인했다.

### 2-1. 가중치의 뜻

놓은 뒤 보드의 특징 × 가중치를 모두 더한 것이 점수다. **음수 = 클수록 나쁨, 양수 = 클수록 좋음.**

**보드 모양 특징 11개**

| 특징 | 무엇을 세나 | 출발 | v6 |
|---|---|---:|---:|
| `landing_height` | 놓은 조각의 평균 높이 | −4.50 | −8.17 |
| `eroded_cells` | 지운 줄 수 × 그 줄에 들어간 이번 조각 칸 수 | 3.42 | 3.72 |
| `row_transitions` | 가로로 훑을 때 빈칸↔블록 전환 수 | −3.22 | −3.49 |
| `col_transitions` | 세로로 훑을 때 빈칸↔블록 전환 수 | −9.35 | −6.95 |
| `holes` | 위가 막힌 빈칸 수 | −7.90 | −6.25 |
| `wells` | 양옆이 막힌 홈의 깊이 (깊을수록 가중) | −3.39 | −2.71 |
| `aggregate_height` | 열 높이의 합 | 0 | 1.44 |
| `max_height` | 가장 높은 열 | 0 | −4.51 |
| `bumpiness` | 이웃 열 높이 차이 합 | 0 | 2.34 |
| `tetris` | 4줄 삭제면 1 | 0 | −2.39 |
| `lines` | 지운 줄 수 | 0 | −0.64 |

**대전 특징 5개** (v6에서 추가)

| 특징 | 무엇을 세나 | v6 |
|---|---|---:|
| `garbage_risk` | 줄을 못 지우면 이번에 올라올 수 있는 garbage 줄 수 | −0.21 |
| `danger` | garbage까지 올라왔을 때의 최고 높이 | 1.39 |
| `cancel` | 이번 공격으로 상쇄하는 garbage 줄 수 | −2.03 |
| `b2b_break` | B2B 중에 1\~3줄로 지워서 체인을 끊으면 1 | 1.73 |
| `attack` | 상쇄 후 상대에게 넘어가는 공격 | 2.35 |

**해석할 때 주의**: 서로 겹치는 특징이 많다. `danger`와 `max_height`는 둘 다 높이를 재고, `cancel`·`attack`·`lines`·`tetris`·`eroded_cells`·`aggregate_height`는 모두 줄 삭제와 엮여 있다. 이런 특징들은 한쪽이 +로 가면 다른 쪽이 −로 가서 서로 상쇄하기도 한다. 그래서 개별 숫자보다 **묶음의 방향**으로 읽는다. v6는 "높이를 강하게 억제(착지 −8.2, 최고 높이 −4.5) + 상쇄보다 공격 선호(attack +2.35, cancel −2.03)"로 읽힌다.

---

## 3. agent.py — 연결 파일

```python
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # 대전 worker는 python -I 로 실행 → 같은 폴더 import에 필수

from agent_core import PlacementAgent  # noqa: E402


def create_agent():
    return PlacementAgent.from_json(HERE / "model.json")
```

1. **`sys.path.insert`**: 대전 서버는 격리 모드(`python -I`)로 실행해서 같은 폴더 파일도 자동으로 import되지 않는다. 이 줄이 없으면 `ModuleNotFoundError`로 시작부터 패배한다.
2. **`create_agent()`**: 서버가 시작할 때 한 번 부른다. `model.json`을 이 파일 기준 경로로 읽는다.

---

## 4. agent_core.py — 두뇌

### 4-1. 파일 지도

```text
[상수]   W, H, FULL, SHAPES, FEATURES(16개), LINE_ATTACK, DELLACHERIE
[함수]   to_rows()          관측 보드 → 비트마스크 줄
         drop()             한 자리에 떨어뜨려 보기
         features()         놓은 뒤 보드의 특징 16개
         battle_features()  그중 대전 특징 5개
[클래스] PlacementAgent
           __init__ / from_json   설정 불러오기
           reset()                경기 시작 초기화
           set_context()          대전 상황(incoming, B2B, cap, 배율) 저장
           act()                  매 tick 호출 → 버튼 1개        ← 서버가 부르는 곳
           plan()                 1수 탐색 (search_depth = 1)
           plan_deep()            2수 앞보기 (search_depth = 2)
           candidates()           한 조각의 모든 자리와 점수
           best_for()             한 조각의 최고 자리 (1수 탐색용)
           to_commands()          고른 자리 → 버튼 목록
           score() / value_of()   특징 → 점수
```

**호출 순서 (v6)**: 서버 → `act()` → (새 조각이면) `set_context()` → `plan()` → `plan_deep()` → `candidates()` → `drop()` + `features()` + `score()`

### 4-2. 상수

- `W, H = 10, 24`: 가로 10칸, 높이는 보이는 20줄 + 조각이 나타나는 숨은 공간 4줄.
- `SHAPES`: 7개 조각 × 회전 4방향의 칸 좌표. 엔진과 같은 표준 SRS 좌표.
- `FEATURES`: 특징 이름 16개. **새 특징은 반드시 맨 뒤에 추가한다.** `model.json`의 V는 이 순서대로 θ를 저장하므로, 중간에 끼워 넣으면 순서가 밀려 V가 엉뚱한 특징에 곱해진다.
- `LINE_ATTACK = (0, 0, 1, 2, 4)`: 지운 줄 수별 기본 공격량. T-spin, combo, B2B 보너스는 반영하지 않은 근사값이다.

### 4-3. `to_rows()` — 보드를 비트마스크로

```python
rows = [sum(1 << x for x, cell in enumerate(row) if cell) for row in board]
```

한 줄을 정수 하나로 바꾼다. x번 칸에 블록이 있으면 x번째 비트가 1이다. "x번 칸이 찼나?"는 `(rows[y] >> x) & 1`, "줄이 꽉 찼나?"는 `rows[y] == FULL`(1023)로 한 번에 확인한다. 후보를 수백 개씩 시뮬레이션하므로 이런 빠른 계산이 50ms 제한을 지키는 데 중요하다.

### 4-4. `drop()` — 한 자리에 떨어뜨려 보기

```text
① 조각이 나타나는 높이(20)에 놓을 수 있나? → 안 되면 None
② 한 칸 아래가 비어 있는 동안 계속 내림 (하드드롭과 같음)
③ 보드에 조각 4칸을 새김
④ 꽉 찬 줄을 지우고 위에 빈 줄을 채움
⑤ 반환: (새 보드, 지운 줄 수, 조각 칸 위치, 지운 줄에 들어간 조각 칸 수, topout 여부)
```

"위에서 똑바로 떨어뜨린다"고 가정한다. 밀어 넣기나 T-spin 자리는 후보에 없다. 실제 경기에서 이 가정 때문에 계획과 다르게 놓인 경우는 3,820번 중 1번이었다.

### 4-5. `features()` — 보드 모양 특징 11개

| 특징 | 계산 | 직관 |
|---|---|---|
| `holes` | 각 열에서 높이 아래쪽의 빈칸 수 | 못 채우는 칸 |
| `row_transitions` | 줄마다 왼쪽 벽→칸들→오른쪽 벽 순으로 찬/빈 상태가 바뀌는 횟수 (벽은 찬 칸) | 줄이 듬성듬성할수록 큼 |
| `col_transitions` | 열마다 바닥→위로 같은 방식 (바닥과 천장은 찬 칸) | 블록-빈칸-블록이 섞일수록 큼 |
| `wells` | 양옆이 막힌 빈칸이 위로 이어지면 깊이 1, 2, 3…을 누적 | 깊이 3짜리 우물 = 1+2+3 = 6 |
| `landing_height` | 놓은 4칸의 (y+1) 평균 | 얼마나 높이 놓았나 |
| `eroded_cells` | 지운 줄 수 × 그 줄에 들어간 이번 조각 칸 수 | 이번 조각의 줄 삭제 기여 |
| `aggregate_height`, `max_height`, `bumpiness` | 열 높이의 합, 최댓값, 이웃 차이 합 | 높이와 울퉁불퉁함 |
| `tetris`, `lines` | 4줄 삭제 여부, 지운 줄 수 | 줄 삭제 |

`row_transitions`는 빈 줄도 2씩 센다. 모든 후보에 똑같이 더해지는 값이라 비교에는 영향이 없다.

### 4-6. `battle_features()` — 대전 특징 5개

```python
attack = LINE_ATTACK[cleared] * ctx["mult"]          # 서든 데스 배율 반영
cancel = min(incoming, attack)                       # 들어올 garbage를 먼저 상쇄
remaining = incoming - cancel
risk = min(remaining, ctx["cap"]) if cleared == 0 else 0   # 줄을 못 지우면 cap줄까지 올라옴
garbage_risk = risk
danger       = max(heights) + risk
cancel       = cancel
b2b_break    = 1 if (ctx["b2b"] > 0 and 1 <= cleared <= 3) else 0
attack       = attack - cancel                       # 상대에게 실제로 넘어가는 양
```

엔진 규칙을 그대로 옮긴 것이다. 공격은 내게 들어올 garbage를 **먼저 상쇄**하고, 남으면 상대에게 간다. garbage는 줄을 **못 지우고** 고정할 때만 최대 cap줄까지 올라온다.

`ctx`(대전 상황)가 없으면 5개 모두 0이다. 그래서 학습 스크립트처럼 상황 정보 없이 `features()`를 불러도 깨지지 않는다.

**한계**: `ctx["b2b"]`가 0이면 "B2B 없음"과 "어려운 삭제를 막 1번 함"을 구분하지 못한다. 그리고 2수째에도 1수째와 같은 상황(incoming 등)을 쓴다. 1수째 공격으로 incoming이 줄어드는 효과는 2수째에 반영하지 않는 근사다.

### 4-7. `PlacementAgent` 주요 함수

#### `set_context()` + `act()`

```python
def act(self, obs):
    me = obs["self"]
    if not me["alive"]:
        return 0
    if me["pieces"] != self.last_pieces:     # 새 조각이 나왔다
        self.last_pieces = me["pieces"]
        self.set_context(obs)                # incoming, b2b, cap, 공격 배율 저장
        self.commands = self.plan(me)        # 버튼 목록 작성
    return self.commands.pop(0) if self.commands else 0
```

`pieces`(놓은 조각 수)는 조각을 고정할 때만 오른다. HOLD로는 바뀌지 않아서, HOLD가 들어간 버튼 목록도 끊기지 않고 끝까지 실행된다.

#### `score()`

```python
s = sum(self.weights[k] * f[k] for k in FEATURES)       # GA 가중합
if self.value is not None and self.beta:
    s += self.beta * self.value_of(f)                   # + β × V
```

v6의 V는 v4에서 학습한 것이라 θ가 11개뿐이다. `value_of()`는 `zip`으로 짝을 맞추므로 새 특징 5개는 V에서 자동으로 무시된다.

#### `plan()` — 1수 탐색 (v1\~v4)

현재 조각의 최고 자리와, hold하면 나올 조각(보관함이 비면 `next[0]`)의 최고 자리를 비교해서, hold가 **더 클 때만** hold한다. 점수가 같으면 덜 움직이는 자리, 회전이 적은 자리를 고른다. `search_depth`가 2면 바로 `plan_deep()`으로 넘긴다.

#### `plan_deep()` — 2수 앞보기 (v5부터)

```text
① 현재 조각 + hold 조각의 모든 자리(약 68곳)에 점수 = 1수째 점수
② 점수 상위 beam(3)개만 남김
③ 각각의 놓은 뒤 보드에 "그다음 조각"의 모든 자리를 놓아 보고 최고 점수 = 2수째 점수
④ 1수째 점수 + 2수째 점수가 가장 큰 1수째 자리를 실행
```

그다음 조각은 hold 여부에 따라 달라진다.

| 1수째 선택 | 이번에 놓는 조각 | 그다음 조각 |
|---|---|---|
| hold 안 함 | 현재 조각 | `next[0]` |
| hold (보관함 빔) | `next[0]` | `next[1]` |
| hold (보관함 있음) | 보관함 조각 | `next[0]` |

- **왜 상위 3개만?** 68개 전부 2수까지 보면 약 2,300번 계산해서 50ms를 넘는다. 빔 3이면 약 170번, 최대 응답 약 15\~19ms다.
- **왜 1수째 점수를 더하나?** 2수째 점수만 쓰자 12전 0패였다. 첫 조각으로 줄을 지운 가점(`eroded_cells`)이 2수째 점수에는 없고 높이 합 감소만 남아서, 첫 조각으로는 줄을 안 지우려 했다.
- **결정한 뒤에는 첫 수만 실행**하고, 다음 조각이 나오면 다시 계산한다. 그사이 garbage가 올라오는 등 보드가 바뀔 수 있기 때문이다.

#### `to_commands()`

```python
cmds = [8] if use_hold else []                          # HOLD (hold하면 x=3, 회전 0에서 다시 시작)
cmds += {0: [], 1: [5], 2: [7], 3: [6]}[(rot - sr) % 4]  # 회전은 버튼 1개 (270°는 반시계 1번)
cmds += ([2] if tx > sx else [1]) * abs(tx - sx)        # 좌우 이동
return cmds + [4]                                       # 하드드롭
```

### 4-8. 한계

1. 똑바로 떨어뜨리는 자리만 본다. 밀어 넣기, T-spin은 불가능하다.
2. 2수째에서는 hold를 고려하지 않는다.
3. 공격량은 줄 수 기반 근사다. combo, B2B 보너스, T-spin은 빠져 있다.
4. 상대 보드 상태(상대 높이 등)는 보지 않는다.
5. V는 v4 시점에 학습한 것이라 v6 가중치와는 짝이 정확히 맞지 않는다.

---

## 5. ga_train.py — 가중치 진화

### 5-1. 흐름

```text
준비: 첫 개체 = --init의 가중치 (없으면 Dellacherie), 나머지 = 거기에 노이즈를 더한 변형
      챔피언 = 첫 개체
세대마다:
  ① seed 몇 개를 새로 뽑음 (같은 세대 안에서는 모두 같은 seed → 공정 비교)
  ② 모든 개체 × 상대(--rivals + 챔피언) × seed × 좌우 자리 → 경기 (병렬)
  ③ 적합도 = 승률 + 0.001 × 공격량 차이 → 정렬
  ④ 1등 기록 (log.jsonl, best_model.json), 챔피언 = 이번 1등
  ⑤ 상위 elite개는 그대로, 나머지는 교배 + 변이 (변이 크기는 세대마다 0.9배)
```

**GA 후보는 항상 1수 탐색, V 없음으로 경기한다.** 빠르게 많은 경기를 하기 위해서다. 찾은 가중치는 평가할 때 2수 앞보기와 V를 켜서 쓴다.

### 5-2. 상대 고르기 (`_job`)

```python
if isinstance(opp, dict):
    rival = PlacementAgent(opp)                 # 챔피언 (가중치 사전)
elif opp.endswith(".json"):
    rival = PlacementAgent.from_json(opp)       # 보관해 둔 이전 버전 Agent
else:
    rival = _opponent(opp)                      # 키트의 룰베이스
```

후보들이 룰베이스를 모두 이기게 되자, 룰베이스와의 경기(전체의 2/3)에서는 후보 간 차이가 나지 않았다. 그래서 `--rivals`로 보관해 둔 v3, v4를 상대로 썼다. 상대가 비슷한 실력이라 로그의 `best`가 0.6\~0.8 정도로 낮게 나오는데, 이게 정상이다.

### 5-3. 옵션

| 옵션 | 기본값 | 뜻 |
|---|---:|---|
| `--init` | 없음 | 출발 가중치 `model.json` (없으면 Dellacherie). 가중치만 읽고 V는 무시 |
| `--rivals` | dellacherie, codemyroad | 상대 목록. 룰베이스 이름 또는 `model.json` 경로. 챔피언은 자동 추가 |
| `--pop` | 20 | 세대당 개체 수 |
| `--gens` | 15 | 세대 수 |
| `--games` | 4 | 상대 1명당 경기 수 (짝수) |
| `--elite` | 4 | 그대로 살아남는 수 |
| `--sigma` | 1.0 | 첫 변이 크기. 새 특징을 0에서 키울 때는 0.8, 이미 좋은 값을 다듬을 때는 0.5를 썼다 |
| `--workers` | 4 | 병렬 프로세스 수 (논리 코어 12개에서 10 사용) |
| `--out` | `results/ga` | 결과 폴더 |

**경기 수 계산**: 세대당 경기 = 개체 수 × 상대 수 × games. v6를 만든 설정(`--pop 20`, 상대 3명, `--games 6`)은 세대당 360경기, 세대당 약 5분이었다.

### 5-4. 출력

```text
[gen 08] best=0.928 mean=0.607 283s
```

`best`는 1등의 적합도(≈ 승률), `mean`은 집단 평균이다. `best_model.json`은 **마지막 세대의 1등**이라 운이 섞여 있다. 그래서 채택은 반드시 공식 `evaluate.py`로 학습에 안 쓴 seed에서 100경기 이상 맞대결해서 정했다.

---

## 6. rl_finetune.py — 가치함수 V 학습

원리는 [`STEP6_RL_GUIDE.md`](STEP6_RL_GUIDE.md)에 자세히 있다. 요약하면 이렇다.

```text
라운드마다:
  ① 현재 정책 + ε-greedy(5%)로 --vs 상대들과 경기, 놓은 뒤 보드 특징 기록
  ② 보상: 공격 1줄당 +0.05, 승리 +1 / 패배 −1  →  리턴 G (γ = 0.98)
  ③ 특징 → G를 ridge 회귀로 맞춰 V를 구함 (정책 평가)
  ④ 점수 = GA 점수 + β × V로 다시 행동 (정책 개선)
```

| 옵션 | 뜻 |
|---|---|
| `--init` | 출발 모델 |
| `--vs` | 경험을 모을 상대 목록 (룰베이스 이름 또는 `model.json` 경로) |
| `--beta` | V 반영 세기. 측정해 보니 50이 결정의 약 5%를 바꾸고, 500은 학습이 무너짐 |
| `--rounds`, `--games` | 반복 횟수, 라운드당 경기 수 |
| `--epsilon` | 탐험 확률 (기본 0.05) |

RL은 v4에서 한 번 적용했고, v5·v6는 그 V를 그대로 물려받았다.

---

## 7. 발표용 1분 설명

> 저희 Agent는 **배치 탐색** 위에 학습을 쌓은 구조입니다. 조각이 나오면 놓을 수 있는 모든 자리를 시뮬레이션하고, 놓은 뒤 보드의 특징 16개에 가중치를 곱해 점수를 매깁니다. 다음 조각까지 2수를 내다보고, hold도 함께 비교합니다.
>
> 가중치는 **유전 알고리즘**으로 찾았습니다. 처음에는 룰베이스를 상대로, 나중에는 이전 버전의 저희 Agent를 상대로 진화시켰습니다. 그 위에 **강화학습**으로 "이 보드 뒤에 실제로 무슨 일이 있었나"를 배운 가치함수를 작게 더했습니다.
>
> 가장 큰 발견은 **숫자보다 정보가 중요하다**는 점이었습니다. 같은 11개 특징으로 GA를 더 돌린 후보는 이전 버전에게 졌지만, garbage와 공격량을 보는 대전 특징 5개를 넣자 200경기 중 112승으로 이겼습니다. 최종 버전은 두 룰베이스에게 모두 40전 전승, 처음 버전에게 100전 71승입니다.

---

## 8. 예상 질문과 답

**Q. 강화학습이 아니라 룰베이스 아닌가요?**
A. 자리를 찾는 틀은 규칙이지만, 판단 기준인 가중치 16개는 대결 결과로 학습했고(GA), 가치함수 V는 경험의 보상으로 학습했습니다(RL). 운영진이 언급한 "GA로 출발점을 잡고 RL로 보정"하는 구조입니다.

**Q. DQN은 썼나요?**
A. 그대로는 쓰지 않았습니다. 같은 가치 기반 방법을 테트리스에 맞게 afterstate 가치 V로 바꿔 적용했습니다. 배치 결과가 정해져 있어서 V(놓은 뒤 보드)를 알면 Q(s, a)를 아는 것과 같습니다. 다만 신경망, TD 목표, replay buffer, target network 대신 선형 함수와 몬테카를로 리턴을 썼습니다.

**Q. 왜 9개 버튼을 직접 학습하지 않았나요?**
A. 한 경기에 수천 번 판단해야 하고, 버튼 하나가 좋았는지는 한참 뒤에 드러나서 학습이 매우 어렵습니다. 운영진도 실습 코드가 랜덤을 겨우 이긴다고 했습니다. 배치 단위로 바꾸면 조각당 수십 개 후보 중 하나를 고르는 문제가 됩니다.

**Q. 50ms 제한은 지키나요?**
A. 2수 앞보기 포함 최대 약 15\~19ms였습니다. 추론은 순수 Python입니다.

**Q. RL은 효과가 있었나요?**
A. 작았습니다. v3 상대 240경기 54.6%였고, β를 고를 때 쓴 평가를 빼면 53.5%라 통계적으로 확실하지 않습니다. 그리고 β가 크면 오히려 학습이 무너졌습니다.

**Q. 가장 큰 약점은?**
A. T-spin과 밀어 넣기를 못 하고, 상대 보드를 보지 않으며, 공격량 계산이 combo와 B2B 보너스를 빼고 근사한다는 점입니다.

---

## 9. 셀프 체크

1. 서버가 가장 먼저 읽는 파일과 매 tick 부르는 함수는?
2. `model.json`에 `search_depth`가 없으면 어떻게 동작하나?
3. 새 특징을 `FEATURES`의 중간에 끼워 넣으면 왜 안 되나?
4. `cancel`과 `attack`은 각각 무엇을 세나?
5. 2수 앞보기에서 hold(보관함 빔)를 선택하면 그다음 조각은?
6. 2수째 점수만 쓰면 왜 무너졌나?
7. GA 후보는 경기할 때 2수 앞보기를 쓰나?
8. `--rivals`를 도입한 이유는?

<details>
<summary>정답</summary>

1. `manifest.json` / `act()`
2. 1수 탐색으로 동작한다 (기본값 1).
3. V의 θ가 특징 순서대로 저장돼 있어서, 순서가 밀리면 V가 엉뚱한 특징에 곱해진다.
4. `cancel`: 이번 공격으로 상쇄하는 garbage 줄 수. `attack`: 상쇄하고 남아서 상대에게 넘어가는 공격.
5. `next[1]` (이번에 `next[0]`을 놓으므로)
6. 첫 조각으로 줄을 지운 가점(`eroded_cells`)이 빠지고 높이 합 감소만 남아서, 첫 조각으로 줄을 안 지우려 했다.
7. 아니다. 빠르게 많이 경기하려고 1수 탐색, V 없음으로 한다. 평가 때 켠다.
8. 후보들이 룰베이스를 모두 이겨서 그 경기들에서는 차이가 나지 않았다. 비슷한 실력의 이전 버전을 상대로 써야 선택이 의미가 생긴다.

</details>
