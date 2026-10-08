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
