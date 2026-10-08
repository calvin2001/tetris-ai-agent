import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # 대전 worker는 python -I 로 실행 → 같은 폴더 import에 필수

from agent_core import PlacementAgent  # noqa: E402


def create_agent():
    return PlacementAgent.from_json(HERE / "model.json")
