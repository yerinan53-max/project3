import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
for module_name in list(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]

from src.config import SAMPLE_DATA_PATH
from src.data import load_cases
from src.llm import build_legal_analysis_prompt
from src.search import expand_legal_query


def test_sample_cases_include_assault_demo_case():
    cases = load_cases(SAMPLE_DATA_PATH)
    assert len(cases) >= 50
    assert "지나가는 행인 폭행 상해 사건" in set(cases["case_name"])


def test_assault_query_expansion_for_demo():
    expanded = expand_legal_query("지나가는 행인을 폭행함")
    assert "행인 폭행" in expanded
    assert "치료비" in expanded


def test_llm_prompt_uses_retrieved_cases_only():
    cases = load_cases(SAMPLE_DATA_PATH).head(3)
    cases = cases.copy()
    cases.insert(0, "similarity", 0.75)
    prompt = build_legal_analysis_prompt("지나가는 행인을 폭행함", cases)
    assert "검색된 판례 정보만 근거" in prompt
    assert "지나가는 행인을 폭행함" in prompt
