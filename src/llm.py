from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request

import pandas as pd


OLLAMA_URL = "http://localhost:11434/api/generate"


def build_case_context(results: pd.DataFrame, max_cases: int = 3) -> str:
    """Convert top search results into compact evidence for the LLM."""
    lines: list[str] = []
    for index, row in results.head(max_cases).iterrows():
        lines.append(
            "\n".join(
                [
                    f"[판례 {index + 1}]",
                    f"사건명: {row.get('case_name', '')}",
                    f"분야: {row.get('category', '')}",
                    f"법원/선고일: {row.get('court', '')} / {row.get('decision_date', '')}",
                    f"사건번호: {row.get('case_number', '')}",
                    f"유사도: {float(row.get('similarity', 0.0)):.1%}",
                    f"핵심 쟁점: {row.get('issues', '')}",
                    f"판결 요약: {row.get('summary', '')}",
                ]
            )
        )
    return "\n\n".join(lines)


def build_legal_analysis_prompt(query: str, results: pd.DataFrame) -> str:
    context = build_case_context(results)
    return f"""
너는 법률 문서 검색 결과를 설명하는 교육용 AI 보조자다.
아래의 검색된 판례 정보만 근거로 사용하고, 판례나 법 조항을 새로 지어내지 마라.
확실하지 않은 내용은 "검색 결과만으로는 단정하기 어렵다"고 말해라.
법률 자문이 아니라 수업 프로젝트용 쟁점 정리임을 전제로 답변해라.

[사용자 사건]
{query}

[검색된 유사 판례]
{context}

다음 형식으로 한국어로 간결하게 정리해라.

1. 사건 핵심 요약
2. 주요 법률 쟁점
3. 유사 판례와의 공통점
4. 유사 판례와의 차이점 또는 한계
5. 추가로 확인하면 좋은 사실관계
6. 한 줄 결론
""".strip()


def generate_ollama_analysis(
    query: str,
    results: pd.DataFrame,
    model: str = "gemma2:2b",
    temperature: float = 0.2,
    timeout: int = 300,
) -> str:
    prompt = build_legal_analysis_prompt(query, results)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 700,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except socket.timeout as error:
        raise RuntimeError(
            "Ollama 응답 시간이 너무 오래 걸려 중단되었습니다. "
            "시연용으로는 `ollama pull gemma2:2b`를 받은 뒤 앱의 Ollama 모델을 "
            "`gemma2:2b`로 설정하는 것을 권장합니다."
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Ollama 서버에 연결할 수 없습니다. 먼저 터미널에서 "
            "`ollama serve`를 실행하고, 사용할 모델을 `ollama pull gemma2:2b`로 "
            "받았는지 확인하세요."
        ) from error

    parsed = json.loads(body)
    if "error" in parsed:
        raise RuntimeError(parsed["error"])
    return str(parsed.get("response", "")).strip()
