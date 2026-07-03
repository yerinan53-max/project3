# LLM 기반 유사 판례 검색 및 법률 쟁점 요약 시스템

기존 법률 문서 임베딩 검색 시스템을 LLM/RAG 구조로 확장한 프로젝트입니다. 사용자가 사건 내용을 입력하면 Sentence-BERT로 유사 판례를 검색하고, Ollama Gemma2 모델이 검색 결과를 근거로 사건 쟁점을 요약합니다.

## 주요 기능

- Sentence-BERT 기반 유사 판례 검색
- 검색 결과 Top-K를 LLM 프롬프트의 근거 자료로 사용
- Ollama Gemma2 기반 법률 쟁점 요약
- 유사 판례와 사용자 사건의 공통점/차이점 정리
- 공개 시연용 합성 판례 데이터 제공
- LLM/RAG 실험용 Jupyter Notebook 제공

## LLM 확장 방식

이 프로젝트는 LLM을 새로 대규모 학습시키는 방식이 아니라 RAG 구조를 사용합니다.

```text
사용자 사건 입력
→ Sentence-BERT 유사 판례 검색
→ 검색된 판례 Top-K를 프롬프트에 삽입
→ Ollama Gemma2가 쟁점 요약 생성
```

## 데이터 안내

실제 실험에는 AI Hub 법률 데이터가 사용되었으나, 데이터 이용 조건을 준수하기 위해 저장소에는 아래 파일을 포함하지 않습니다.

- AI Hub 원본 데이터
- 전처리된 CSV 데이터
- 학습된 모델 파일
- 임베딩 인덱스 파일

공개 시연 버전은 합성 샘플 데이터로 동작합니다.

## 실행 방법

1. Python 패키지 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ollama 모델 준비

```powershell
ollama pull gemma2:2b
```

3. Streamlit 실행

```powershell
streamlit run app.py
```

## 예시 질의

```text
지나가는 행인을 폭행함
중고차를 무사고 차량이라고 해서 구매했는데 사고 이력이 발견되었다
개인정보가 광고업체에 넘어갔다
```

## 프로젝트 구조

```text
.
├─ app.py
├─ data/
│  └─ sample_cases.csv
├─ notebooks/
│  └─ llm_rag_issue_summary.ipynb
├─ scripts/
│  └─ generate_demo_cases.py
├─ src/
│  ├─ llm.py
│  ├─ search.py
│  ├─ embedding.py
│  └─ data.py
└─ requirements.txt
```
