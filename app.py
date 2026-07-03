import pandas as pd
import streamlit as st

from src.analysis import category_counts, extract_issue_keywords
from src.config import INDEX_DATA_PATH, INDEX_PATH, default_case_data_path
from src.data import documents_from_cases, load_cases
from src.embedding import LegalEmbedder, resolve_model_name
from src.llm import generate_ollama_analysis
from src.search import load_index, semantic_search

st.set_page_config(
    page_title="LLM 기반 유사 판례 검색",
    page_icon="⚖",
    layout="wide",
)


@st.cache_resource
def get_embedder() -> LegalEmbedder:
    return LegalEmbedder()


@st.cache_data
def get_index() -> tuple[pd.DataFrame, object]:
    if INDEX_PATH.exists() and INDEX_DATA_PATH.exists():
        return load_index()

    cases = load_cases(default_case_data_path())
    embeddings = get_embedder().encode(documents_from_cases(cases))
    return cases, embeddings


if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "llm_analysis" not in st.session_state:
    st.session_state.llm_analysis = ""


st.title("LLM 기반 유사 판례 검색 및 법률 쟁점 요약 시스템")
st.caption("Sentence-BERT 임베딩 검색 결과를 Ollama LLM이 근거로 삼아 쟁점을 요약합니다.")

with st.sidebar:
    st.header("데이터 및 모델")
    st.text(f"모델: {resolve_model_name()}")
    try:
        indexed_cases, indexed_embeddings = get_index()
        using_saved_index = INDEX_PATH.exists() and INDEX_DATA_PATH.exists()
        st.success(f"검색 판례 {len(indexed_cases):,}건")
        if using_saved_index:
            st.info("학습 모델 검색 준비 완료")
        else:
            st.info("합성 데이터 공개 데모")
    except (ValueError, FileNotFoundError) as error:
        indexed_cases = None
        indexed_embeddings = None
        using_saved_index = False
        st.error(str(error))

    st.divider()
    st.header("LLM 설정")
    ollama_model = st.text_input("Ollama 모델", value="gemma2:2b")
    st.caption("시연용 권장: gemma2:2b / 더 정교하지만 느림: gemma2:9b")

if using_saved_index:
    st.info(
        "현재 로컬 실행 환경에서는 저장된 인덱스를 사용해 전체 판례 데이터로 검색합니다. "
        "GitHub/Streamlit 공개 배포판에는 데이터 이용 조건상 샘플 데이터만 포함됩니다.",
        icon="ℹ️",
    )
else:
    st.warning(
        "공개 시연 버전은 AI Hub 데이터 이용 조건을 준수하기 위해 합성 샘플 데이터로 동작합니다. "
        "전체 데이터 기반 실험 결과는 발표 자료에 별도로 제시했습니다.",
        icon="ℹ️",
    )

query = st.text_area(
    "사건 내용 또는 법률 쟁점",
    placeholder="예: 중고차를 무사고 차량이라고 해서 구매했는데 사고 이력이 발견되었다.",
    height=130,
)

col1, col2 = st.columns([3, 1])
with col1:
    search_clicked = st.button("유사 판례 검색", type="primary", use_container_width=True)
with col2:
    top_k = st.selectbox("검색 개수", [3, 5, 10], index=1)

if search_clicked:
    if indexed_cases is None or indexed_embeddings is None:
        st.warning("검색 데이터를 준비하지 못했습니다.")
    elif not query.strip():
        st.warning("검색할 내용을 입력하세요.")
    else:
        try:
            with st.spinner("의미가 유사한 판례를 검색하고 있습니다..."):
                results = semantic_search(
                    query,
                    indexed_cases,
                    indexed_embeddings,
                    get_embedder(),
                    top_k=top_k,
                )

            st.session_state.last_query = query
            st.session_state.last_results = results
            st.session_state.llm_analysis = ""
        except (ValueError, FileNotFoundError) as error:
            st.error(str(error))

if st.session_state.last_results is not None:
    results = st.session_state.last_results

    st.subheader("검색 결과")
    for rank, row in results.iterrows():
        title = (
            f"{rank + 1}. {row['case_name']} "
            f"· 검색점수 {row['similarity']:.1%}"
        )
        with st.expander(title, expanded=rank == 0):
            st.write(f"**의미 유사도:** {row['semantic_similarity']:.1%}")
            st.write(f"**법원/선고일:** {row['court']} / {row['decision_date']}")
            st.write(f"**사건번호:** {row['case_number']}")
            st.write(f"**분야:** {row['category']}")
            st.write(f"**핵심 쟁점:** {row['issues']}")
            st.write(f"**판결 요약:** {row['summary']}")
            if row["source_url"]:
                st.link_button("원문 확인", row["source_url"])

    st.subheader("검색 기반 쟁점 분석")
    analysis_left, analysis_right = st.columns(2)
    with analysis_left:
        st.write("**공통 쟁점 키워드**")
        keywords = extract_issue_keywords(results)
        st.write(", ".join(word for word, _ in keywords) or "추출 결과 없음")
    with analysis_right:
        st.write("**사건 분야 분포**")
        st.bar_chart(category_counts(results), x="category", y="count")

    st.divider()
    st.subheader("LLM 쟁점 요약")
    st.caption("LLM은 위 검색 결과를 근거로 요약합니다. 실제 법률 판단에는 원문 확인이 필요합니다.")

    if st.button("검색 결과 기반 쟁점 요약 생성", type="secondary"):
        try:
            with st.spinner("Ollama LLM이 검색된 판례를 읽고 쟁점을 정리하고 있습니다..."):
                st.session_state.llm_analysis = generate_ollama_analysis(
                    st.session_state.last_query,
                    results,
                    model=ollama_model.strip() or "gemma2:2b",
                )
        except RuntimeError as error:
            st.error(str(error))

    if st.session_state.llm_analysis:
        st.markdown(st.session_state.llm_analysis)

st.divider()
st.caption(
    "이 프로젝트는 과학기술정보통신부의 재원으로 한국지능정보사회진흥원의 지원을 받아 구축된 "
    "AI Hub 법률 데이터를 학습 목적으로 사용했습니다. 공개 데모에는 합성 샘플 데이터만 포함합니다."
)
