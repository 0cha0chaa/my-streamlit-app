
"""
AIVLE School 학습 도우미 (4차 미니프로젝트)
- 백서(PDF)를 업로드하면 rag_pipeline()으로 RAG 체인을 만들어 답변합니다.
- 답변 아래에 관련 유튜브 영상 1~2개를 '유사도 기준 이상일 때만' 보여줍니다.
- 사이드바 '영상 둘러보기' 탭에서 카테고리별 영상을 볼 수 있습니다.

필요 파일(GitHub 저장소 루트)
- app.py, requirements.txt
- videos.csv              : title, url, category, keywords, description
- AIVLE_School 백서.pdf    : (선택) 업로드 전 기본으로 사용할 백서
API 키: Streamlit Community Cloud > Settings > Secrets 에 OPENAI_API_KEY = "sk-..." 등록
"""

import os
import tempfile
import unicodedata

import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="🎓", layout="wide")

try:  # 배포 환경: Secrets에 등록한 키 사용
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # 로컬 실행 시 환경 변수에 등록된 키 사용

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PL_WCuvyChN3giYr6Xa2gWMOvRjDUrfb0a"
DEFAULT_DOC = "AIVLE School 백서.pdf"
VIDEO_CSV = "videos.csv"
CATEGORIES = ["지원 안내", "교육 과정", "후기", "취업", "기타"]
EMBED_MODEL = "text-embedding-3-small"

SAMPLE_QUESTIONS = [
    "비전공자도 따라갈 수 있나요?",
    "하루 공부 시간은 어느 정도가 적당한가요?",
    "프로젝트는 어떻게 준비해야 하나요?",
    "인적성 검사는 어떻게 준비하나요?",
    "AICE Associate는 어떤 시험인가요?",
]


def find_file(target, folder="."):
    """확장자 대소문자, 한글 자모 분리(NFD)와 상관없이 파일 찾기. 없으면 None"""
    norm = lambda x: unicodedata.normalize("NFC", x).lower()
    for f in os.listdir(folder):
        if norm(f) == norm(target):
            return os.path.join(folder, f)
    return None


# ------------------------------------------------------------
# 1. RAG 파이프라인 (미션 2 함수와 동일한 흐름)
# ------------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def rag_pipeline(filepath, k=4):
    # PDF 파일 읽기
    loader = PyMuPDF4LLMLoader(filepath)
    docs = loader.load()

    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " ", ""])
    chunks = splitter.split_documents(docs)

    # Embedding 모델
    embedding = OpenAIEmbeddings(model=EMBED_MODEL)

    # 벡터 저장소 만들기 (백서 전용 컬렉션)
    vectorstore = Chroma(collection_name="aivle_docs", embedding_function=embedding)

    # 기존 데이터 비우기 → 문서 추가
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)

    # 검색기 구성
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # LLM, 프롬프트
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    s_msg = """당신은 KT AIVLE School 교육생을 돕는 학습 도우미입니다.
1. 반드시 제공된 백서 내용(context)에 근거하여 답변하세요.
2. 백서에 없는 내용은 추측하지 말고 "백서에서 확인되지 않습니다. 운영 공지나 사무국에 확인해 주세요."라고 답하세요.
3. 답변은 친절하고 간결하게, 필요하면 목록으로 정리하세요.
[백서]
{context}"""
    prompt = ChatPromptTemplate.from_messages([("system", s_msg), ("human", "{input}")])

    # RAG 체인
    rag_chain = (
        {"context": itemgetter("input") | retriever | format_docs,
         "input": itemgetter("input")}
        | prompt | llm | StrOutputParser()
    )
    return rag_chain, len(chunks)


# ------------------------------------------------------------
# 2. 영상 목록과 영상 전용 벡터 저장소 (백서와 다른 컬렉션)
# ------------------------------------------------------------
@st.cache_data
def load_videos(path=VIDEO_CSV):
    path = find_file(path)
    if path is None:
        return pd.DataFrame(columns=["title", "url", "category", "keywords", "description"])
    df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    df = df[df["url"].str.startswith("http")]
    df.loc[~df["category"].isin(CATEGORIES), "category"] = "기타"
    return df.reset_index(drop=True)


@st.cache_resource
def get_video_store(n_rows: int):
    df = load_videos()
    if df.empty:
        return None
    docs = [
        Document(
            page_content=f"{r.title}\n키워드: {r.keywords}\n{r.description}",
            metadata={"title": r.title, "url": r.url, "category": r.category},
        )
        for r in df.itertuples()
    ]
    store = Chroma(
        collection_name="aivle_videos",
        embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
        collection_metadata={"hnsw:space": "cosine"},   # 관련도 점수 = 1 - 코사인 거리
    )
    store.reset_collection()
    store.add_documents(docs)
    return store


def find_related_videos(store, question, threshold, top_n=2):
    """유사도 기준(threshold) 이상인 영상만 최대 top_n개 반환"""
    if store is None:
        return []
    hits = store.similarity_search_with_relevance_scores(question, k=top_n)
    return [(doc, score) for doc, score in hits if score >= threshold]


# ------------------------------------------------------------
# 3. 세션 상태
# ------------------------------------------------------------
ss = st.session_state
ss.setdefault("messages", [])          # (role, text, videos)
ss.setdefault("chain", None)
ss.setdefault("doc_name", None)


def build_chain_from_path(filepath, name, k):
    with st.spinner(f"'{name}' 문서를 벡터화하는 중..."):
        ss.chain, n_chunks = rag_pipeline(filepath, k=k)
    ss.doc_name = name
    st.sidebar.success(f"{name} 준비 완료 (청크 {n_chunks}개)")


# ------------------------------------------------------------
# 4. 사이드바: 문서 / 영상 둘러보기 / 관리
# ------------------------------------------------------------
videos = load_videos()
video_store = get_video_store(len(videos))

with st.sidebar:
    st.image("https://raw.githubusercontent.com/Jangrae/img/master/aivle.png", width=140)
    tab_doc, tab_video, tab_admin = st.tabs(["📄 문서", "🎬 영상 둘러보기", "⚙️ 관리"])

    # --- 관리 탭 (다른 탭보다 먼저 값이 필요하므로 위에서 선언) ---
    with tab_admin:
        k = st.slider("백서 검색 문서 수(k)", 2, 8, 4)
        threshold = st.slider("관련 영상 표시 기준(유사도)", 0.0, 1.0, 0.45, 0.05,
                              help="이 값 이상으로 관련 있는 영상만 답변 아래에 보여줍니다.")
        if st.button("대화 기록 삭제", use_container_width=True):
            ss.messages = []
            st.rerun()
        st.caption(f"등록된 영상: {len(videos)}개")

    # --- 문서 탭 ---
    with tab_doc:
        uploaded = st.file_uploader("AIVLE School 백서(PDF) 업로드", type=["pdf"])
        if uploaded is not None and uploaded.name != ss.doc_name:
            # 업로드 파일은 경로가 없으므로 임시 파일로 저장 후 파이프라인에 전달
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.getvalue())
            build_chain_from_path(tmp.name, uploaded.name, k)
        elif ss.chain is None and find_file(DEFAULT_DOC):
            build_chain_from_path(find_file(DEFAULT_DOC), DEFAULT_DOC, k)

        if ss.doc_name:
            st.info(f"사용 중인 문서: {ss.doc_name}")
        else:
            st.warning("백서 PDF를 업로드해 주세요.")

    # --- 영상 둘러보기 탭 ---
    with tab_video:
        st.markdown(f"[▶ 유튜브 재생목록 전체 보기]({PLAYLIST_URL})")
        if videos.empty:
            st.caption("videos.csv가 비어 있습니다. make_videos_csv.py로 목록을 만들어 주세요.")
        else:
            counts = videos["category"].value_counts()
            options = [c for c in CATEGORIES if c in counts.index]
            category = st.radio("카테고리", options, format_func=lambda c: f"{c} ({counts[c]})")
            for r in videos[videos["category"] == category].itertuples():
                with st.expander(r.title):
                    st.video(r.url)
                    if r.description:
                        st.caption(r.description)
                    st.markdown(f"[유튜브에서 보기]({r.url})")


# ------------------------------------------------------------
# 5. 메인 화면: 채팅
# ------------------------------------------------------------
st.title("🎓 AIVLE School 학습 도우미")
st.caption("백서를 근거로 답하고, 관련 영상이 있으면 함께 보여드려요.")


def show_videos(video_hits):
    if not video_hits:
        return
    st.markdown("**🎬 관련 영상**")
    cols = st.columns(len(video_hits))
    for col, (doc, score) in zip(cols, video_hits):
        with col:
            st.video(doc.metadata["url"])
            st.caption(f"{doc.metadata['title']} · {doc.metadata['category']} · 관련도 {score:.2f}")


# 입력 받기 (추천 질문 버튼 클릭도 질문으로 처리)
user_input = st.chat_input("에이블스쿨에 대해 무엇이든 물어보세요.")

if not ss.messages and not user_input:
    st.markdown("**이런 걸 물어볼 수 있어요**")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, q in zip(cols, SAMPLE_QUESTIONS):
        if col.button(q, use_container_width=True):
            user_input = q

# 기존 대화 출력
for role, text, video_hits in ss.messages:
    with st.chat_message(role):
        st.markdown(text)
        if role == "assistant":
            show_videos(video_hits)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    if ss.chain is None:
        answer, video_hits = "먼저 사이드바 '문서' 탭에서 백서 PDF를 업로드해 주세요.", []
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        with st.chat_message("assistant"):
            answer = st.write_stream(ss.chain.stream({"input": user_input}))
            video_hits = find_related_videos(video_store, user_input, threshold)
            show_videos(video_hits)

    ss.messages.append(("user", user_input, []))
    ss.messages.append(("assistant", answer, video_hits))
