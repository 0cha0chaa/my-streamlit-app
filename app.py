
"""
AIVLE School 학습 도우미 (4차 미니프로젝트)
- 백서(PDF)를 업로드하면 rag_pipeline()으로 RAG 체인을 만들어 답변합니다.
- 표는 행마다 열 이름을 붙여 읽고, 검색된 조각의 소제목 전체(표 포함)를 근거로 사용합니다.
- 답변 아래에 관련 유튜브 영상 1~2개를 '유사도 기준 이상일 때만' 보여줍니다.
- 사이드바 '영상 둘러보기' 탭에서 카테고리별 영상을 볼 수 있습니다.

필요 파일(GitHub 저장소 루트)
- app.py, requirements.txt
- videos.csv              : title, url, category, keywords, description
- AIVLE_School_백서.pdf   : (선택) 업로드 전 기본으로 사용할 백서
API 키: Streamlit Community Cloud > Settings > Secrets 에 OPENAI_API_KEY = "sk-..." 등록
"""

import os
import tempfile
import unicodedata
import re

import pymupdf

import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
DEFAULT_DOC = "AIVLE_School_백서.pdf"
VIDEO_CSV = "videos.csv"
CATEGORIES = ["지원 안내", "교육 과정", "후기", "취업", "기타"]
EMBED_MODEL = "text-embedding-3-small"

SAMPLE_QUESTIONS = [
    "비전공자도 따라갈 수 있나요?",
    "하루 공부 시간은 어느 정도가 적당한가요?",
    "프로젝트는 어떻게 준비해야 하나요?",
    "인적성 검사는 어떻게 준비하나요?",
    "AICE Associate는 어떤 시험인가요?",
    "졸업생들이 취업한 현황을 알려줘",
]


def find_file(target, folder="."):
    """확장자 대소문자, 한글 자모 분리(NFD)와 상관없이 파일 찾기. 없으면 None"""
    norm = lambda x: unicodedata.normalize("NFC", x).lower()
    for f in os.listdir(folder):
        if norm(f) == norm(target):
            return os.path.join(folder, f)
    return None


# ------------------------------------------------------------
# 1. RAG 파이프라인 (고도화: 표 인식 + 소제목 단위 검색)
# ------------------------------------------------------------

# (1) PDF 읽기: 본문과 표를 따로 읽어서 표는 행마다 '열 이름: 값'으로 바꾸기
def _clean(c):
    return re.sub(r"\s+", " ", (c or "")).strip()

def table_to_lines(tab, prev_header=None):
    """표 → 행마다 '열 이름: 값 / 열 이름: 값' 한 줄 (행이 혼자 떨어져도 뜻이 통하게)"""
    rows = [[_clean(c) for c in r if _clean(c)] for r in tab.extract()]   # 병합 셀로 생긴 빈 칸 제거
    rows = [r for r in rows if r]
    out, header, last = [], None, None
    # 페이지를 넘어 이어진 표(머리행 없음)는 앞 표의 머리행을 이어 씀
    if rows and prev_header and tab.bbox[1] < 120 and len(rows[0]) == len(prev_header) and rows[0] != prev_header:
        header = prev_header
    for r in rows:
        if header is None:
            if len(r) >= 2:
                header = r                      # 2칸 이상인 첫 행 = 머리행
            else:
                out.append(r[0])                # 머리행 앞의 제목·인용 문장
            continue
        if len(r) == 1 and last is not None:    # 다음 줄로 넘어간 셀 → 앞 행에 이어 붙이기
            out[last] += " " + r[0]
            continue
        if len(r) == len(header):
            out.append("[표] " + " / ".join(f"{h}: {v}" for h, v in zip(header, r)))
        else:
            out.append("[표] " + " / ".join(r))
        last = len(out) - 1
    if header is not None and last is None:     # 머리행만 있는 1행짜리 표(숫자 카드 등)
        out.append(" / ".join(header))
    return out, header


def load_pdf_with_tables(filepath, top=50, bottom=50):
    """PyMuPDF로 본문과 표를 따로 읽어 읽는 순서대로 합치기 (OCR 사용 안 함)"""
    pdf = pymupdf.open(filepath)
    docs, prev_header = [], None
    for pno, page in enumerate(pdf):
        H = page.rect.height
        tabs = page.find_tables().tables
        tab_boxes = [pymupdf.Rect(t.bbox) for t in tabs]
        items = []
        for t in tabs:
            lines, prev_header = table_to_lines(t, prev_header)
            if lines:
                items.append((t.bbox[1], "\n".join(lines)))
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
            r = pymupdf.Rect(x0, y0, x1, y1)
            if y1 < top or y0 > H - bottom:                    # 머리글·바닥글
                continue
            if any(r.intersects(b) for b in tab_boxes):         # 표 안의 글자는 위에서 처리
                continue
            t = re.sub(r"[ \t]+", " ", text).strip()
            if t:
                items.append((y0, t))
        items.sort(key=lambda x: x[0])
        docs.append(Document(page_content="\n".join(t for _, t in items),
                             metadata={"source": filepath, "page": pno + 1}))
    return docs


# (2) 소제목 단위로 묶기
HEAD = re.compile(r"^\s*(■.*|\d{1,2}\)\s.*|0\d\s+.*|R E F E R E N C E S|C O N T E N T S)$")   # ■ 소제목, 1) 제목, 01 장제목
SKIP = re.compile(r"^(목차|C O N T E N T S|R E F E R E N C E S|참고 자료)")                      # 검색에서 뺄 부분

def split_by_section(docs, chunk_size=500, chunk_overlap=100):
    """페이지 문서 → 소제목 단위로 묶기 → 청크마다 [소제목] 붙이기"""
    sections, title, buf = [], "AIVLE School 백서", []

    def flush():
        text = "\n".join(buf).strip()
        if text and not SKIP.match(title):
            sections.append(Document(page_content=text, metadata={"section": title}))

    for d in docs:
        for line in d.page_content.split("\n"):
            if HEAD.match(line) and len(line.strip()) < 60:
                flush()
                buf, title = [], line.strip()
            buf.append(line)
    flush()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ".", " ", ""])
    chunks = splitter.split_documents(sections)
    for c in chunks:
        c.page_content = f"[{c.metadata['section']}]\n{c.page_content}"
    return chunks, sections


# (3) RAG 파이프라인
def rag_pipeline(filepath, k=6):
    # PDF 파일 읽기 (표 인식)
    docs = load_pdf_with_tables(filepath)

    # 텍스트 분할 (소제목 단위 + 소제목 붙이기)
    chunks, sections = split_by_section(docs)

    # 소제목별 전체 내용(표 포함) 모아 두기
    section_text = {}
    for s in sections:
        section_text.setdefault(s.metadata["section"], []).append(s.page_content)
    section_text = {t: "\n".join(v) for t, v in section_text.items()}

    def expand_to_sections(found):
        """검색된 조각이 속한 소제목 전체(표 포함)를 근거로 넘기기"""
        seen, out = set(), []
        for d in found:
            t = d.metadata["section"]
            if t in seen:
                continue
            seen.add(t)
            out.append(f"[{t}]\n{section_text.get(t, d.page_content)[:3000]}")
        return "\n\n".join(out)

    # Embedding 모델, 벡터 저장소 (백서 전용 컬렉션)
    embedding = OpenAIEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(collection_name="aivle_docs", embedding_function=embedding)
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)

    # 검색기
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 질문 → 백서 용어로 바꾼 검색 문장 (쿼리 재작성)
    llm_rewrite = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    rewrite_prompt = ChatPromptTemplate.from_template(
        """다음 질문을 'AIVLE School 백서'에서 검색하기 좋은 한 줄 문장으로 바꾸세요.
- 백서에서 쓰는 말로 바꿉니다. 예) 졸업생→수료생, 어디 취업했나/취업 현황→취업후기·진출 기업·직무·취업 성과,
  돈·비용→교육비·자기부담금·훈련장려금, 입학시험→역량심사·인적성, 결석→출석·출결 규정
- 관련 키워드 3~6개를 덧붙이고, 검색 문장 한 줄만 출력하세요.
질문: {input}""")
    rewriter = rewrite_prompt | llm_rewrite | StrOutputParser()

    # 답변 LLM, 프롬프트 (미션 4: 역할 지정)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    s_msg = """당신은 KT AIVLE School 10기 선배 에이블러이자 학습 도우미 '에이블'입니다.
에이블스쿨 지원을 고민하는 사람과 교육생이 용기를 내 도전할 수 있도록 친근하고 정확하게 답합니다.

[답변 원칙]
1. 반드시 아래 [백서] 내용에 근거해 답하세요. 여러 곳에 흩어진 정보는 하나로 모아 정리하세요.
2. 근거에 '[표]'로 시작하는 줄이 있으면 표의 모든 행을 빠짐없이 마크다운 표로 보여 주세요.
3. 백서에 없는 내용은 추측하지 말고 "백서에서 확인되지 않아요. 공식 홈페이지(aivle.kt.co.kr)나 운영 사무국에 확인해 주세요."라고 답하세요.
4. 답변 끝에 근거가 된 소제목을 '📌 출처: [소제목]' 형식으로 적으세요.
5. 마지막 줄에는 짧은 응원 한 마디와, 이어서 물어보면 좋을 질문 1개를 제안하세요.

[백서]
{context}"""
    prompt = ChatPromptTemplate.from_messages([("system", s_msg), ("human", "{input}")])

    # RAG 체인: 검색은 재작성한 문장으로, 답변은 원래 질문으로
    rag_chain = (
        {"context": rewriter | retriever | expand_to_sections,
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
        k = st.slider("백서 검색 문서 수(k)", 2, 10, 6)
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
