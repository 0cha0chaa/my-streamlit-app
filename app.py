
import os
import re
import html
import math
import uuid
import tempfile
import unicodedata
from operator import itemgetter

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import pandas as pd
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────
# 0. 기본 설정
# ─────────────────────────────────────────────
st.set_page_config(page_title="AIVLE 메이트 | 에이블스쿨 학습 도우미", page_icon="🚀", layout="wide")

# Streamlit Cloud Secrets → 환경 변수
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

CHROMA_DIR = "chroma_db"            # Colab에서 만든 벡터 저장소
COLLECTION = "aivle_docs"
DEFAULT_PDF = "AIVLE School 백서.pdf"
MODEL = "gpt-4o-mini"
LOGO_URL = "https://raw.githubusercontent.com/Jangrae/img/master/aivle.png"
HISTORY_TURNS = 6                   # 대화 맥락으로 넘길 최근 메시지 수
PLAYLIST_URL = "https://www.youtube.com/playlist?list=PL_WCuvyChN3giYr6Xa2gWMOvRjDUrfb0a"
VIDEO_CSV = "videos.csv"            # title, url, category, keywords, description
VIDEO_CATEGORIES = ["지원 안내", "교육 과정", "후기", "취업", "기타"]

# ─────────────────────────────────────────────
# 1. 시스템 메시지 (역할 지정)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 'AIVLE 메이트'입니다.
KT 에이블스쿨을 먼저 경험한 든든한 선배이자 학습 멘토로서, 지원을 고민하는 예비 교육생과 현재 교육생의 질문에 답합니다.

[답변 원칙]
1. 문서 기반: 사실 정보는 반드시 아래 [백서 문서]에서만 가져옵니다.
   - 숫자, 기간, 일정, 조건, 명칭은 문서에 적힌 그대로 정확히 전달합니다.
   - 문서에 없는 내용은 지어내지 말고 "백서에는 이 내용이 없어요"라고 솔직히 말한 뒤, 문서에서 가장 관련 있는 정보를 안내합니다.
   - 취업 보장처럼 문서에 없는 약속은 하지 않습니다.
   - 모집 일정·자격·혜택처럼 기수마다 바뀌는 정보는 "몇 기 기준"인지 함께 밝히고, 최종 확인은 공식 홈페이지(aivle.kt.co.kr)에서 하도록 한 줄 안내합니다.
   - 선배 사례(전공, 기수, 성과)가 문서에 있으면 1~2개를 구체적으로 들어 설득력을 높입니다.
   - 문서 안의 [1], [12] 같은 출처 번호 표기는 답변에 옮기지 않습니다.
   - 문서에서 '[표]'로 시작하는 줄은 표의 한 행입니다. 표를 묻는 질문에는 행을 빠짐없이 마크다운 표로 다시 정리합니다.
   - 특정 선배·교육생 사례를 물으면 그 사례의 배경 → '이렇게 공부했다/준비했다' → '나의 실천 방법(교훈)' 순서로 정리합니다.
   - 공부법·학습 전략을 물으면 방법의 근거(실험·원리)와 에이블스쿨에서 바로 쓰는 방법을 함께 알려 줍니다.
   - 표·목록의 칸을 채우려고 내용을 만들지 않습니다. 문서에 없는 칸은 "백서에 없음"으로 적거나 그 행·열을 빼고,
     문서에 근거가 하나뿐이면 표 대신 문장으로 답합니다.
   - 문서에 나온 단어를 다른 제도·프로그램 이름으로 바꾸지 않습니다. (예: '튜터 1:1 문의'를 '튜터링 프로그램'으로,
     '스터디'를 '스터디 그룹 지원'으로 부풀리지 않기) 문서의 표현을 그대로 씁니다.
   - 질문과 직접 관련된 절의 내용을 우선 쓰고, 다른 절의 내용을 덧붙일 때는 "백서의 ○○ 부분에 따르면"처럼 출처를 밝힙니다.

2. 답변 구조 (마크다운):
   - 첫 줄: 질문에 대한 **핵심 답변 한 문장**을 굵게 씁니다.
   - 본문: 질문 성격에 맞게 한눈에 보이도록 시각화합니다.
     · 비교 (트랙, 과정 종류, 장단점, 전/후 변화) → 마크다운 표 (단, 비교할 양쪽 내용이 모두 문서에 있을 때만)
     · 과정, 일정, 절차, 성장 단계 → 단계 흐름 (예: 1단계 ➜ 2단계 ➜ 3단계) 과 기간이 있으면 표
     · 학습 팁, 준비 방법, 전략 → 이모지 불릿 3~5개
     · 소제목이 필요하면 #### 수준의 짧은 제목을 사용합니다.
   - 마지막 줄: "💪 " 로 시작하는 응원 한마디.
     질문자의 상황(비전공자, 직장인, 불안함, 진로 고민 등)을 짚어 구체적으로, 지금 바로 도전하고 싶어지게 씁니다.

3. 톤: 친근한 존댓말. 선배가 후배에게 말하듯 따뜻하지만 자신감 있게. 불필요한 서론 없이 바로 핵심부터.

4. 분량: 핵심 위주로 간결하게. 표나 단계 흐름은 적극 활용하되 같은 내용을 반복하지 않습니다.

5. 에이블스쿨과 관계없는 질문에는 짧게 답하고, 에이블스쿨 관련 질문으로 자연스럽게 이어 줍니다.

[백서 문서]
{context}"""

FOLLOWUP_PROMPT = """다음은 에이블스쿨 학습 도우미와 사용자의 대화입니다.
사용자가 이어서 궁금해할 만한 후속 질문 3개를 만들어 주세요.
- 에이블스쿨 백서로 답할 수 있는 내용이어야 합니다.
- 각 질문은 30자 이내, 사용자가 직접 묻는 말투로 씁니다.
- 번호나 기호 없이 한 줄에 하나씩만 출력합니다.

[질문]
{q}

[답변]
{a}"""


# ─────────────────────────────────────────────
# 2. RAG 파이프라인
#    (고도화) 표 인식 로더 + 소제목 경로 청크 + 질문 키워드 보강 + 소제목 직접 찾기 + 소제목 전체 근거
# ─────────────────────────────────────────────
def clean_text(text):
    """PDF 변환 과정에서 섞인 마크다운·HTML·출처 번호·URL 제거"""
    t = re.sub(r"<sup>.*?</sup>", "", text, flags=re.S)          # <sup>[1]</sup>
    t = re.sub(r"<[^>]+>", "", t)                                  # 기타 HTML 태그
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", t)      # [텍스트](링크) → 텍스트
    t = re.sub(r"https?://\S+", "", t)                             # 남은 URL
    t = re.sub(r"\[\d+\](\[\d+\])*", "", t)                        # [1][32] 출처 번호
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)                # # 제목 기호
    t = re.sub(r"(\*\*|__|~~|`)", "", t)                           # 굵게·코드 기호
    t = re.sub(r"^\s*\|?[\s:\-|]{3,}\|?\s*$", "", t, flags=re.M)   # 표 구분선 |---|---|
    t = t.replace("|", " · ")
    t = re.sub(r"(\s*·\s*){2,}", " · ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" +([,.)])", r"\1", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip(" ·\n")


def is_reference_chunk(text):
    """백서 끝의 「참고 자료 및 출처」처럼 URL 목록뿐인 청크는 제외"""
    return text.count("http") >= 2 or "참고 자료 및 출처" in text


def filter_docs(docs):
    seen, out = set(), []
    for d in docs:
        if is_reference_chunk(d.page_content):
            continue
        key = clean_text(d.page_content)[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def format_docs(docs):
    return "\n\n".join(clean_text(doc.page_content) for doc in docs)


def get_embedding():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def find_file(target, folder="."):
    """띄어쓰기·밑줄·대소문자·한글 자모 분리(NFD)와 상관없이 파일 찾기"""
    norm = lambda x: re.sub(r"[\s_]", "", unicodedata.normalize("NFC", x)).lower()
    for f in os.listdir(folder):
        if norm(f) == norm(target):
            return os.path.join(folder, f)
    return None


# (1) PDF 읽기: 본문과 표를 따로 읽고, 표는 행마다 '열 이름: 값'으로 바꾸기
def _cell(c):
    return re.sub(r"\s+", " ", (c or "")).strip()


def table_to_lines(tab, prev_header=None):
    rows = [[_cell(c) for c in r if _cell(c)] for r in tab.extract()]   # 병합 셀로 생긴 빈 칸 제거
    rows = [r for r in rows if r]
    out, header, last = [], None, None
    # 페이지를 넘어 이어진 표(머리행 없음)는 앞 표의 머리행을 이어 씀
    if rows and prev_header and tab.bbox[1] < 120 and len(rows[0]) == len(prev_header) and rows[0] != prev_header:
        header = prev_header
    for r in rows:
        if header is None:
            if len(r) >= 2:
                header = r                          # 2칸 이상인 첫 행 = 머리행
            else:
                out.append(r[0])                    # 머리행 앞의 제목·인용 문장
            continue
        if len(r) == 1 and last is not None:        # 다음 줄로 넘어간 셀 → 앞 행에 이어 붙이기
            out[last] += " " + r[0]
            continue
        if len(r) == len(header):
            out.append("[표] " + " / ".join(f"{h}: {v}" for h, v in zip(header, r)))
        else:
            out.append("[표] " + " / ".join(r))
        last = len(out) - 1
    if header is not None and last is None:         # 머리행만 있는 1행짜리 표(숫자 카드 등)
        out.append(" / ".join(header))
    return out, header


def load_pdf_with_tables(path, top=50, bottom=50):
    """표지·목차·참고문헌은 빼고, 본문과 표를 읽는 순서대로 합쳐 페이지별 Document로"""
    pdf = pymupdf.open(path)
    texts = [p.get_text() for p in pdf]
    start = next((i for i, t in enumerate(texts) if "Before You Read" in t or "About KT AIVLE School" in t), 0)
    end = next((i for i, t in enumerate(texts) if "R E F E R E N C E S" in t), len(texts))
    docs, prev_header = [], None
    for pno in range(start, end):
        page = pdf[pno]
        H = page.rect.height
        tabs = page.find_tables().tables
        boxes = [pymupdf.Rect(t.bbox) for t in tabs]
        items = []
        for t in tabs:
            lines, prev_header = table_to_lines(t, prev_header)
            if lines:
                items.append((t.bbox[1], "\n".join(lines)))
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
            if y1 < top or y0 > H - bottom:                            # 머리글·바닥글
                continue
            if any(pymupdf.Rect(x0, y0, x1, y1).intersects(b) for b in boxes):
                continue                                                 # 표 안의 글자는 위에서 처리
            t = re.sub(r"[^\S\n]+", " ", text).strip()                       # 탭·특수 공백 정리
            if t:
                items.append((y0, t))
        items.sort(key=lambda x: x[0])
        text = "\n".join(t for _, t in items)
        text = re.sub(r"(\d) (기|주|편|일|회|건|개월|차|명|시간)", r"\1\2", text)   # PDF의 '10 기' → '10기'
        docs.append(Document(page_content=text, metadata={"page": pno}))
    return docs


# (2) 장(01) › 절(1)) › 소제목(■) 경로를 붙여 청크 만들기
H1 = re.compile(r"^\s*0\d\s+\S.*$|^\s*(들어가며|—\s*맺음말|—\s*부록|부록)\b.*$")      # 01 에이블스쿨 소개
H2 = re.compile(r"^\s*\d{1,2}\)\s+\S.*$|^\s*(지원·합격 단계의 선배 이야기|선배 후기에서 발견한|"
                r"최근 선배 후기에서 뽑은|에이블러가 후배에게 전하는|알아둘 변화).*$")     # 5) 매일 기록한 선배
H3 = re.compile(r"^\s*■\s*\S.*$")                                                          # ■ 이렇게 공부했다


def split_by_section(docs, chunk_size=500, chunk_overlap=100):
    sections = []
    h1 = h2 = h3 = ""
    buf, page = [], 0

    def path():
        return " › ".join(x for x in (h1, h2, h3) if x)

    def flush():
        text = "\n".join(buf).strip()
        if text:
            sections.append(Document(page_content=text,
                                     metadata={"section": path() or "AIVLE School 백서",
                                               "parent": " › ".join(x for x in (h1, h2) if x),
                                               "page": page}))

    for d in docs:
        for line in d.page_content.split("\n"):
            s = line.strip()
            if len(s) < 60 and (H1.match(s) or H2.match(s) or H3.match(s)):
                flush()
                buf, page = [], d.metadata["page"]
                if H1.match(s):
                    h1, h2, h3 = s, "", ""
                elif H2.match(s):
                    h2, h3 = s, ""
                else:
                    h3 = s
            elif not buf:
                page = d.metadata["page"]
            buf.append(line)
    flush()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ".", " ", ""])
    chunks = splitter.split_documents(sections)
    for c in chunks:   # 표의 행만 있는 청크도 어떤 주제인지 알 수 있게 경로를 붙임
        c.page_content = f"[{c.metadata['section']}]\n{c.page_content}"
    return chunks, sections


# (3) 질문 키워드 보강: 질문에 쓰는 말 → 백서 소제목에 쓰인 말
QUERY_HINTS = [
    (r"졸업생|선배들", "수료생 선배 후기"),
    (r"공부법|학습법|공부 방법|공부를 어떻게|암기|기억|까먹|잊", "인출 연습 간격 반복 학습법 복습"),
    (r"복습", "인출 연습 간격 반복 복습 간격 경험칙"),
    (r"TIL|til|기록|일지|블로그|회고", "TIL 기록 학습법 매일 기록한 선배 95편 학습 일지"),
    (r"AICE|에이스|자격증", "AICE Associate 합격"),
    (r"합격|붙|자소서|자기소개서|지원서|서류", "합격 후기 자기소개서 인재상 준비법"),
    (r"인적성|역량검사|역량심사|도형|시험", "인적성 준비 도형 시험 환경 합격"),
    (r"10기|처음|초반|첫 ?(달|주)|두 ?달|8주|입교하면|주차", "10기 DX 트랙 교육생의 첫 8주"),
    (r"교육후기|후기|경험|성장|스터디|공모전|수상|비전공|전공", "공식 교육후기 10건 교육생 경험과 성장 전공 여부 프로젝트 스터디 도전 활동"),
    (r"전략|원칙|시기별|단계별|점검|체크", "시기별 전략 학습·프로젝트의 다섯 가지 원칙 단원 마무리 셀프 점검"),
    (r"난이도|어렵|어려움|어려워|힘들|따라가|못 ?따라|벅차", "학습 난이도와 경험 해석 어려움 대응 3주차의 벽"),
]


def expand_query(q):
    extra = [hint for pat, hint in QUERY_HINTS if re.search(pat, q)]
    return q + (" " + " ".join(extra) if extra else "")


def _tokens(s):
    s = re.sub(r"(\d)\s+(기|주|편|일|회|건|개월|차)", r"\1\2", s)     # '10 기' → '10기'
    return [w for w in re.findall(r"[가-힣A-Za-z0-9]+", s) if len(w) >= 2]


class SectionIndex:
    """소제목 전체 내용 보관 + 질문과 소제목 이름을 직접 비교해 해당 절 찾기"""

    def __init__(self, sections):
        self.text, self.parent_text = {}, {}
        for s in sections:
            self.text.setdefault(s.metadata["section"], []).append(s.page_content)
            self.parent_text.setdefault(s.metadata["parent"], []).append(s.page_content)
        self.text = {k: "\n".join(v) for k, v in self.text.items()}
        self.parent_text = {k: "\n".join(v) for k, v in self.parent_text.items()}
        self.first, self.sec_docs, self.parent_docs = {}, {}, {}
        for s in sections:
            self.first.setdefault(s.metadata["section"], s)
            self.sec_docs.setdefault(s.metadata["section"], []).append(s)
            self.parent_docs.setdefault(s.metadata["parent"], []).append(s)
        titles = list(self.text)
        df = {}
        for t in titles:
            for w in set(_tokens(t)):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + len(titles) / n) for w, n in df.items()}

    def route(self, query, top=2):
        q = set(_tokens(query))
        scored = []
        for title in self.text:
            tt = set(_tokens(title.split(" › ", 1)[-1]))        # 장 이름은 빼고 절·소제목으로 비교
            hit = [w for w in tt if any(w in x or x in w for x in q if len(x) >= 2)]
            score = sum(self.idf.get(w, 0) for w in hit)
            if score > 0:
                scored.append((score, title))
        scored.sort(reverse=True)
        best = scored[0][0] if scored else 0
        out, parents = [], set()
        for s, t in scored:                       # 같은 절(1))에서는 하나만 → 서로 다른 절을 고르게
            p = self.first[t].metadata["parent"]
            if s < max(2.0, best * 0.6) or p in parents:
                continue
            parents.add(p)
            out.append(self.first[t])
            if len(out) == top:
                break
        return out

    def expand(self, docs, limit=3500):
        """검색된 조각 → 그 절(1)) 전체가 짧으면 절 전체, 길면 소제목(■) 전체"""
        seen, out = set(), []
        for d in docs:
            parent, sec = d.metadata.get("parent", ""), d.metadata.get("section", "")
            ptxt = self.parent_text.get(parent, "")
            key, body = (parent, ptxt) if parent and len(ptxt) <= limit else (sec, self.text.get(sec, d.page_content))
            if key in seen:
                continue
            seen.add(key)
            out.append(f"[{key}]\n{body[:limit]}")
        return "\n\n".join(clean_text(x) for x in out)

    def evidence(self, docs, limit=3500):
        """답변 근거 후보: expand()와 같은 단위로, 줄마다 (내용, 소제목, 쪽) 보관"""
        seen, out = set(), []
        for d in docs:
            parent, sec = d.metadata.get("parent", ""), d.metadata.get("section", "")
            ptxt = self.parent_text.get(parent, "")
            key, parts = (parent, self.parent_docs.get(parent, [])) if parent and len(ptxt) <= limit \
                else (sec, self.sec_docs.get(sec, [d]))
            if key in seen:
                continue
            seen.add(key)
            lines = [(line, p.metadata.get("section", ""), p.metadata.get("page"))
                     for p in parts for line in p.page_content.split("\n") if line.strip()]
            out.append({"key": key, "lines": lines})
        return out


CONDENSE_PROMPT = """이전 대화를 참고해, 마지막 질문을 앞의 맥락 없이도 이해되는 하나의 검색용 질문으로 바꿔 주세요.
질문만 한 줄로 출력합니다.

[이전 대화]
{history}

[마지막 질문]
{input}"""


def build_chain(retriever, index=None):
    """검색 결과(docs)와 답변(answer)을 함께 돌려주는 RAG 체인"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    llm = ChatOpenAI(model=MODEL, temperature=0, streaming=True)   # 사실 답변은 지어내지 않도록 0
    condense_llm = ChatOpenAI(model=MODEL, temperature=0)

    # "그럼 DX 트랙은요?" 같은 이어지는 질문도 백서에서 잘 찾도록 검색어 재작성 + 키워드 보강
    def search_query(x):
        q = x["input"]
        if x.get("history"):
            hist = "\n".join(
                f"{'사용자' if isinstance(m, HumanMessage) else '도우미'}: {m.content[:300]}"
                for m in x["history"][-4:])
            try:
                q = condense_llm.invoke(CONDENSE_PROMPT.format(history=hist, input=x["input"])).content.strip()
            except Exception:
                pass
        return expand_query(q)

    # 벡터 검색 결과 앞에, 질문과 이름이 맞는 소제목을 먼저 넣기
    def retrieve(q):
        found = retriever.invoke(q)
        routed = index.route(q) if index else []
        return filter_docs(routed + found)

    def make_context(x):
        return index.expand(x["docs"]) if index else format_docs(x["docs"])

    answer_chain = (
        RunnablePassthrough.assign(context=make_context)
        | prompt
        | llm
        | StrOutputParser()
    )
    return (
        RunnablePassthrough.assign(docs=RunnableLambda(search_query) | RunnableLambda(retrieve))
        .assign(evidence=lambda x: index.evidence(x["docs"]) if index else None)
        .assign(answer=answer_chain)
    )


def rag_pipeline(path):
    """01번 노트북의 rag_pipeline(): PDF 경로 → 로드(표 인식) → 소제목 분할 → 임베딩 → 벡터 저장소 → RAG 체인"""
    docs = load_pdf_with_tables(path)
    chunks, sections = split_by_section(docs)
    chunks = [c for c in chunks if not is_reference_chunk(c.page_content)]

    # 업로드 문서는 메모리 저장소에 (이름 충돌 방지용 고유 컬렉션)
    vectorstore = Chroma.from_documents(
        chunks,
        get_embedding(),
        collection_name=f"upload_{uuid.uuid4().hex[:8]}")

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})
    return build_chain(retriever, SectionIndex(sections))


@st.cache_resource(show_spinner="백서를 불러오는 중입니다...")
def get_default_chain():
    """앱 시작 시 기본 백서 체인 (한 번만 생성되어 모든 세션이 공유)
    - 저장소에 백서 PDF가 있으면 고도화 파이프라인으로 생성
    - PDF가 없으면 Colab에서 만든 chroma_db 사용(기본 검색)"""
    pdf = find_file(DEFAULT_PDF)
    if pdf:
        return rag_pipeline(pdf)
    if os.path.isdir(CHROMA_DIR):
        vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=get_embedding(),
            persist_directory=CHROMA_DIR)
        if vectorstore._collection.count() > 0:
            retriever = vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})
            return build_chain(retriever)
    return None


def suggest_followups(question, answer):
    try:
        llm = ChatOpenAI(model=MODEL, temperature=0.7)
        text = llm.invoke(FOLLOWUP_PROMPT.format(q=question, a=answer[:1500])).content
        lines = [re.sub(r"^[\s\-\d\.\)•·*]+", "", l).strip().strip('"') for l in text.splitlines()]
        return [l for l in lines if l][:3]
    except Exception:
        return []


# ─────────────────────────────────────────────
# 3. 세션 상태 (벡터화가 반복되지 않도록 State에 저장)
# ─────────────────────────────────────────────
defaults = {
    "messages": [],          # {"role", "content", "sources", "followups"}
    "chain": None,
    "document_name": None,
    "pending": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.chain is None:
    try:
        st.session_state.chain = get_default_chain()
        if st.session_state.chain is not None:
            st.session_state.document_name = DEFAULT_PDF
    except Exception as e:
        st.error(f"기본 백서를 불러오지 못했습니다: {e}")


def ask(q):
    st.session_state.pending = q


def clear_chat():
    st.session_state.messages = []
    st.session_state.pending = None


# ─────────────────────────────────────────────
# 4. 스타일
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── 다크 테마 팔레트 ── */
:root {
  --bg: #0D171C; --bg2: #111F25; --surface: #16272E; --surface2: #1C323A;
  --line: #27434C; --text: #E4EFEE; --muted: #93ABA8;
  --aivle: #1CC7B6; --aivle-dark: #13A596; --aivle-soft: rgba(28,199,182,.12);
}
.stApp { background: radial-gradient(1200px 600px at 85% -10%, #12333A 0%, var(--bg) 55%) fixed; color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; max-width: 1050px; }
.stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 { color: var(--text); }
.stApp [data-testid="stCaptionContainer"], .stApp small { color: var(--muted) !important; }
.stApp a { color: var(--aivle); }
hr { border-color: var(--line) !important; }

/* 헤더 */
.hero {
  background: linear-gradient(120deg, #0A5E58 0%, #0F8F84 50%, #17B3A4 100%);
  border: 1px solid rgba(28,199,182,.35);
  border-radius: 20px; padding: 26px 30px; margin-bottom: 18px;
  box-shadow: 0 12px 34px rgba(0,0,0,.35); position: relative; overflow: hidden;
}
.hero:after { content: "🚀"; position: absolute; right: 28px; top: 14px; font-size: 64px; opacity: .22; }
.hero h1 { color: #fff !important; font-size: 1.95rem; margin: 0 0 4px 0; padding: 0; }
.hero p { margin: 0; color: #E6FFFB !important; opacity: .92; font-size: 1.02rem; }
.badge { display: inline-block; background: rgba(0,0,0,.22); color: #DFFCF8; border-radius: 999px;
  padding: 3px 12px; font-size: .78rem; margin-bottom: 10px; }

/* 카드 */
.feature, .stat { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; }
.feature { padding: 16px 18px; height: 100%; }
.feature .ico { font-size: 1.6rem; }
.feature b { display: block; margin: 6px 0 2px; color: var(--text); }
.feature span { font-size: .86rem; color: var(--muted) !important; }
.stat { border-left: 5px solid var(--aivle); padding: 12px 14px; }
.stat .num { font-size: 1.45rem; font-weight: 800; color: var(--aivle); }
.stat .lbl { font-size: .8rem; color: var(--muted); }
.section-title { font-weight: 700; font-size: 1.08rem; margin: 22px 0 8px; color: var(--text); }

/* 버튼 */
div.stButton > button {
  border-radius: 12px; border: 1px solid var(--line); background: var(--surface); color: var(--text);
  text-align: left; transition: all .15s ease; width: 100%;
}
div.stButton > button p { color: var(--text); }
div.stButton > button:hover { border-color: var(--aivle); background: var(--surface2); transform: translateY(-1px); }
div.stButton > button:hover p { color: var(--aivle); }
div.stButton > button[kind="primary"] { background: #3A1F24; border-color: #6B3139; }
div.stButton > button[kind="primary"] p { color: #FFB4B4; }
div.stButton > button[kind="primary"]:hover { background: #4A252B; border-color: #FF8A8A; }

/* 사이드바 */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0F2228 0%, #0B1418 100%);
  border-right: 1px solid var(--line); }
.logo-card { background: rgba(255,255,255,.92); border-radius: 14px; padding: 12px 16px; margin-bottom: 8px; }
.logo-card img { width: 100%; display: block; }
.doc-chip { background: var(--aivle-soft); border: 1px solid rgba(28,199,182,.35); border-radius: 12px;
  padding: 8px 12px; font-size: .85rem; color: var(--text); }
.doc-chip b { color: var(--aivle); }

/* 입력·업로드·펼침 */
[data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }
[data-testid="stExpander"] summary:hover p { color: var(--aivle); }
[data-testid="stFileUploaderDropzone"] { background: var(--bg2); border: 1px dashed var(--line); }
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] { background: transparent; }
[data-testid="stChatInput"] { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; }
[data-testid="stChatInput"] textarea { color: var(--text); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted); }

/* 채팅 */
[data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--line);
  border-radius: 16px; padding: 12px 16px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) { background: var(--surface2); }
[data-testid="stChatMessage"] table { width: 100%; border-collapse: collapse; }
[data-testid="stChatMessage"] th { background: rgba(28,199,182,.18); color: var(--text); }
[data-testid="stChatMessage"] th, [data-testid="stChatMessage"] td { border: 1px solid var(--line); padding: 6px 10px; }
[data-testid="stChatMessage"] strong { color: #7FF0E3; }
[data-testid="stChatMessage"] code { background: var(--bg2); color: #9FF5EA; }
.followup-label { font-size: .85rem; color: var(--muted); margin: 6px 0 2px; }

/* 답변 근거 카드 */
.src-card { background: var(--bg2); border: 1px solid var(--line); border-left: 4px solid var(--aivle);
  border-radius: 10px; padding: 10px 14px; margin: 0 0 10px 0; }
.src-head { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.src-no { background: var(--aivle); color: #062B27 !important; font-weight: 800; font-size: .75rem;
  border-radius: 999px; width: 20px; height: 20px; display: inline-flex; align-items: center; justify-content: center; }
.src-page { font-size: .75rem; color: var(--aivle) !important; border: 1px solid rgba(28,199,182,.4);
  border-radius: 999px; padding: 1px 8px; }
.src-title { font-weight: 700; font-size: .9rem; color: var(--text); margin: 2px 0 4px; }
.src-text { font-size: .84rem; line-height: 1.55; color: var(--muted); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 5. 사이드바: 로고 / 관리자(문서 업로드) / 추천 질문 / 대화 삭제
# ─────────────────────────────────────────────
SIDEBAR_QUESTIONS = [
    "🎓 에이블스쿨은 어떤 곳인지 숫자로 한눈에 보여 주세요",
    "📊 AI 개발자 트랙과 DX 컨설턴트 트랙을 표로 비교해 주세요",
    "📝 지원 자격과 선발 절차를 단계별로 알려 주세요",
    "⏰ 하루 일정과 출석 규정은 어떻게 되나요?",
    "💼 수료 후에는 어떤 취업 지원을 받을 수 있나요?",
]

# 02 선배 후기(매일 기록·합격 후기·10기 첫 8주·공식 교육후기) + 03 학습 전략 질문
STORY_QUESTIONS = [
    "✍️ 95편을 매일 기록한 선배는 어떻게 공부했나요?",
    "📝 합격한 선배들은 자기소개서와 인적성을 어떻게 준비했나요?",
    "🗓️ 10기 교육생의 첫 8주는 어땠나요?",
    "🌱 공식 교육후기로 본 교육생들의 성장 사례를 알려 주세요",
    "🧠 인출 연습과 간격 반복 학습법은 어떻게 하나요?",
    "📒 TIL은 어떻게 쓰면 되나요? 템플릿도 보여 주세요",
    "🧭 시기별 학습 전략과 다섯 가지 원칙을 알려 주세요",
]


@st.cache_data
def load_videos():
    """videos.csv(title, url, category, keywords, description) 읽기. 없으면 빈 표"""
    path = find_file(VIDEO_CSV)
    if path is None:
        return pd.DataFrame(columns=["title", "url", "category", "keywords", "description"])
    df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    df = df[df["url"].astype(str).str.startswith("http")]
    df.loc[~df["category"].isin(VIDEO_CATEGORIES), "category"] = "기타"
    return df.reset_index(drop=True)

with st.sidebar:
    st.markdown(f'<div class="logo-card"><img src="{LOGO_URL}" alt="AIVLE"></div>', unsafe_allow_html=True)
    st.caption("에이블스쿨은 AI/DX 인재를 양성하는 KT의 교육 프로그램입니다.")

    st.markdown("#### 🔐 관리자")
    with st.expander("📂 백서 문서 업로드", expanded=st.session_state.chain is None):
        uploaded_file = st.file_uploader("PDF 업로드", type=["pdf"], label_visibility="collapsed")
        if uploaded_file:
            if st.session_state.document_name != uploaded_file.name:
                with st.spinner("문서를 처리하고 있습니다..."):
                    try:
                        # ① 임시 파일 생성 ② 내용 저장 ③ 경로 확보 ④ RAG 파이프라인에 '경로' 전달
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        st.session_state.chain = rag_pipeline(tmp_path)
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.messages = []
                        st.toast("새 문서로 학습 도우미가 준비되었어요! ✨")
                    except Exception as e:
                        st.error(f"문서 처리 중 오류: {e}")

    if st.session_state.document_name:
        st.markdown(f'<div class="doc-chip">📘 사용 중인 문서<br><b>{st.session_state.document_name}</b></div>',
                    unsafe_allow_html=True)
    else:
        st.warning("문서를 업로드해 주세요.")

    st.divider()
    st.markdown("#### 💡 추천 질문")
    for i, q in enumerate(SIDEBAR_QUESTIONS):
        st.button(q, key=f"side_q_{i}", on_click=ask, args=(q.split(" ", 1)[1],))

    with st.expander("📖 선배 후기 · 학습 전략 더 물어보기"):
        for i, q in enumerate(STORY_QUESTIONS):
            st.button(q, key=f"story_q_{i}", on_click=ask, args=(q.split(" ", 1)[1],))

    # 유튜브 재생목록으로 이동 + 카테고리별 영상 둘러보기
    st.divider()
    st.markdown("#### 🎬 에이블스쿨 영상")
    st.link_button("▶ 유튜브 재생목록 전체 보기", PLAYLIST_URL, use_container_width=True)
    videos = load_videos()
    if not videos.empty:
        counts = videos["category"].value_counts()
        options = [c for c in VIDEO_CATEGORIES if c in counts.index]
        category = st.selectbox("카테고리", options, format_func=lambda c: f"{c} ({counts[c]}편)")
        for r in videos[videos["category"] == category].itertuples():
            with st.expander(r.title):
                st.video(r.url)
                if r.description:
                    st.caption(r.description)
                st.link_button("유튜브에서 보기", r.url, use_container_width=True)

    st.divider()
    n_q = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.caption(f"💬 지금까지 나눈 질문: **{n_q}개**")
    st.button("🗑️ 대화 삭제", key="clear", on_click=clear_chat, type="primary")


# ─────────────────────────────────────────────
# 6. 헤더
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">KT AIVLE School · 백서 기반 RAG 학습 도우미</div>
  <h1>AIVLE 메이트</h1>
  <p>먼저 걸어간 선배처럼, 백서에 근거해 정확하게 답하고 여러분의 도전을 진심으로 응원합니다.</p>
</div>
""", unsafe_allow_html=True)

user_input = st.chat_input("에이블스쿨에 대해 무엇이든 물어보세요 ✨")
if st.session_state.pending and not user_input:
    user_input = st.session_state.pending
st.session_state.pending = None


# ─────────────────────────────────────────────
# 7. 첫 화면 (대화가 없을 때)
# ─────────────────────────────────────────────
WELCOME_QUESTIONS = {
    "🤔 지원을 고민하고 있다면": [
        "비전공자인데 6개월 과정을 잘 따라갈 수 있을까요?",
        "저에게 맞는 트랙은 어떻게 고르면 되나요?",
    ],
    "📚 과정이 궁금하다면": [
        "입교하면 첫 8주는 어떻게 흘러가나요?",
        "미니프로젝트와 빅프로젝트는 뭐가 다른가요?",
    ],
    "🚀 성장과 수료 후가 궁금하다면": [
        "선배들은 어떤 공부법으로 성장했나요?",
        "에이블스쿨을 수료하면 무엇을 얻을 수 있나요?",
    ],
}

# 백서 01장 「숫자로 보는 에이블스쿨」
STATS = [("840시간", "이론·실습·프로젝트 (10기)"), ("약 3,500명", "누적 수료생"),
         ("500여 개", "수료생 진출 기업"), ("4회 연속", "KDT 해커톤 대상")]

if not st.session_state.messages and not user_input:
    st.markdown('<div class="section-title">📈 숫자로 보는 에이블스쿨</div>', unsafe_allow_html=True)
    for col, (num, label) in zip(st.columns(4), STATS):
        col.markdown(f'<div class="stat"><div class="num">{num}</div><div class="lbl">{label}</div></div>',
                     unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3)
    for col, (ico, title, desc) in zip(
        (c1, c2, c3),
        [("📘", "백서 기반 정확한 답변", "근거가 된 백서 내용을 함께 보여 드려요"),
         ("📊", "표·단계로 한눈에", "비교와 일정은 표와 흐름도로 정리해요"),
         ("💪", "도전을 응원해요", "내 상황에 맞춘 선배의 한마디까지")]):
        col.markdown(f'<div class="feature"><div class="ico">{ico}</div><b>{title}</b><span>{desc}</span></div>',
                     unsafe_allow_html=True)

    st.markdown('<div class="section-title">이런 것부터 물어보세요 👇</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (cat, qs) in zip(cols, WELCOME_QUESTIONS.items()):
        with col:
            st.markdown(f"**{cat}**")
            for j, q in enumerate(qs):
                st.button(q, key=f"welcome_{cat}_{j}", on_click=ask, args=(q,))


# ─────────────────────────────────────────────
# 8. 대화 출력
# ─────────────────────────────────────────────
AVATAR = {"user": "🧑‍🎓", "assistant": "🤖"}


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 답변 근거 · 백서 발췌 {len(sources)}건"):
        cards = []
        for i, s in enumerate(sources, 1):
            page = f'<span class="src-page">p.{s["page"]}</span>' if s.get("page") else ""
            title = f'<div class="src-title">{html.escape(s["title"])}</div>' if s.get("title") else ""
            cards.append(
                f'<div class="src-card"><div class="src-head"><span class="src-no">{i}</span>{page}</div>'
                f'{title}<div class="src-text">{html.escape(s["text"])}</div></div>')
        st.markdown("".join(cards), unsafe_allow_html=True)


def _bigrams(text):
    t = re.sub(r"[^가-힣A-Za-z0-9]", "", text.lower())
    return {t[i:i + 2] for i in range(len(t) - 1)}


def to_sources(docs, evidence=None, question="", answer="", top=3):
    """답변 근거 카드: 실제 답변과 가장 많이 겹치는 백서 문장을 골라 보여 주기"""
    if not evidence:
        return _to_sources_simple(docs)

    ans, qb = _bigrams(clean_text(answer)), _bigrams(question)
    rows = []                                   # (전체 순번, 근거 묶음 번호, 문장, 소제목, 쪽, bigram)
    for gi, ev in enumerate(evidence):
        for li, (line, sec, page) in enumerate(ev["lines"]):
            t = re.sub(r"^[●•▸\-]\s*", "", clean_text(line).replace("[표] ", ""))
            if len(t) < 10 or re.match(r"^(■|\d{1,2}\)|0\d\s)", t):   # 제목 줄은 발췌에서 제외
                continue
            rows.append((li, gi, t, sec, page, _bigrams(t)))
    if not rows:
        return _to_sources_simple(docs)

    # 흔한 말(합니다, 있습니다 등)은 낮게, 드문 말은 높게
    df = {}
    for r in rows:
        for g in r[5]:
            df[g] = df.get(g, 0) + 1
    idf = {g: math.log(1 + len(rows) / n) for g, n in df.items()}

    def score(b):
        hit = sum(idf[g] for g in b & ans) + 0.3 * sum(idf[g] for g in b & qb)
        return hit / (sum(idf[g] for g in b) ** 0.5 + 1e-9)

    groups = {}
    for li, gi, t, sec, page, b in rows:
        groups.setdefault(gi, []).append((score(b), li, t, sec, page))
    ranked = []
    for gi, items in groups.items():
        best = sorted(items, reverse=True)[:3]
        ranked.append((sum(x[0] for x in best[:2]), best))
    ranked.sort(key=lambda x: -x[0])
    if not ranked or ranked[0][0] <= 0:
        return []

    out = []
    for total, best in ranked:
        if total < ranked[0][0] * 0.45 or len(out) == top:
            break
        lead = best[0]
        picked = sorted([x for x in best if x[0] >= lead[0] * 0.5], key=lambda x: x[1])   # 원래 순서대로
        text = " … ".join(x[2] for x in picked)
        page = lead[4] + 1 if isinstance(lead[4], int) else None
        title = clean_text(lead[3].split(" › ", 1)[-1])[:70]
        out.append({"page": page, "title": title,
                    "text": text[:320] + ("…" if len(text) > 320 else "")})
    return out


def _to_sources_simple(docs):
    """(기본 chroma_db 사용 시) 검색된 조각 앞부분을 그대로 보여 주기"""
    out = []
    for d in docs or []:
        page = d.metadata.get("page")
        if isinstance(page, int):
            page += 1
        raw = d.page_content
        m = re.search(r"^\s*#{1,6}\s*(.+)$", raw, flags=re.M)
        title = clean_text(m.group(1))[:60] if m else ""
        text = clean_text(raw).replace("\n", " ")
        if title and text.startswith(title):
            text = text[len(title):].strip()
        if not text:
            continue
        out.append({"page": page, "title": title,
                    "text": text[:260] + ("…" if len(text) > 260 else "")})
    return out[:4]


last_idx = len(st.session_state.messages) - 1
for idx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"], avatar=AVATAR[m["role"]]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources"))
    # 마지막 답변 아래에 이어서 물어볼 질문
    if idx == last_idx and m["role"] == "assistant" and m.get("followups") and not user_input:
        st.markdown('<div class="followup-label">👉 이어서 물어보기</div>', unsafe_allow_html=True)
        fcols = st.columns(len(m["followups"]))
        for j, (fc, fq) in enumerate(zip(fcols, m["followups"])):
            fc.button(fq, key=f"follow_{idx}_{j}", on_click=ask, args=(fq,))


# ─────────────────────────────────────────────
# 9. 질문 처리
# ─────────────────────────────────────────────
if user_input:
    with st.chat_message("user", avatar=AVATAR["user"]):
        st.markdown(user_input)

    if st.session_state.chain is None:
        with st.chat_message("assistant", avatar=AVATAR["assistant"]):
            st.warning("먼저 왼쪽 '관리자' 메뉴에서 백서 PDF를 업로드해 주세요.")
        st.stop()

    history = []
    for m in st.session_state.messages[-HISTORY_TURNS:]:
        history.append(HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]))

    holder = {}

    def stream_answer():
        for chunk in st.session_state.chain.stream({"input": user_input, "history": history}):
            if "docs" in chunk:
                holder["docs"] = chunk["docs"]
            if "evidence" in chunk:
                holder["evidence"] = chunk["evidence"]
            if "answer" in chunk:
                yield chunk["answer"]

    with st.chat_message("assistant", avatar=AVATAR["assistant"]):
        try:
            with st.spinner("백서에서 관련 내용을 찾고 있어요... 🔎"):
                answer = st.write_stream(stream_answer())
        except Exception as e:
            answer = f"⚠️ 답변 생성 중 문제가 발생했어요: {e}"
            st.error(answer)
        sources = to_sources(holder.get("docs"), holder.get("evidence"),
                             user_input, answer if isinstance(answer, str) else "")
        render_sources(sources)

    with st.spinner("이어서 물어볼 질문을 고르는 중..."):
        followups = suggest_followups(user_input, answer if isinstance(answer, str) else "")

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({
        "role": "assistant", "content": answer if isinstance(answer, str) else str(answer),
        "sources": sources, "followups": followups,
    })
    st.rerun()
