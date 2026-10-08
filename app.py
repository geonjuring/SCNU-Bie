import os
import json
import time
import uuid
import streamlit as st
from rag import CurriculumAdvisor, SCHOOL_DEPARTMENT_MAP

# ----------------- 1. 페이지 기본 설정 -----------------
st.set_page_config(
    page_title="스누비(SCNU-bie)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded", # 대화 저장/불러오기 기능을 위해 사이드바 기본 활성화
)

# ----------------- 2. 네이비 & 블루 아카데믹 CSS 주입 -----------------
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    html, body, [class*="css"] {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #F1F5F9 0%, #E2E8F0 100%) !important;
    }

    .academic-header {
        background: linear-gradient(135deg, #0F2942 0%, #1E3A8A 50%, #2563EB 100%);
        padding: 26px 32px;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 6px 20px rgba(15, 41, 66, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .academic-header h1 {
        color: #FFFFFF !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .academic-header p {
        color: #E2E8F0 !important;
        font-size: 0.98rem !important;
        margin: 0;
        opacity: 0.95;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        padding: 18px 22px !important;
        border-radius: 12px !important;
        border: 1px solid #CBD5E1 !important;
        border-top: 4px solid #1E3A8A !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04) !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.92rem !important;
        color: #475569 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        color: #0F172A !important;
        font-weight: 800 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E2E8F0 !important;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #CBD5E1;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        padding: 10px 22px !important;
        border-radius: 8px !important;
        border: none !important;
        color: #475569 !important;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(30, 58, 138, 0.25);
    }

    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-left: 5px solid #2563EB !important;
        border-radius: 10px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        padding: 14px 20px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02) !important;
    }
    div[data-testid="stExpanderDetails"] {
        border-left: 1px solid #CBD5E1;
        border-right: 1px solid #CBD5E1;
        border-bottom: 1px solid #CBD5E1;
        border-bottom-left-radius: 10px;
        border-bottom-right-radius: 10px;
        padding: 20px !important;
        background-color: #FFFFFF !important;
        margin-top: -8px;
        margin-bottom: 14px;
    }

    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
        border: none !important;
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
    }
    div.stButton > button:hover {
        background-color: #2563EB !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
    }

    /* 사이드바 대화 목록 버튼: 흰 배경의 가벼운 스타일 */
    section[data-testid="stSidebar"] div.stButton > button {
        background-color: #FFFFFF !important;
        color: #1E3A8A !important;
        border: 1px solid #CBD5E1 !important;
        font-weight: 600 !important;
        padding: 8px 12px !important;
        justify-content: flex-start !important;
        text-align: left !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #EFF6FF !important;
        border-color: #2563EB !important;
        box-shadow: none;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- 3. RAG 엔진 및 캐시 최적화 -----------------
@st.cache_resource(show_spinner="학사 엔진을 초기화하는 중입니다...")
def load_advisor():
    return CurriculumAdvisor()

try:
    advisor = load_advisor()
except Exception as e:
    st.error(f"엔진 로드 실패: {e}")
    st.stop()

@st.cache_data(show_spinner=False)
def fetch_cached_curriculum(school: str, dept: str):
    return advisor.get_curriculum_info(school, dept)

@st.cache_data(show_spinner=False)
def fetch_cached_answer(question: str) -> str:
    return advisor.ask_consultant(question)

# ----------------- 4. 사이드바: 대화 목록 -----------------
# --- chat helpers (start) ---
TITLE_MAX = 28


def create_chat():
    """빈 대화를 하나 만들고 현재 대화로 지정합니다."""
    cid = uuid.uuid4().hex[:8]
    st.session_state.chats[cid] = {"title": "", "messages": []}
    st.session_state.current_chat = cid


def prune_empty(keep_id):
    """질문이 없는 빈 대화는 keep_id만 남기고 정리합니다."""
    for cid in list(st.session_state.chats):
        if cid != keep_id and not st.session_state.chats[cid]["messages"]:
            del st.session_state.chats[cid]


def init_chat_state():
    if "chats" not in st.session_state:
        st.session_state.chats = {}
        st.session_state.current_chat = None
    if st.session_state.current_chat not in st.session_state.chats:
        create_chat()
    # 기존 코드가 쓰는 st.session_state.messages는 현재 대화의 메시지 목록을 가리킨다
    st.session_state.messages = st.session_state.chats[st.session_state.current_chat]["messages"]


def on_new_chat():
    if st.session_state.chats[st.session_state.current_chat]["messages"]:
        create_chat()
        prune_empty(st.session_state.current_chat)


def on_select_chat(cid):
    if cid in st.session_state.chats:
        st.session_state.current_chat = cid
        prune_empty(cid)


def on_delete_chat():
    st.session_state.chats.pop(st.session_state.current_chat, None)
    if st.session_state.chats:
        st.session_state.current_chat = list(st.session_state.chats)[-1]
    else:
        create_chat()


def export_chats_json():
    data = [
        {"title": c["title"], "messages": c["messages"]}
        for c in st.session_state.chats.values()
        if c["messages"]
    ]
    return json.dumps(data, ensure_ascii=False, indent=2)


def _clean_messages(msgs):
    return [
        {"role": m["role"], "content": m["content"]}
        for m in msgs
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]


def import_chats(raw):
    """JSON을 읽어 대화 목록에 추가하고, 추가된 대화 수를 반환합니다."""
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("지원하지 않는 파일 형식입니다.")
    # 예전 형식(메시지 목록 하나)도 하나의 대화로 읽는다
    if data and all(isinstance(m, dict) and "role" in m for m in data):
        data = [{"title": "", "messages": data}]
    added = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        msgs = item.get("messages")
        if not isinstance(msgs, list):
            continue
        clean = _clean_messages(msgs)
        if not clean:
            continue
        first_user = next((m["content"] for m in clean if m["role"] == "user"), "새 대화")
        title = str(item.get("title") or first_user).replace("\n", " ")[:TITLE_MAX]
        st.session_state.chats[uuid.uuid4().hex[:8]] = {"title": title, "messages": clean}
        added += 1
    return added
# --- chat helpers (end) ---

with st.sidebar:
    init_chat_state()
    st.markdown("### 💬 대화 목록")
    st.button("➕ 새 대화", key="new_chat_btn", on_click=on_new_chat, use_container_width=True)

    saved_chats = [(cid, c) for cid, c in reversed(list(st.session_state.chats.items())) if c["messages"]]
    if not saved_chats:
        st.caption("아직 대화가 없습니다. 질문하면 여기에 쌓입니다.")
    for cid, c in saved_chats:
        marker = "▶ " if cid == st.session_state.current_chat else ""
        n_q = sum(1 for m in c["messages"] if m["role"] == "user")
        st.button(
            f"{marker}{c['title']}",
            key=f"chat_btn_{cid}",
            on_click=on_select_chat,
            args=(cid,),
            use_container_width=True,
            help=f"질문 {n_q}개",
        )

    st.markdown("---")
    st.button("🗑️ 현재 대화 삭제", key="delete_chat_btn", on_click=on_delete_chat, use_container_width=True)

    with st.expander("💾 대화 저장 / 불러오기"):
        st.download_button(
            label="📥 대화 전체 저장 (JSON)",
            data=export_chats_json(),
            file_name="scnu_bie_chats.json",
            mime="application/json",
            use_container_width=True,
            disabled=not saved_chats,
        )
        uploaded_file = st.file_uploader("📂 저장한 대화 불러오기", type=["json"], key="chat_uploader")
        if uploaded_file is None:
            st.session_state["_loaded_sig"] = None
        else:
            sig = (uploaded_file.name, uploaded_file.size)
            if st.session_state.get("_loaded_sig") != sig:
                load_ok = False
                try:
                    n_added = import_chats(uploaded_file.getvalue().decode("utf-8"))
                    st.session_state["_load_msg"] = f"{n_added}개의 대화를 불러왔습니다."
                    load_ok = True
                except Exception as e:
                    st.session_state["_load_msg"] = f"파일을 읽는 중 오류가 발생했습니다: {e}"
                st.session_state["_loaded_sig"] = sig
                if load_ok:
                    st.rerun()
            if st.session_state.get("_load_msg"):
                st.caption(st.session_state["_load_msg"])

    st.caption("💡 저장하지 않으면 새로고침 시 대화가 사라집니다.")

# ----------------- 5. 상단 헤더 배너 -----------------
st.markdown("""
<div class="academic-header">
    <h1>🎓 스누비(SCNU-bie)</h1>
    <p>2026 SCNU 뉴비(Newbie)들을 위한 학사 네비게이터</p>
</div>
""", unsafe_allow_html=True)

# ----------------- 6. 메인 탭 분리 -----------------
tab1, tab2 = st.tabs(["📚 전공 교육과정 & 졸업 요건", "💬 스누비"])

# ==================== [탭 1] 전공 교육과정 조회 ====================
with tab1:
    st.markdown("### 🏛️ 학과 및 전공 선택")
    
    col_school, col_dept = st.columns(2)
    
    school_options = ["단과대를 선택해 주세요."] + list(SCHOOL_DEPARTMENT_MAP.keys())
    with col_school:
        selected_school = st.selectbox(
            "1. 소속 스쿨 / 단과대학 선택",
            options=school_options,
            key="main_selected_school"
        )
    
    with col_dept:
        if selected_school == "단과대를 선택해 주세요.":
            selected_department = st.selectbox(
                "2. 학과(전공) 선택",
                options=["먼저 단과대를 선택해 주세요."],
                key="main_selected_dept_disabled",
                disabled=True
            )
            selected_department = None
        else:
            available_departments = SCHOOL_DEPARTMENT_MAP[selected_school]
            dept_options = ["학과를 선택해 주세요."] + available_departments
            selected_department = st.selectbox(
                "2. 학과(전공) 선택",
                options=dept_options,
                key="main_selected_dept"
            )
            if selected_department == "학과를 선택해 주세요.":
                selected_department = None

    st.caption("학과를 선택하면 졸업요건과 학년·학기별 교육과정이 바로 아래에 표시됩니다.")
    st.markdown("---")

    if selected_department:
        data = fetch_cached_curriculum(selected_school, selected_department)

        st.markdown(f"## 📋 {selected_department} 이수 체계 및 졸업 요건")

        col1, col2, col3 = st.columns(3)
        col1.metric("총 졸업 요구학점", f"{data['total_credits']}학점")
        col2.metric("전공 요구학점", f"{data['major_total']}학점", f"전필 {data['major_req']} / 전선 {data['major_elec']}")
        col3.metric("교양 요구학점", f"{data['total_ge']}학점", f"기초 {data['basic_ge']} / 핵심 {data['core_ge']} / 창의 {data['creative_ge']}")

        st.markdown(f"**스쿨 학문기초 교과목**:\n{data['foundation_list']}")
        st.markdown("---")

        st.markdown("### 🔴 전공필수(전필) 핵심 교과목")
        with st.expander("📌 전공필수 전체 목록 보기 (클릭하여 접기/펼치기)", expanded=True):
            if data["required_summary"]:
                for item in data["required_summary"]:
                    st.markdown(item)
            else:
                st.info("지정된 전공필수 과목이 없습니다.")

        st.markdown("---")

        st.markdown("### 📚 권장 학년·학기별 전공 교과목 이수 체계")
        if data["grade_data"]:
            for grade_label, g_info in data["grade_data"].items():
                with st.expander(f"📌 {grade_label} 교과목 목록 (클릭하여 접기/펼치기)", expanded=True):
                    if g_info.get("note"):
                        st.info(f"💡 {g_info['note']}")
                    
                    for sem_name, sem_courses in g_info.get("semesters", {}).items():
                        st.markdown(f"##### 🔹 {sem_name}")
                        if sem_courses.get("필수"):
                            st.markdown("* **[전공필수]**")
                            for c in sem_courses["필수"]:
                                st.markdown(f"  {c}")
                        if sem_courses.get("선택"):
                            st.markdown("* **[전공선택]**")
                            for c in sem_courses["선택"]:
                                st.markdown(f"  {c}")
                        st.markdown("")
        else:
            st.warning("개설된 전공 교과목 정보가 없습니다.")

        st.markdown("---")

        st.markdown("### 📌 별도 지정 교과목 (교직이수 / 전공인정 타학과 교과목 등)")
        if data["separated_data"]:
            for cat_title, c_list in data["separated_data"].items():
                with st.expander(f"🔗 {cat_title} (클릭하여 접기/펼치기)", expanded=False):
                    for c in c_list:
                        st.markdown(c)
        else:
            st.info("해당 전공은 별도 지정 과목이 없습니다.")
    else:
        st.info("👆 위에서 소속 스쿨과 학과(전공)를 선택해 주세요.")


# ==================== [탭 2] AI 학사 지도 챗봇 ====================
with tab2:
    st.subheader("💬 스누비")
    st.caption("졸업학점, 복수전공 규정 등을 2026 교육과정 기반으로 알려드려요!")

    EXAMPLE_QUESTIONS = [
        "졸업하려면 학점이 총 몇 학점 필요해?",
        "복수전공은 어떻게 신청해?",
        "자유전공학부는 언제 전공을 정해?",
    ]

    def set_pending_question(q):
        st.session_state.pending_question = q

    # 대화 출력 컨테이너
    chat_container = st.container(height=450)

    with chat_container:
        if not st.session_state.messages:
            st.info("👋 질문을 입력하시거나 아래 추천 질문을 눌러 학사 규정을 확인해보세요.")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 💡 추천 질문 버튼들을 입력창 바로 위에 상시 고정 유지
    st.markdown("📌 **추천 질문 빠른 선택**")
    ex_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        ex_cols[i].button(q, key=f"example_q_{i}", on_click=set_pending_question, args=(q,), use_container_width=True)

    typed_prompt = st.chat_input("질문을 입력하세요")
    prompt = st.session_state.pop("pending_question", None) or typed_prompt

    if prompt:
        current_chat = st.session_state.chats[st.session_state.current_chat]
        if not current_chat["title"]:
            current_chat["title"] = prompt.strip().replace("\n", " ")[:TITLE_MAX]
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("2026 학사 규정을 분석하여 답변을 생성 중입니다..."):
                    try:
                        response = fetch_cached_answer(prompt.strip())
                    except Exception as e:
                        response = "⚠️ 답변을 가져오지 못했습니다. 잠시 후 같은 질문을 다시 시도해 주세요."
                        print(f"[답변 생성 오류] {e}")
                    st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
