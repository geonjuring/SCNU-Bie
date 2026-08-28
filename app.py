import os
import time
import streamlit as st
from rag import CurriculumAdvisor, SCHOOL_DEPARTMENT_MAP

# ----------------- 1. 페이지 기본 설정 -----------------
st.set_page_config(
    page_title="스누비(SCNU-bie)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------- 2. 네이비 & 블루 아카데믹 CSS 주입 -----------------
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    html, body, [class*="css"] {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
    }

    section[data-testid="stSidebar"] {
        display: none !important;
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
        transition: all 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        background-color: #F8FAFC !important;
        border-color: #94A3B8 !important;
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
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.03);
    }

    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 8px 18px !important;
        border: none !important;
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #2563EB !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
    }

    .cache-badge {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        border: 1px solid #BFDBFE;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- 3. RAG 엔진 로드 & 메모리 캐싱 -----------------
@st.cache_resource(show_spinner="스누비 학사 AI 엔진을 초기화하는 중입니다...")
def load_advisor():
    return CurriculumAdvisor()

try:
    advisor = load_advisor()
except Exception as e:
    st.error(f"엔진 로드 실패: {e}")
    st.stop()

# 💡 정형 편람 데이터 조회 속도를 위해 데이터 캐싱 추가
@st.cache_data(show_spinner=False)
def get_cached_curriculum_info(school: str, dept: str):
    return advisor.get_curriculum_info(school, dept)

@st.cache_data(show_spinner=False)
def fetch_cached_answer(question: str) -> str:
    return advisor.ask_consultant(question)

def get_answer_with_cache_tracker(question: str):
    start_time = time.time()
    answer = fetch_cached_answer(question)
    elapsed = time.time() - start_time
    is_cached = elapsed < 0.08
    return answer, is_cached

# ----------------- 4. 상단 헤더 배너 -----------------
st.markdown("""
<div class="academic-header">
    <h1>🎓 스누비(SCNU-bie)</h1>
    <p>2026 SCNU 뉴비(Newbie)들을 위한 맞춤형 학사 규정 및 교육과정 네비게이터</p>
</div>
""", unsafe_allow_html=True)

# ----------------- 5. 메인 탭 분리 -----------------
tab1, tab2 = st.tabs(["📚 전공 교육과정 & 졸업 요건", "💬 스누비"])

# ==================== [탭 1] 전공 교육과정 조회 ====================
with tab1:
    st.markdown("### 🏛️ 학과 및 전공 선택")
    
    col_school, col_dept = st.columns(2)
    with col_school:
        selected_school = st.selectbox(
            "1. 소속 스쿨 / 단과대학 선택",
            options=list(SCHOOL_DEPARTMENT_MAP.keys()),
            key="main_selected_school"
        )
    
    with col_dept:
        available_departments = SCHOOL_DEPARTMENT_MAP.get(selected_school, [])
        selected_department = st.selectbox(
            "2. 학과(전공) 선택",
            options=available_departments,
            key="main_selected_dept"
        )

    st.markdown("---")

    # 💡 버튼 클릭 없이도 학과 변경 시 즉시 렌더링되도록 캐시 적용
    if selected_school and selected_department:
        data = get_cached_curriculum_info(selected_school, selected_department)

        st.markdown(f"## 📋 {selected_department} 이수 체계 및 졸업 요건")

        # 1. 졸업 요건 카드 메트릭
        col1, col2, col3 = st.columns(3)
        col1.metric("총 졸업 요구학점", f"{data.get('total_credits', '-')}학점")
        col2.metric("전공 요구학점", f"{data.get('major_total', '-')}학점", f"전필 {data.get('major_req', '-')} / 전선 {data.get('major_elec', '-')}")
        col3.metric("교양 요구학점", f"{data.get('total_ge', '-')}학점", f"기초 {data.get('basic_ge', '-')} / 핵심 {data.get('core_ge', '-')} / 창의 {data.get('creative_ge', '-')}")

        if data.get('foundation_list'):
            st.markdown(f"**스쿨 학문기초 교과목**:\n{data['foundation_list']}")
        st.markdown("---")

        # 2. 전공필수 요약
        st.markdown("### 🔴 전공필수(전필) 핵심 교과목")
        with st.expander("📌 전공필수 전체 목록 보기 (클릭하여 접기/펼치기)", expanded=True):
            if data.get("required_summary"):
                for item in data["required_summary"]:
                    st.markdown(item)
            else:
                st.info("지정된 전공필수 과목이 없습니다.")

        st.markdown("---")

        # 3. 학년·학기별 전공 교과목
        st.markdown("### 📚 권장 학년·학기별 전공 교과목 이수 체계")
        if data.get("grade_data"):
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

        # 4. 별도 지정 교과목
        st.markdown("### 📌 별도 지정 교과목 (교직이수 / 전공인정 타학과 교과목 등)")
        if data.get("separated_data"):
            for cat_title, c_list in data["separated_data"].items():
                with st.expander(f"🔗 {cat_title} (클릭하여 접기/펼치기)", expanded=False):
                    for c in c_list:
                        st.markdown(c)
        else:
            st.info("해당 전공은 별도 지정 과목이 없습니다.")


# ==================== [탭 2] AI 학사 지도 챗봇 ====================
with tab2:
    st.subheader("💬 스누비")
    st.caption("졸업학점, 복수전공 규정 등을 2026 교육과정 기반으로 알려드려요!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container(height=520)

    # 이전 대화 렌더링
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 질문을 입력하시면 2026 편람 규정을 분석해 답변해 드립니다.")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("is_cached"):
                    st.markdown("<span class='cache-badge'>⚡ 캐시 적중 (즉시 반환됨)</span>", unsafe_allow_html=True)

    # 💡 불필요한 rerun을 제거하여 딜레이 최소화
    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt, "is_cached": False})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("2026 학사 규정을 분석하여 답변을 생성 중입니다..."):
                    response, is_cached = get_answer_with_cache_tracker(prompt.strip())
                    st.markdown(response)
                    if is_cached:
                        st.markdown("<span class='cache-badge'>⚡ 캐시 적중 (즉시 반환됨)</span>", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response, "is_cached": is_cached})

    # 하단 대화 삭제 버튼
    if st.session_state.messages:
        if st.button("🗑️ 대화 내용 지우기", use_container_width=False):
            st.session_state.messages = []
            st.rerun()
