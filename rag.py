import os
import json
import re
import time
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# .env 환경변수 로드
load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "gemini-embedding-2"
RULES_PATH = os.path.join(DATA_DIR, "rules.json")
CURRICULUM_PATH = os.path.join(DATA_DIR, "curriculum.json")

SCHOOL_DEPARTMENT_MAP = {
    "우주항공·첨단소재스쿨": [
        "인공지능공학전공", "컴퓨터공학전공", "기계우주항공공학전공", "첨단신소재공학전공",
        "화학공학전공", "전기공학전공", "전자공학전공", "토목공학전공", "환경공학전공",
        "화학전공", "에너지응용공학전공", "글로벌에너지응용공학과"
    ],
    "그린스마트팜스쿨": [
        "농생명과학전공", "산림자원학전공", "조경학전공", "동물자원과학전공",
        "원예학전공", "식품공학전공", "농업경제학전공", "의생명과학전공",
        "조리과학전공", "바이오한약자원학전공", "국제농축산학과(농업·원예전공트랙)",
        "국제농축산학과(동물자원전공트랙)"
    ],
    "애니메이션·문화콘텐츠스쿨": [
        "경제학전공", "무역학전공", "경영학전공", "법학전공", "행정학전공",
        "회계학전공", "물류학전공", "사회복지학전공", "글로벌중국학전공",
        "일본어일본문화학전공", "사학전공", "철학전공", "문예창작학전공",
        "사회체육학전공", "음악예술융합학전공", "사진미디어학전공",
        "영상디자인학전공", "만화애니메이션학전공", "패션디자인학전공"
    ],
    "본부직속": [
        "자유전공학부(인문사회·자연)", "식품영양학과", "융합바이오시스템기계공학과", "간호학과",
        "국제한국어교육학과", "건축학부", "글로벌인재학부(글로벌매니지먼트전공)",
        "글로벌인재학부(글로벌ICT문화예술콘텐츠전공)", "스마트안전관리학과(계약학과)"
    ],
    "사범대학": [
        "국어교육과", "영어교육과", "사회교육과", "농업교육과(식물자원·조경 전공)",
        "농업교육과(동물자원 전공)", "수학교육과", "컴퓨터교육과", "환경교육과",
        "물리교육과", "화학교육과"
    ],
    "약학대학": ["약학과"],
    "평생교육스쿨": [
        "미래융합학부(물류비즈니스전공트랙)", "미래융합학부 (융합산업전공트랙)",
        "미래융합학부 (동물생명산업전공트랙)", "미래융합학부(정원문화산업전공트랙)",
        "미래융합학부(스포츠레저전공트랙)"
    ]
}

@st.cache_resource
def load_json_file(file_path: str):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif os.path.exists(os.path.basename(file_path)):
        with open(os.path.basename(file_path), "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def sort_grade_key(key: str) -> int:
    digits = re.findall(r'\d+', str(key))
    return int(digits[0]) if digits else 99

def format_course_detail(course: dict, is_required: bool) -> str:
    badge = "🔴" if is_required else "🔵"
    code = course.get("code", "")
    name = course.get("name", "")
    credit = course.get("credit", 3)
    remark = course.get("remark", "")
    target_note = course.get("target_grade_note", "")

    extra_tags = []
    if remark:
        extra_tags.append(remark)
    if target_note:
        extra_tags.append(target_note)
    extra_info = f" ({', '.join(extra_tags)})" if extra_tags else ""

    return f"- {badge} **{code}** {name} ({credit}학점){extra_info}"

def format_separated_course_item(course: dict) -> str:
    code = course.get("code", "")
    name = course.get("name", "")
    credit = course.get("credit", 3)
    remark = course.get("remark", "")
    target_note = course.get("target_grade_note", "")

    extra_tags = []
    if remark:
        extra_tags.append(remark)
    if target_note:
        extra_tags.append(target_note)
    extra_info = f" ({', '.join(extra_tags)})" if extra_tags else ""

    return f"* 🟢 **{code}** {name} ({credit}학점){extra_info}"

def normalize_dept_name(name: str) -> str:
    return re.sub(r'\(.*?\)', '', name).strip().replace(" ", "")

def detect_department_from_query(query: str) -> str:
    """질문 내용에서 등록된 학과 이름이 포함되어 있는지 감지합니다.
    반환값: 학과명 또는 None. 필터링에 사용하려면 normalize_dept_name() 적용 필요."""
    for school, depts in SCHOOL_DEPARTMENT_MAP.items():
        for dept in depts:
            clean_dept = normalize_dept_name(dept)
            if clean_dept in query.replace(" ", "") or dept in query:
                return dept
    return None

def get_vectorstore_base():
    """캐시 없이 순수 벡터스토어 객체를 반환합니다."""
    resolved_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if os.path.exists(CHROMA_DIR) and resolved_key:
        try:
            from langchain_community.vectorstores import Chroma
            embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL_NAME,
                google_api_key=resolved_key,
            )
            return Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
            )
        except Exception as e:
            print(f"Vectorstore load error: {e}")
    return None

def get_department_curriculum(department: str):
    curriculum_list = load_json_file(CURRICULUM_PATH)
    dept_obj = {}
    target_clean = department.strip().replace(" ", "")
    target_norm = normalize_dept_name(department)

    for item in curriculum_list:
        d_name = item.get("department", "").strip()
        if d_name == department or normalize_dept_name(d_name) == target_norm or d_name.replace(" ", "") == target_clean:
            dept_obj = item
            break

    curriculum = dept_obj.get("curriculum", {})
    separated = dept_obj.get("separated_courses", {})
    
    required_summary = []
    grade_data_dict = {}

    sorted_grades = sorted(curriculum.keys(), key=sort_grade_key)

    for grade_key in sorted_grades:
        sems = curriculum[grade_key]
        grade_str = str(grade_key).replace(",", "·")
        grade_label = f"{grade_str}학년" if not grade_str.endswith("학년") else grade_str
        
        grade_info = {"note": "", "semesters": {}}
        has_content = False

        if isinstance(sems, dict):
            if "note" in sems and sems["note"]:
                grade_info["note"] = sems["note"]
                has_content = True

            for sem_name, types in sems.items():
                if sem_name == "note" or not isinstance(types, dict):
                    continue

                req_courses = types.get("필수", [])
                elec_courses = types.get("선택", [])

                if not req_courses and not elec_courses:
                    continue

                has_content = True
                grade_info["semesters"][sem_name] = {
                    "필수": [format_course_detail(c, is_required=True) for c in req_courses],
                    "선택": [format_course_detail(c, is_required=False) for c in elec_courses]
                }

                for c in req_courses:
                    code = c.get("code", "")
                    name = c.get("name", "")
                    credit = c.get("credit", 3)
                    required_summary.append(f"* 🔴 **{code}** {name} ({credit}학점) - **{grade_label} {sem_name}**")

        if has_content:
            grade_data_dict[grade_label] = grade_info

    separated_results = {}
    teaching = separated.get("teaching_courses", [])
    if teaching:
        separated_results["교직이수(교과교육영역) 교과목"] = [format_separated_course_item(c) for c in teaching]

    cross = separated.get("cross_recognized_courses", [])
    if cross:
        separated_results["전공 인정 타학과 교과목"] = [format_separated_course_item(c) for c in cross]

    jnu_shared = separated.get("jnu_shared_courses", [])
    if jnu_shared:
        separated_results["대학간 공동교육과정 개설 교과목"] = [format_separated_course_item(c) for c in jnu_shared]

    return required_summary, grade_data_dict, separated_results

class CurriculumAdvisor:
    def __init__(self, api_key: str = None, model_name: str = "gemini-3.5-flash-lite"):
        self.rules = load_json_file(RULES_PATH)
        resolved_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=resolved_key,
            google_api_key=resolved_key,
            temperature=0.1,
            max_output_tokens=4096,
        ) if resolved_key else None

    def get_curriculum_info(self, school: str, department: str) -> dict:
        school_info = self.rules.get(school, {}) if isinstance(self.rules, dict) else {}
        dept_info = school_info.get(department, {})
        if not dept_info:
            target_norm = normalize_dept_name(department)
            for k, v in school_info.items():
                if normalize_dept_name(k) == target_norm:
                    dept_info = v
                    break

        dept_code = dept_info.get("code", "")
        total_credits = dept_info.get("total_credits", 140)

        general_req = dept_info.get("general", {})
        basic_ge = general_req.get("basic", 10)
        core_ge = general_req.get("core", 6)
        creative_ge = general_req.get("creative", 14)
        total_ge = general_req.get("total", "30~46")

        major_req_info = dept_info.get("major", {})
        major_req = major_req_info.get("required", 12)
        major_elec = major_req_info.get("elective", 60)
        major_total = major_req_info.get("total", 72)

        foundation_courses = dept_info.get("school_foundation", [])
        foundation_list = "\n".join([f"  - {c}" for c in foundation_courses]) if foundation_courses else "  - 해당 없음"

        req_summary, grade_data_dict, separated_data = get_department_curriculum(department)

        return {
            "department": department,
            "school": school,
            "dept_code": dept_code,
            "total_credits": total_credits,
            "major_total": major_total,
            "major_req": major_req,
            "major_elec": major_elec,
            "total_ge": total_ge,
            "basic_ge": basic_ge,
            "core_ge": core_ge,
            "creative_ge": creative_ge,
            "foundation_list": foundation_list,
            "required_summary": req_summary,
            "grade_data": grade_data_dict,
            "separated_data": separated_data
        }

    def ask_consultant(self, user_question: str) -> str:
        if not self.llm:
            return "⚠️ Gemini API Key가 설정되지 않았습니다."

        t0 = time.perf_counter()
        vectorstore = get_vectorstore_base()
        t1 = time.perf_counter()
        t2 = t1
        n_docs = 0
        rules_context = ""
        
        if vectorstore:
            try:
                # 💡 질문에서 학과명 감지 후 메타데이터 $or 필터 동적 적용
                detected_dept = detect_department_from_query(user_question)
                search_kwargs = {"k": 2}
                
                if detected_dept:
                    # 필터에 사용할 학과명을 정규화 (괄호 제거)
                    normalized_dept = normalize_dept_name(detected_dept)
                    search_kwargs["filter"] = {
                        "$or": [
                            {"target_department": normalized_dept},
                            {"target_department": "공통"}
                        ]
                    }
                    print(f"🔍 [메타데이터 필터 적용] 대상 학과: {detected_dept} -> {normalized_dept}")
                else:
                    print("🔍 [전체/공통 검색] 특정 학과가 감지되지 않아 공통 규정 위주로 검색합니다.")

                retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
                retrieved_docs = retriever.invoke(user_question)
                t2 = time.perf_counter()
                n_docs = len(retrieved_docs)
                rules_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            except Exception as e:
                rules_context = f"(규정 검색 중 오류: {e})"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 국립순천대학교 교육과정 학사 지도 전문 컨설턴트입니다.
2026학년도 교육과정 편람 지침(교양/전공 이수원칙, 자유전공학부 2학년 진입규정, 다전공/부전공/융합전공 요건, 졸업 기준 등)을 바탕으로 학생의 질문에 명확하고 친절하게 답변하세요."""),
            ("user", f"""[질문 내용]:
{user_question}

[편람 규정 검색 결과]:
{rules_context if rules_context else "검색된 관련 규정이 없습니다."}

위 검색된 편람 규정을 바탕으로 학생의 질문에 대해서만 구체적이고 정확하게 답변해 주세요.""")
        ])
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({})
        t3 = time.perf_counter()
        # 진단용: 어느 단계가 느린지 로그로 확인 (Streamlit Cloud > Manage app > Logs)
        # flush=True: 로그에 바로 나타나도록 (없으면 출력이 늦게 보일 수 있음)
        print(
            f"[타이밍] 질문='{user_question[:20]}' | 벡터스토어 준비 {t1-t0:.2f}s | "
            f"검색(질문 임베딩 포함) {t2-t1:.2f}s (문서 {n_docs}개, 근거 {len(rules_context)}자) | "
            f"LLM 답변 {t3-t2:.2f}s (답변 {len(answer)}자) | 전체 {t3-t0:.2f}s",
            flush=True,
        )
        return answer
