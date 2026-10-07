import os
import re
import time
import pymupdf4llm
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# .env 환경변수 로드
load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "gemini-embedding-2"


def get_embedding_model():
    resolved_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=resolved_key,
    )


def ingest_pdfs_to_markdown():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️ '{DATA_DIR}' 폴더에 처리할 PDF 파일이 없습니다.")
        return

    all_chunks = []

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
        ("제", "Article_Header"),
        ("■", "Section_Header"),
    ]
    md_header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    # 💡 429 TPM 제한을 피하기 위해 청크 사이즈를 600으로 축소
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n\n", "\n\n", "\n제", "\n|", "\n", " "],
    )

    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_DIR, pdf_file)
        print(f"📄 PDF ➡️ 마크다운 변환 중: {pdf_file}")
        
        md_text = pymupdf4llm.to_markdown(file_path)
        md_docs = md_header_splitter.split_text(md_text)

        for doc in md_docs:
            header_context = " > ".join([str(v) for v in doc.metadata.values()])
            
            dept_tag = "공통"
            combined_header_str = " ".join([str(v) for v in doc.metadata.values()])
            dept_match = re.search(r'([가-힣a-zA-Z·]+(?:전공|학과|학부|트랙))', combined_header_str)
            if dept_match:
                dept_tag = dept_match.group(1).strip()

            sub_chunks = text_splitter.split_text(doc.page_content)

            for txt in sub_chunks:
                final_content = f"[{header_context}]\n{txt}" if header_context else txt

                all_chunks.append(
                    Document(
                        page_content=final_content,
                        metadata={
                            "source": pdf_file,
                            "context": header_context,
                            "target_department": dept_tag,
                        },
                    )
                )

    if not all_chunks:
        return

    embeddings = get_embedding_model()
    
    # 빈 벡터스토어 생성
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    # 💡 무료 플랜 TPM(30K) 초과 방지: 3개씩 전송하고 5초간 대기
    batch_size = 3
    total_chunks = len(all_chunks)
    print(f"🔄 총 {total_chunks}개 청크 임베딩 전송 시작 (무료 플랜 제한 회피를 위해 안전하게 전송합니다)...")

    for i in range(0, total_chunks, batch_size):
        batch = all_chunks[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"🔄 진행 상황: {min(i + batch_size, total_chunks)} / {total_chunks} 완료")
        
        if i + batch_size < total_chunks:
            time.sleep(5.0)

    print(f"✅ 편람 규정 및 잔여 표 인덱싱 완료: 총 {total_chunks}개 청크")


if __name__ == "__main__":
    ingest_pdfs_to_markdown()
