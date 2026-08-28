import os
import re
import pymupdf4llm
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

DATA_DIR = "data"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
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

    # 1. 마크다운 대제목 / 중제목 / 조항 분할 기준
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

    # 💡 규정 텍스트와 남은 표가 깨지지 않도록 최적화된 청크 사이즈 (1000~1200자)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n\n", "\n\n", "\n제", "\n|", "\n", " "],
    )

    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_DIR, pdf_file)
        print(f"📄 PDF ➡️ 마크다운 변환 및 색인 중: {pdf_file}")
        
        # pymupdf4llm으로 표 구조를 마크다운 테이블(|---|---|)로 깔끔하게 보존 변환
        md_text = pymupdf4llm.to_markdown(file_path)

        md_docs = md_header_splitter.split_text(md_text)

        for doc in md_docs:
            header_context = " > ".join([str(v) for v in doc.metadata.values()])
            
            # 학과 태그 추출
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

    # 기존 DB 캐시 충돌 방지 및 인덱싱
    embeddings = get_embedding_model()
    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"✅ 편람 규정 및 잔여 표 인덱싱 완료: 총 {len(all_chunks)}개 청크")


if __name__ == "__main__":
    ingest_pdfs_to_markdown()