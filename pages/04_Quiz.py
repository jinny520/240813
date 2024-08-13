import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# API KEY 정보로드
load_dotenv()

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("PDF 기반 퀴즈 생성 및 답안 판별기📚❓")

# 처음 1번만 실행하기 위한 코드
if "pdf_retriever" not in st.session_state:
    st.session_state["pdf_retriever"] = None

if "quiz_question" not in st.session_state:
    st.session_state["quiz_question"] = None

if "correct_answer" not in st.session_state:
    st.session_state["correct_answer"] = None

# 사이드바 생성
with st.sidebar:
    # 파일 업로드
    uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])

    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4-turbo", "gpt-4o"], index=0
    )

    generate_quiz_btn = st.button("퀴즈 생성")

# 파일을 캐시 저장 및 임베딩 생성 (시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))

    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    retriever = vectorstore.as_retriever()
    return retriever

# 퀴즈 생성 체인
def create_quiz_chain(retriever, model_name="gpt-4-turbo"):
    quiz_prompt = "Generate a quiz question and provide the correct answer based on the following content:"
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | quiz_prompt
        | llm
        | StrOutputParser()
    )
    return chain

# 파일이 업로드 되었을 때
if uploaded_file:
    retriever = embed_file(uploaded_file)
    st.session_state["pdf_retriever"] = retriever
    st.success("PDF 파일이 성공적으로 업로드 및 처리되었습니다.")

# 퀴즈 생성 버튼이 눌리면...
if generate_quiz_btn and st.session_state.get("pdf_retriever"):
    quiz_chain = create_quiz_chain(st.session_state["pdf_retriever"], model_name=selected_model)
    quiz_question = "Please generate a quiz question and provide the correct answer based on the PDF content."
    response = quiz_chain(quiz_question)
    
    # 결과를 파싱하여 질문과 정답을 분리
    if "?" in response:
        question, answer = response.split("?", 1)
        question += "?"
        st.session_state["quiz_question"] = question.strip()
        st.session_state["correct_answer"] = answer.strip()
    
    st.write("생성된 퀴즈 문제:")
    st.write(st.session_state["quiz_question"])
elif generate_quiz_btn:
    st.warning("먼저 PDF 파일을 업로드하세요.")

# 사용자가 정답 입력 후 제출하는 인터페이스
if st.session_state.get("quiz_question"):
    user_answer = st.text_input("퀴즈에 대한 답을 입력하세요:")
    submit_btn = st.button("제출")

    if submit_btn and user_answer:
        correct_answer = st.session_state["correct_answer"]
        if user_answer.strip().lower() == correct_answer.lower():
            st.success("정답입니다!")
        else:
            st.error(f"오답입니다. 정답은: {correct_answer}")

