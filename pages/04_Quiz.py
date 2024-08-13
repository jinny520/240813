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

st.title("PDF 기반 QA 및 퀴즈 생성💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "pdf_chain" not in st.session_state:
    st.session_state["pdf_chain"] = None

if "pdf_retriever" not in st.session_state:
    st.session_state["pdf_retriever"] = None

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")

    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )

    selected_prompt = st.selectbox(
        "프롬프트 선택",
        ["prompts/pdf-rag.yaml", "prompts/pdf-quiz.yaml"],
        index=0,
    )

    update_btn = st.button("설정 업데이트")
    
    quiz_btn = st.button("퀴즈 생성")  # 새로운 퀴즈 생성 버튼

# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
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

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key = st.session_state.api_key)

    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    retriever = vectorstore.as_retriever()
    return retriever

# 체인 생성
def create_chain(retriever, prompt_path="prompts/pdf-rag.yaml", model_name="gpt-4o"):
    prompt = load_prompt(prompt_path, encoding="utf-8")
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key = st.session_state.api_key)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# 퀴즈 체인 생성
def create_quiz_chain(retriever, model_name="gpt-4o"):
    quiz_prompt = load_prompt("prompts/pdf-quiz.yaml", encoding="utf-8")
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key = st.session_state.api_key)
    
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
    chain = create_chain(retriever, prompt_path=selected_prompt, model_name=selected_model)
    st.session_state["pdf_retriever"] = retriever
    st.session_state["pdf_chain"] = chain

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

if update_btn:
    if st.session_state["pdf_retriever"] is not None:
        retriever = st.session_state["pdf_retriever"]
        chain = create_chain(retriever, prompt_path=selected_prompt, model_name=selected_model)
        st.session_state["pdf_chain"] = chain

# 퀴즈 생성 버튼이 눌리면...
if quiz_btn and st.session_state.get("pdf_retriever"):
    quiz_chain = create_quiz_chain(st.session_state["pdf_retriever"], model_name=selected_model)
    quiz_question = "Please generate a quiz question based on the PDF content."
    response = quiz_chain(quiz_question)
    st.write("퀴즈 문제:")
    st.write(response)

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    chain = st.session_state["pdf_chain"]

    if chain is not None:
        st.chat_message("user").write(user_input)
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        warning_msg.error("파일을 업로드 해주세요.")
