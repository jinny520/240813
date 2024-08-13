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

    # 모델 선택 메
