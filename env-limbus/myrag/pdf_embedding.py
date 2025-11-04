import os
import sys
import time
import random
import logging
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 

load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

pdf_filepath = 'LIMBUS_INFO2.pdf'
loader = PyPDFLoader(pdf_filepath)
pages = loader.load()
print(f"총 페이지 수: {len(pages)}") 

text_splitter = CharacterTextSplitter(
    separator = '{}',
    chunk_size = 300,
    chunk_overlap  = 144,
    length_function = len,
)
texts = text_splitter.split_documents(pages)
print(f"생성된 청크 수: {len(texts)}")

embedding = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask") #사용자에게 한국어로 잡변을 제공해야하므로, 또한 사용하는 문서는 영문명과 한글명이 혼재되어있음 -> 이중에서 한국어만 추출되도 상관x 
vectorstore = FAISS.from_documents(texts, embedding)
vectorstore.save_local("index_storage") #index_storage 폴더에 저장  (로컬 저장)  

logging.info("모든 문서가 임베딩 및 저장되었습니다.")