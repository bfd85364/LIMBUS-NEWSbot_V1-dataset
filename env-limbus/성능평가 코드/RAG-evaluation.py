# -*- coding: utf-8 -*-

#11월 1일 
#출처: https://wikidocs.net/259208 위키독스 - LLM-AS-Judge , https://docs.smith.langchain.com/evaluation/faq/evaluator-implementations -랜체인 공식 문서 evaluation
# PDFRAG 관련: https://wikidocs.net/265516
# 임베딩 기반 평가방식 - https://wikidocs.net/259210

#현재 RAG 모듈은 각각 LIMBUS_NEWSbot_v1.py (봇구동 파일), pdf_embedding.py(PDF문서 임베딩), querying_utf8.py(사용자 입력 처리)
#세 가지의 파일로 모듈화 시킨 관계로 RAG 모듈 객체를 가져오는 것은 어려움 
#그러므로, 모듈들의 객체중 검색기의 기능을 지닌 QA_chin의 객체 및 관련 코드들을 가져와서 사용함
#따라서, LLM-AS-Judge 기법 및  임베딩 기반  평가방식과  유사한 방식의 평가를 진행 
#이전에 langsmith으로 생성한 합성 데이터를 가져와서 사용함 

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


from langchain_teddynote import logging

logging.langsmith("Limbus-Evaluation")

from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langsmith.evaluation import evaluate, LangChainStringEvaluator

pdf_filepath = 'LIMBUS_INFO2.pdf'
loader = PyPDFLoader(pdf_filepath)
docs= loader.load()

embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
index = FAISS.from_documents(docs, embeddings)
retriever = index.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY, verbose=True)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

def ask_question(inputs: dict):
    # qa_chain 호출 → dict 반환
    output = qa_chain(inputs["question"])  # 또는 qa_chain.invoke(inputs["question"])
    # result만 반환
    return {"answer": output['result']}

dataset_name = "LIMBUS_RAG_QA"

qa_evaluator = LangChainStringEvaluator("qa")

def print_evaluator_prompt(evaluator):
    print(evaluator.evaluator.prompt.pretty_print())

print_evaluator_prompt(qa_evaluator)

experiment_results = evaluate(
    ask_question,
    data=dataset_name,
    evaluators=[qa_evaluator],
    experiment_prefix="RAG_EVAL",
    metadata={
        "variant": "QA Evaluator + PDF RAG",
    },
)

# 정확: 6개 

# 부정확: 8개 

# 응답 거부: 1개 

# 따라서 사용자 만족도의 정확도보다 정확도가 더 낮게  측정됨 

# QA 템플릿 내용 수정 필요 