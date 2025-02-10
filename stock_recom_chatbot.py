import streamlit as st
import torch
import mplfinance as mpf
import FinanceDataReader as fdr
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def process_news_data(news_items):
    """RecursiveCharacterTextSplitter를 활용하여 뉴스 조각내기"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = [news['title'] + " " + news['description'] for news in news_items]
    split_texts = text_splitter.split_text(" ".join(texts))
    return split_texts


def create_embeddings(texts):
    """KoBERT를 활용하여 뉴스 임베딩 생성 및 저장"""
    embeddings = HuggingFaceEmbeddings(model_name="skt/kobert-base-v1")
    vector_store = Chroma.from_texts(texts, embeddings)
    return vector_store


def retrieve_relevant_sentences(query, vector_store):
    """LangChain Retriever를 사용하여 관련 문장 검색"""
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)
    return " ".join([doc.page_content for doc in relevant_docs])


def generate_summary(query, retriever, openai_key):
    """GPT-4를 활용하여 요약 생성"""
    llm = ChatOpenAI(openai_api_key=openai_key, model_name='gpt-4', temperature=0)
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    return qa_chain.run(query)


def analyze_sentiment(news_text):
    """KoBERT 모델을 사용하여 뉴스 감성 분석"""
    tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
    model = AutoModelForSequenceClassification.from_pretrained("skt/kobert-base-v1")

    inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = "긍정" if probs[0][1] > probs[0][0] else "부정"
    return sentiment


def visualize_stock(symbol, period):
    """MPLFinance를 활용한 주가 시각화 (일/주/월/년)"""
    df = fdr.DataReader(symbol, '2024-01-01')
    if period == "일":
        df = df.tail(30)
    elif period == "주":
        df = df.resample('W').last()
    elif period == "월":
        df = df.resample('M').last()
    elif period == "년":
        df = df.resample('Y').last()
    mpf.plot(df, type='candle', style='charles', title=f"{symbol} 주가 ({period})", volume=True)


st.title("국내 주식 뉴스 기반 추천 QA 챗봇")
company_name = st.text_input("기업명을 입력하세요:")
openai_key = st.text_input("OpenAI API Key", type="password")
period = st.selectbox("조회 기간", ["일", "주", "월", "년"])

if st.button("분석 실행"):
    # 뉴스 데이터 수집 (예제 데이터 사용)
    news_items = [
        {"title": "기업 A의 주가 상승", "description": "기업 A의 주가가 급등했습니다."},
        {"title": "기업 B의 실적 발표", "description": "기업 B가 좋은 실적을 발표했습니다."}
    ]
    texts = process_news_data(news_items)
    vector_store = create_embeddings(texts)
    retriever = vector_store.as_retriever()
    context = retrieve_relevant_sentences(company_name, vector_store)
    summary = generate_summary(company_name, retriever, openai_key)
    sentiment = analyze_sentiment(context)
    
    st.subheader("뉴스 요약")
    st.write(summary)
    st.subheader("감성 분석 결과")
    st.write(f"{company_name} 관련 뉴스 감성: {sentiment}")
    
    visualize_stock(company_name, period)
















