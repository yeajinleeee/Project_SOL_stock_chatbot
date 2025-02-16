import requests
from bs4 import BeautifulSoup
import time
import random
import urllib.parse
import re
from datetime import datetime, timedelta
import streamlit as st
import tiktoken
import mplfinance as mpf
import FinanceDataReader as fdr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 유사 단어 필터링을 위한 정규화 함수
def 정규화_단어(단어):
    단어 = re.sub(r'[^a-zA-Z가-힣]', '', 단어).lower()
    return 단어

# 뉴스 크롤링 함수 (여러 페이지 처리)
def 네이버_뉴스_크롤링(기업명, 시작날짜, 종료날짜, 페이지_수=5):
    try:
        encoded_query = urllib.parse.quote(기업명)
        date_filter = f"nso=so:r,p:from{시작날짜}to{종료날짜}"

        titles, channels, links, contents = [], [], [], []

        for page in range(1, 페이지_수 + 1):
            url = f"https://search.naver.com/search.naver?where=news&sm=tab_jum&query={encoded_query}&{date_filter}&start={((page-1)*10) + 1}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }

            time.sleep(random.uniform(1, 3))
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            news_items = soup.select("ul.list_news > li")

            for item in news_items:
                title_element = item.select_one("a.news_tit")
                title = title_element.text.strip() if title_element else "제목 없음"
                channel_element = item.select_one("a.info")
                channel = channel_element.text.strip() if channel_element else "언론사 정보 없음"
                link = title_element['href'] if title_element else "링크 없음"

                content_element = item.select_one("div.news_dsc")
                content = content_element.text.strip() if content_element else ""

                if link and not link.startswith("http"):
                    link = "https://search.naver.com" + link

                if 기업명 in title or 기업명 in content:
                    titles.append(title)
                    channels.append(channel)
                    links.append(link)
                    contents.append(content)

        filtered_titles = []
        filtered_channels = []
        filtered_links = []
        filtered_contents = []
        seen_titles = set()
        seen_words = set()

        for title, channel, link, content in zip(titles, channels, links, contents):
            title_without_company = title.replace(기업명, '').strip()
            words = set(정규화_단어(word) for word in title_without_company.split())

            if title not in seen_titles and not seen_words & words:
                filtered_titles.append(title)
                filtered_channels.append(channel)
                filtered_links.append(link)
                filtered_contents.append(content)
                seen_titles.add(title)
                seen_words.update(words)

        if filtered_titles:
            return filtered_titles, filtered_channels, filtered_links, filtered_contents
        else:
            print(f"🔍 '{기업명}' 관련 뉴스가 {시작날짜}부터 {종료날짜}까지 없습니다.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 에러 발생: {e}")
        return None
    except Exception as e:
        print(f"❌ 크롤링 중 에러 발생: {e}")
        return None

# 주식 추천 및 챗봇 부분
def main():
    st.set_page_config(page_title="Stock Analysis Chatbot", page_icon=":chart_with_upwards_trend:")
    st.title("_기업 정보 분석 주식 추천 :red[QA Chat]_ :chart_with_upwards_trend:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        company_name = st.text_input("분석할 기업명 (코스피 상장)")
        process = st.button("분석 시작")

    if process:
        if not openai_api_key or not company_name:
            st.info("OpenAI API 키와 기업명을 입력해주세요.")
            st.stop()

        # 뉴스 크롤링
        news_data = 네이버_뉴스_크롤링(company_name, "20230101", "20230201")
        if not news_data:
            st.warning("해당 기업의 최근 뉴스를 찾을 수 없습니다.")
            st.stop()

        titles, channels, links, contents = news_data

        # 챗봇 시스템 설정
        text_chunks = get_text_chunks(news_data)
        vectorstore = get_vectorstore(text_chunks)

        st.session_state.conversation = create_chat_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True

        # 기업 주식 분석
        st.subheader(f"📈 {company_name} 최근 주가 추이")
        visualize_stock(company_name, "일")

        with st.chat_message("assistant"):
            st.markdown("📢 최근 기업 뉴스 목록:")
            for title, link in zip(titles, links):
                st.markdown(f"**{title}** ([링크]({link}))")

    if query := st.chat_input("질문을 입력해주세요."):
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("분석 중..."):
                result = st.session_state.conversation({"question": query})
                response = result['answer']

                st.markdown(response)
                with st.expander("참고 뉴스 확인"):
                    for doc in result['source_documents']:
                        st.markdown(f"- [{doc.metadata['source']}]({doc.metadata['source']})")

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text_chunks(news_data):
    texts = [f"{item['title']}\n{item['content']}" for item in news_data]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    return text_splitter.create_documents(texts)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(text_chunks, embeddings)

def create_chat_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4', temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h, return_source_documents=True)

def get_ticker(company):
    try:
        listing = fdr.StockListing('KRX')
        if listing.empty:
            listing = fdr.StockListing('KOSPI')
        if listing.empty:
            st.error("KRX 혹은 KOSPI 상장 기업 정보를 불러올 수 없습니다.")
            return None

        if "Code" in listing.columns and "Name" in listing.columns:
            name_col = "Name"
            ticker_col = "Code"
        elif "Symbol" in listing.columns and "Name" in listing.columns:
            name_col = "Name"
            ticker_col = "Symbol"
        elif "종목코드" in listing.columns and "기업명" in listing.columns:
            name_col = "기업명"
            ticker_col = "종목코드"
        else:
            st.error("상장 기업 정보의 컬럼명이 예상과 다릅니다: " + ", ".join(listing.columns))
            return None

        ticker_row = listing[listing[name_col].str.strip() == company.strip()]
        if ticker_row.empty:
            st.error(f"입력한 기업명 '{company}'에 해당하는 정보가 없습니다.")
            return None
        else:
            ticker = ticker_row.iloc[0][ticker_col]
            return str(ticker).zfill(6)
    except Exception as e:
        st.error(f"티커 변환 중 오류 발생: {e}")
        return None


def visualize_stock(company, period):
    ticker = get_ticker(company)
    if not ticker:
        st.error("해당 기업의 티커 코드를 찾을 수 없습니다.")
        return

    try:
        df = fdr.DataReader(ticker, '2024-01-01')
    except Exception as e:
        st.error(f"주가 데이터를 불러오는 중 오류 발생: {e}")
        return

    if df.empty:
        st.warning("주가 데이터가 없습니다.")
        return

    df.index = pd.to_datetime(df.index)
    mpf.plot(df, type='candle', style='charles', title=company, ylabel='Price')
