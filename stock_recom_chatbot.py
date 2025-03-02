import streamlit as st
import requests
import urllib.parse
import FinanceDataReader as fdr
import tiktoken
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import os
import random
from difflib import SequenceMatcher
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt

def main():
    st.set_page_config(page_title="Stock Analysis Chatbot", page_icon=":chart_with_upwards_trend:")
    st.title("기업 정보 분석 QA Chat")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False
    if "news_data" not in st.session_state:
        st.session_state.news_data = None
    if "company_name" not in st.session_state:
        st.session_state.company_name = None
    if "selected_period" not in st.session_state:
        st.session_state.selected_period = "1day"


    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        company_name = st.text_input("분석할 기업명 (코스피 상장)")
        days = st.number_input("최근 며칠 동안의 기사를 검색할까요?", min_value=1, max_value=30, value=7)  # 기간을 사용자 입력받기
        process = st.button("분석 시작")

    if process:
        if not openai_api_key or not company_name:
            st.info("OpenAI API 키와 기업명을 입력해주세요.")
            st.stop()

        news_data = crawl_news(company_name, days)
        if not news_data:
            st.warning("해당 기업의 최근 뉴스를 찾을 수 없습니다.")
            st.stop()
            

        # 분석 결과를 session_state에 저장
        st.session_state.news_data = news_data
        st.session_state.company_name = company_name

        text_chunks = get_text_chunks(news_data)
        vectorstore = get_vectorstore(text_chunks)
        
        st.session_state.conversation = create_chat_chain(vectorstore, openai_api_key)
        st.session_state.processComplete = True

    # 분석 결과가 있으면 항상 상단에 출력
    if st.session_state.processComplete and st.session_state.company_name:
        st.subheader(f"📈 {st.session_state.company_name} 최근 주가 추이")
        selected_period = st.radio(
            "기간 선택",
            options=["1day", "week", "1month", "1year"],
            horizontal=True,
            index=["1day", "week", "1month", "1year"].index(st.session_state.selected_period)
        )
      
        if selected_period != st.session_state.selected_period:
            st.session_state.selected_period = selected_period

        st.write(f"🔍 선택된 기간: {st.session_state.selected_period}")

        with st.spinner(f"📊 {st.session_state.company_name} ({st.session_state.selected_period}) 데이터 불러오는 중..."):
            if selected_period in ["1day", "week"]:
                ticker = get_ticker(st.session_state.company_name, source="yahoo")  # ✅ 야후 파이낸스용 티커
                if not ticker:
                    st.error("해당 기업의 야후 파이낸스 티커 코드를 찾을 수 없습니다.")
                    return

                interval = "1m" if selected_period == "1day" else "5m"
                df = get_intraday_data_yahoo(ticker, period="5d" if selected_period == "week" else "1d",
                                             interval=interval)

            else:
                ticker = get_ticker(st.session_state.company_name, source="fdr")  # ✅ FinanceDataReader용 티커
                if not ticker:
                    st.error("해당 기업의 FinanceDataReader 티커 코드를 찾을 수 없습니다.")
                    return

                df = get_daily_stock_data_fdr(ticker, selected_period)

            if df.empty:
                st.warning(
                    f"📉 {st.session_state.company_name} - 해당 기간({st.session_state.selected_period})의 거래 데이터가 없습니다.")
            else:
                plot_stock_plotly(df, st.session_state.company_name, st.session_state.selected_period)

        st.markdown("최근 기업 뉴스 목록을 보려면 누르시오")

    if st.session_state.processComplete:
        with st.expander("뉴스 보기"):
            news_data = st.session_state.news_data

            # 처음 10개 뉴스만 표시
            for i, news in enumerate(news_data[:10]):
                st.markdown(f"- **{news['title']}** ([링크]({news['link']}))")

            # '더 많은 뉴스보기' 버튼 상태 확인 (기본값 False)
            if "show_more_clicked" not in st.session_state:
                st.session_state.show_more_clicked = False

            # 버튼 클릭 시 상태 변경
            if not st.session_state.show_more_clicked:
                if st.button("더 많은 뉴스보기"):
                    st.session_state.show_more_clicked = True
                    st.rerun()  # 버튼 클릭 시 페이지 새로고침하여 버튼 제거

            # 버튼이 눌리면 추가 뉴스 표시 (rerun 이후 실행됨)
            if st.session_state.get("show_more_clicked", False):  # ✅ 세션 상태 확인 방식 개선
                for news in news_data[10:]:
                    st.markdown(f"- **{news['title']}** ([링크]({news['link']}))")

    # 채팅 부분: 사용자가 질문을 입력하면 대화가 이어짐
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

def crawl_news(company, days):
    today = datetime.today()
    start_date = (today - timedelta(days=days)).strftime('%Y%m%d')
    end_date = today.strftime('%Y%m%d')
    encoded_query = urllib.parse.quote(company)

    url_template = f"https://search.naver.com/search.naver?where=news&query={encoded_query}&nso=so:r,p:from{start_date}to{end_date}&start={{}}"

    headers = {
        "User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36"
        ])
    }

    data = []
    seen_urls = set()
    seen_titles = []
    seen_contents = []

    for page in range(1, 6):  # 1~5 페이지 크롤링
        url = url_template.format((page - 1) * 10 + 1)
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.select("ul.list_news > li")

        for article in articles:
            title = article.select_one("a.news_tit").text.strip()
            link = article.select_one("a.news_tit")['href']
            content = article.select_one("div.news_dsc").text.strip() if article.select_one("div.news_dsc") else ""

            # ✅ 1. URL 중복 검사
            if link in seen_urls:
                continue
            seen_urls.add(link)

            # ✅ 2. 제목 중복 검사 (TF-IDF 기반 유사도 체크)
            is_duplicate_title = False
            for existing_title in seen_titles:
                vectorizer = TfidfVectorizer().fit_transform([title, existing_title])
                similarity = cosine_similarity(vectorizer)[0, 1]
                if similarity > 0.6:  # 60% 이상 유사하면 중복으로 판단
                    is_duplicate_title = True
                    break
            if is_duplicate_title:
                continue
            seen_titles.append(title)

            # ✅ 3. 본문 유사도 검사 (Jaccard Similarity)
            def jaccard_similarity(str1, str2):
                set1, set2 = set(str1.split()), set(str2.split())
                return len(set1 & set2) / len(set1 | set2)

            is_duplicate_content = False
            for existing_content in seen_contents:
                if jaccard_similarity(content, existing_content) > 0.5:  # 50% 이상 유사하면 중복 처리
                    is_duplicate_content = True
                    break
            if is_duplicate_content:
                continue
            seen_contents.append(content)

            # ✅ 4. 본문이 너무 짧거나 없는 경우 제외
            if len(content) < 20:  # 20자 이하는 광고성, 불완전 기사일 가능성 높음
                continue

            data.append({"title": title, "link": link, "content": content})

    return data



def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text_chunks(news_data):
    texts = [f"{item['title']}\n{item['content']}" for item in news_data]
    metadatas = [{"source": item["link"]} for item in news_data]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    return text_splitter.create_documents(texts, metadatas=metadatas)

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

# ✅ 1. 최근 거래일 찾기 함수
def get_recent_trading_day():
    today = datetime.now()
    if today.hour < 9:  # 9시 이전이면 전날을 기준으로
        today -= timedelta(days=1)

    while today.weekday() in [5, 6]:  # 토요일(5), 일요일(6)이면 하루씩 감소
        today -= timedelta(days=1)

    return today.strftime('%Y-%m-%d')

# ✅ 2. 티커 조회 함수 (야후 & FinanceDataReader)
def get_ticker(company, source="yahoo"):
    try:
        listing = fdr.StockListing('KRX')
        ticker_row = listing[listing["Name"].str.strip() == company.strip()]
        if not ticker_row.empty:
            krx_ticker = str(ticker_row.iloc[0]["Code"]).zfill(6)
            if source == "yahoo":
                return krx_ticker + ".KS"  # ✅ 야후 파이낸스용 티커 변환
            return krx_ticker  # ✅ FinanceDataReader용 티커
        return None

    except Exception as e:
        st.error(f"티커 조회 중 오류 발생: {e}")
        return None

# ✅ 3. 야후 파이낸스에서 분봉 데이터 가져오기 (1day, week)
def get_intraday_data_yahoo(ticker, period="1d", interval="1m"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df = df.rename(columns={"Datetime": "Date", "Close": "Close"})

        # ✅ 주말 데이터 제거 (혹시 남아있는 경우 대비)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Date"].dt.weekday < 5].reset_index(drop=True)

        return df
    except Exception as e:
        st.error(f"야후 파이낸스 데이터 불러오기 오류: {e}")
        return pd.DataFrame()

# ✅ 4. FinanceDataReader를 통한 일별 시세 (1month, 1year)
def get_daily_stock_data_fdr(ticker, period):
    try:
        end_date = get_recent_trading_day()
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=30 if period == "1month" else 365)).strftime('%Y-%m-%d')
        df = fdr.DataReader(ticker, start_date, end_date)

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df = df.rename(columns={"Date": "Date", "Close": "Close"})

        # ✅ 주말 데이터 완전 제거
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Date"].dt.weekday < 5].reset_index(drop=True)

        return df
    except Exception as e:
        st.error(f"FinanceDataReader 데이터 불러오기 오류: {e}")
        return pd.DataFrame()

# ✅ 5. Plotly를 이용한 주가 시각화 함수 (x축 포맷 최적화)
def plot_stock_plotly(df, company, period):
    if df is None or df.empty:
        st.warning(f"📉 {company} - 해당 기간({period})의 거래 데이터가 없습니다.")
        return

    fig = go.Figure()

    # ✅ x축 날짜 형식 설정
    if period == "1day":
        df["FormattedDate"] = df["Date"].dt.strftime("%H:%M")  # ✅ 1day → HH:MM 형식
    elif period == "week":
        df["FormattedDate"] = df["Date"].dt.strftime("%m-%d %H:%M")  # ✅ week → MM-DD HH:MM 형식
    else:
        df["FormattedDate"] = df["Date"].dt.strftime("%m-%d")  # ✅ 1month, 1year → MM-DD 형식

    if period in ["1day", "week"]:
        fig.add_trace(go.Scatter(
            x=df["FormattedDate"],
            y=df["Close"],
            mode="lines+markers",
            line=dict(color="royalblue", width=2),
            marker=dict(size=5),
            name="체결가"
        ))
    else:
        fig.add_trace(go.Candlestick(
            x=df["FormattedDate"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="캔들 차트"
        ))

    fig.update_layout(
        title=f"{company} 주가 ({period})",
        xaxis_title="시간" if period == "1day" else "날짜",
        yaxis_title="주가 (KRW)",
        template="plotly_white",
        xaxis=dict(showgrid=True, type="category", tickangle=-45),
        hovermode="x unified"
    )

    st.plotly_chart(fig)


    

if __name__ == '__main__':
    main()
