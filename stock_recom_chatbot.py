import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import random
import urllib.parse
import re
from datetime import datetime, timedelta
from transformers import pipeline

# Hugging Face 요약 모델 로드
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# 유사 단어 필터링을 위한 정규화 함수
def normalize_word(word):
    return re.sub(r'[^a-zA-Z가-힣]', '', word).lower()

# 뉴스 크롤링 함수
def crawl_naver_news(company_name, start_date, end_date):
    try:
        encoded_query = urllib.parse.quote(company_name)
        date_filter = f"nso=so:r,p:from{start_date}to{end_date}"
        base_url = f"https://search.naver.com/search.naver?where=news&sm=tab_jum&query={encoded_query}&{date_filter}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        titles, channels, links, contents = [], [], [], []
        news_items = soup.select("ul.list_news > li")

        for item in news_items:
            title_elem = item.select_one("a.news_tit")
            title = title_elem.text.strip() if title_elem else "제목 없음"
            channel_elem = item.select_one("a.info")
            channel = channel_elem.text.strip() if channel_elem else "언론사 정보 없음"
            link = title_elem['href'] if title_elem else "링크 없음"
            content_elem = item.select_one("div.news_dsc")
            content = content_elem.text.strip() if content_elem else ""

            # 필터링 조건: 기업명 포함
            if company_name in title or company_name in content:
                titles.append(title)
                channels.append(channel)
                links.append(link)
                contents.append(content)

        return titles, channels, links, contents
    except Exception as e:
        st.error(f"크롤링 오류 발생: {e}")
        return None

# 뉴스 요약 함수
def summarize_news(content):
    try:
        summary = summarizer(content, max_length=50, min_length=20, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.warning(f"요약 실패: {e}")
        return content[:100] + "..."  # 요약 실패 시 원본 일부 반환

# Streamlit 앱 구성
st.title("📈 뉴스 기반 주식 정보 요약")
st.subheader("기업명을 입력하여 관련 뉴스의 요약 정보를 확인하세요.")

# 사용자 입력
search_query = st.text_input("검색할 기업명을 입력하세요:", value="삼성전자")
search_days = st.slider("검색 기간 (일 단위)", 1, 30, 7)

# 검색 버튼
if st.button("요약된 뉴스 검색"):
    with st.spinner("뉴스를 수집하고 요약 중입니다..."):
        try:
            # 날짜 계산
            today = datetime.today()
            start_date = today - timedelta(days=search_days)
            start_date_str = start_date.strftime('%Y%m%d')
            end_date_str = today.strftime('%Y%m%d')

            # 뉴스 크롤링 실행
            result = crawl_naver_news(search_query, start_date_str, end_date_str)

            if result:
                titles, channels, links, contents = result
                st.success(f"{len(titles)}개의 뉴스를 가져왔습니다.")

                # 요약 결과 출력
                for title, channel, link, content in zip(titles, channels, links, contents):
                    summary = summarize_news(content)
                    st.markdown(f"### [{title}]({link})")
                    st.write(f"**언론사**: {channel}")
                    st.write(f"**요약**: {summary}")
                    st.markdown("---")
            else:
                st.warning("관련 뉴스를 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")



