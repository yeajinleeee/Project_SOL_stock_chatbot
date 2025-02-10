import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import random
import urllib.parse
import re
from datetime import datetime, timedelta


# 유사 단어 필터링을 위한 정규화 함수
def 정규화_단어(단어):
    단어 = re.sub(r'[^a-zA-Z가-힣]', '', 단어).lower()
    return 단어


# 뉴스 크롤링 함수
def 네이버_뉴스_크롤링(기업명, 시작날짜, 종료날짜):
    try:
        encoded_query = urllib.parse.quote(기업명)
        date_filter = f"nso=so:r,p:from{시작날짜}to{종료날짜}"

        titles, channels, links, contents = [], [], [], []

        url = f"https://search.naver.com/search.naver?where=news&sm=tab_jum&query={encoded_query}&{date_filter}"
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

            if 기업명 in title or 기업명 in content:
                titles.append(title)
                channels.append(channel)
                links.append(link)
                contents.append(content)

        return titles, channels, links, contents
    except Exception as e:
        st.error(f"크롤링 중 오류가 발생했습니다: {e}")
        return None


# Streamlit 앱 구성
st.title("\ud83d\udcc8 뉴스 기반 주식 정보")
st.subheader("기업명을 입력하여 관련 뉴스를 확인하세요.")

search_query = st.text_input("검색할 기업명을 입력하세요:")
search_days = st.slider("검색 기간(일)", 1, 30, 3)

if st.button("검색 시작"):
    with st.spinner("뉴스를 가져오는 중입니다..."):
        today = datetime.today()
        start_date = today - timedelta(days=search_days)
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = today.strftime('%Y%m%d')

        result = 네이버_뉴스_크롤링(search_query, start_date_str, end_date_str)

    if result:
        titles, channels, links, contents = result
        st.success(f"{len(titles)}개의 뉴스를 가져왔습니다.")

        # 뉴스 카드 생성
        for title, channel, link, content in zip(titles, channels, links, contents):
            with st.container():
                st.markdown(f"### [{title}]({link})")
                st.write(f"\ud83d\udd16 언론사: {channel}")
                st.write(f"\ud83d\udcc4 요약: {content[:100]}...")
                st.markdown("---")
    else:
        st.error("관련 뉴스를 찾을 수 없습니다.")

