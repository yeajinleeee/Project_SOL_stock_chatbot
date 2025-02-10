import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import random
import urllib.parse
import re
from datetime import datetime, timedelta
from transformers import pipeline

# 유사 단어 필터링을 위한 정규화 함수
def 정규화_단어(단어):
    단어 = re.sub(r'[^a-zA-Z가-힣]', '', 단어).lower()
    return 단어

# 뉴스 크롤링 함수 (여러 페이지 처리)
def 네이버_뉴스_크롤링(기업명, 시작날짜, 종료날짜, 페이지_수=5):
    try:
        # 기업명 URL 인코딩
        encoded_query = urllib.parse.quote(기업명)
        date_filter = f"nso=so:r,p:from{시작날짜}to{종료날짜}"

        titles, channels, links, contents = [], [], [], []
        
        for page in range(1, 페이지_수 + 1):
            url = f"https://search.naver.com/search.naver?where=news&sm=tab_jum&query={encoded_query}&{date_filter}&start={((page-1)*10) + 1}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }

            # 요청을 보내고 응답 받기
            time.sleep(random.uniform(1, 3))  # 서버 과부하 방지를 위한 지연
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            news_items = soup.select("ul.list_news > li")  # 뉴스 항목 선택

            for item in news_items:
                title_element = item.select_one("a.news_tit")
                title = title_element.text.strip() if title_element else "제목 없음"
                channel_element = item.select_one("a.info")
                channel = channel_element.text.strip() if channel_element else "언론사 정보 없음"
                link = title_element['href'] if title_element else "링크 없음"

                content_element = item.select_one("div.news_dsc")
                content = content_element.text.strip() if content_element else ""

                # 링크가 상대 경로일 경우 절대 경로로 변환
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
        seen_titles = set()  # 이미 등장한 제목을 추적
        seen_words = set()   # 이미 등장한 단어를 추적

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
            return None
    except requests.exceptions.RequestException as e:
        return None
    except Exception as e:
        return None

# 기업명과 날짜 범위 입력 받기
st.title("기업 뉴스 요약")

search_query = st.text_input("검색할 기업명을 입력하세요:")
search_days = st.number_input("며칠 동안의 기사를 검색할까요? (예: 3 입력 시 3일 전부터 오늘까지):", min_value=1, value=3)

# 날짜 계산
today = datetime.today()
start_date = today - timedelta(days=search_days)
start_date_str = start_date.strftime('%Y%m%d')
end_date_str = today.strftime('%Y%m%d')

if search_query:
    result = 네이버_뉴스_크롤링(search_query, start_date_str, end_date_str)

    if result:
        titles, channels, links, contents = result
        st.write(f"**{search_query}** 관련 뉴스 ({start_date_str}부터 {end_date_str}까지)")

        # 뉴스 기사 요약 및 링크만 출력
        summarizer = pipeline("summarization")

        for title, channel, link, content in zip(titles, channels, links, contents):
            # 기사 내용 요약
            summary = summarizer(content[:1024], max_length=150, min_length=50, do_sample=False)[0]['summary_text']

            st.subheader(f"제목: {title}")
            st.write(f"**언론사**: {channel}")
            st.write(f"**링크**: [링크로 이동]({link})")
            st.write(f"**요약**: {summary}")
            st.write("---")
    else:
        st.write("🔍 해당 기업에 관련된 뉴스가 없습니다.")














