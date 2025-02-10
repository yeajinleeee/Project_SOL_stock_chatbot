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

                # 본문 내용 추출 (기업명이 본문에 포함되어 있는지 확인)
                content_element = item.select_one("div.news_dsc")
                content = content_element.text.strip() if content_element else ""

                # 링크가 상대 경로일 경우 절대 경로로 변환
                if link and not link.startswith("http"):
                    link = "https://search.naver.com" + link  # 네이버 뉴스 링크 앞에 base URL 붙이기

                # 기업명이 제목에 포함된 뉴스만 필터링하거나 본문에 포함된 뉴스도 포함
                if 기업명 in title or 기업명 in content:
                    titles.append(title)
                    channels.append(channel)
                    links.append(link)
                    contents.append(content)

        # 중복된 제목 필터링 (제목 기준)
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
            st.warning(f"🔍 '{기업명}' 관련 뉴스가 {시작날짜}부터 {종료날짜}까지 없습니다.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ 요청 에러 발생: {e}")
        return None
    except Exception as e:
        st.error(f"❌ 크롤링 중 에러 발생: {e}")
        return None

# 텍스트 요약 함수
def summarize_text(texts):
    if texts:
        summarizer = pipeline("summarization")
        summaries = []
        
        for text in texts:
            if text.strip():  
                try:
                    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    summaries.append(f"요약 중 오류 발생: {e}")
            else:
                summaries.append("내용 없음")

        return "\n\n".join(summaries)
    else:
        return "요약할 내용이 없습니다."

# Streamlit 앱 구성
st.title("📈 기업 뉴스 요약")
st.subheader("기업명을 입력하여 해당 기업의 최신 뉴스를 요약해줍니다.")

# 사용자 입력
search_query = st.text_input("검색할 기업명을 입력하세요:", value="삼성전자")
search_days = st.slider("검색 기간 (일 단위)", 1, 30, 7)

# 검색 버튼
if st.button("뉴스 요약"):
    with st.spinner("뉴스를 수집 중입니다..."):
        try:
            today = datetime.today()
            start_date = today - timedelta(days=search_days)
            start_date_str = start_date.strftime('%Y%m%d')
            end_date_str = today.strftime('%Y%m%d')

            # 뉴스 크롤링 실행 (최대 5페이지 가져오기)
            result = 네이버_뉴스_크롤링(search_query, start_date_str, end_date_str, 페이지_수=5)

            if result:
                titles, channels, links, contents = result
                st.success(f"{len(titles)}개의 뉴스를 가져왔습니다.")

                # 뉴스 내용 요약
                summary = summarize_text(contents)

                st.markdown("### 기업 뉴스 요약")
                st.write(summary)
            else:
                st.warning("관련 뉴스를 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")












