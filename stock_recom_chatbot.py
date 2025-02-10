import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import urllib.parse
import re
from transformers import pipeline

# 유사 단어 필터링을 위한 정규화 함수
def normalize_word(word):
    return re.sub(r'[^a-zA-Z가-힣]', '', word).lower()

# 뉴스 크롤링 함수 (페이지 추가)
def crawl_naver_news(company_name, start_date, end_date, max_pages=5):
    try:
        encoded_query = urllib.parse.quote(company_name)
        date_filter = f"nso=so:r,p:from{start_date}to{end_date}"
        base_url = f"https://search.naver.com/search.naver?where=news&sm=tab_jum&query={encoded_query}&{date_filter}"

        titles, contents = [], []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        # 페이지를 순차적으로 가져오기
        for page in range(1, max_pages + 1):
            url = f"{base_url}&start={1 + (page - 1) * 10}"  # start=1 부터 10개씩
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            news_items = soup.select("ul.list_news > li")
            
            for item in news_items:
                title_elem = item.select_one("a.news_tit")
                title = title_elem.text.strip() if title_elem else "제목 없음"
                content_elem = item.select_one("div.news_dsc")
                content = content_elem.text.strip() if content_elem else ""

                # 필터링 조건: 기업명 포함
                if company_name in title or company_name in content:
                    titles.append(title)
                    contents.append(content)

        return titles, contents
    except Exception as e:
        st.error(f"크롤링 오류 발생: {e}")
        return None

# 텍스트 요약 함수
def summarize_text(texts):
    # 내용이 비어있지 않으면 요약
    if texts:
        summarizer = pipeline("summarization")
        summaries = []
        
        # 각 뉴스 기사를 개별적으로 요약
        for text in texts:
            if text.strip():  # 비어있는 텍스트는 건너뜀
                try:
                    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    summaries.append(f"요약 중 오류 발생: {e}")
            else:
                summaries.append("내용 없음")

        # 모든 요약을 하나로 합침
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
            # 날짜 계산
            today = datetime.today()
            start_date = today - timedelta(days=search_days)
            start_date_str = start_date.strftime('%Y%m%d')
            end_date_str = today.strftime('%Y%m%d')

            # 뉴스 크롤링 실행 (최대 5페이지 가져오기)
            result = crawl_naver_news(search_query, start_date_str, end_date_str, max_pages=5)

            if result:
                titles, contents = result
                st.success(f"{len(titles)}개의 뉴스를 가져왔습니다.")

                # 뉴스 내용 요약
                summary = summarize_text(contents)

                st.markdown("### 기업 뉴스 요약")
                st.write(summary)
            else:
                st.warning("관련 뉴스를 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")











