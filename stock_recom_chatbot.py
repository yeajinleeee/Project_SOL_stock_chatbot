import requests
from bs4 import BeautifulSoup
import time
import random
import urllib.parse
import re
from datetime import datetime, timedelta

# 유사 단어 필터링을 위한 정규화 함수
def 정규화_단어(단어):
    # 숫자나 특수문자 제거하고 소문자화
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
            # 기업명을 제목에서 제외하고 단어 단위로 분리
            title_without_company = title.replace(기업명, '').strip()
            words = set(정규화_단어(word) for word in title_without_company.split())
            
            # 제목이 이미 등장했거나, 제목에 포함된 단어들이 이미 등장한 단어라면 필터링
            if title not in seen_titles and not seen_words & words:
                filtered_titles.append(title)
                filtered_channels.append(channel)
                filtered_links.append(link)
                filtered_contents.append(content)
                seen_titles.add(title)  # 새로운 제목은 seen_titles에 추가
                seen_words.update(words)  # 새로운 단어들은 seen_words에 추가

        # 결과 반환
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

# 사용자로부터 기업명과 검색할 날짜 범위를 입력 받기
search_query = input("검색할 기업명을 입력하세요: ")
search_days = int(input("며칠 동안의 기사를 검색할까요? (예: 3 입력 시 3일 전부터 오늘까지): "))

# 날짜 계산
today = datetime.today()
start_date = today - timedelta(days=search_days)
start_date_str = start_date.strftime('%Y%m%d')
end_date_str = today.strftime('%Y%m%d')

# 뉴스 크롤링 함수 실행
result = 네이버_뉴스_크롤링(search_query, start_date_str, end_date_str)

if result:
    titles, channels, links, contents = result
    for title, channel, link, content in zip(titles, channels, links, contents):
        print(f"\n\n제목: {title}\n언론사: {channel}\n링크: {link}\n본문: {content[:100]}...\n")
else:
    print("❌ 크롤링에 실패했습니다.")













