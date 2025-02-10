import requests
from bs4 import BeautifulSoup
import time
import random
import urllib.parse
import re
from datetime import datetime, timedelta
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import torch

# 네이버 뉴스 크롤링 함수

def 네이버_뉴스_크롤링(기업명, 시작날짜, 종료날짜, 페이지_수=5):
    try:
        encoded_query = urllib.parse.quote(기업명)
        date_filter = f"nso=so:r,p:from{시작날짜}to{종료날짜}"
        titles, contents = [], []
        
        for page in range(1, 페이지_수 + 1):
            url = f"https://search.naver.com/search.naver?where=news&query={encoded_query}&{date_filter}&start={((page-1)*10) + 1}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            time.sleep(random.uniform(1, 3))
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = soup.select("ul.list_news > li")
            
            for item in news_items:
                title_element = item.select_one("a.news_tit")
                content_element = item.select_one("div.news_dsc")
                title = title_element.text.strip() if title_element else ""
                content = content_element.text.strip() if content_element else ""
                
                if 기업명 in title or 기업명 in content:
                    titles.append(title)
                    contents.append(content)
        
        return titles, contents if titles else None
    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 에러 발생: {e}")
        return None

# SBERT 임베딩 모델
embedding_model = SentenceTransformer("jhgan/ko-sbert-sts")

def create_embeddings(texts):
    embeddings = embedding_model.encode(texts, convert_to_tensor=True)
    return FAISS.from_embeddings(embeddings, texts)

def retrieve_relevant_sentences(query, vector_store):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    results = vector_store.similarity_search_by_vector(query_embedding, k=5)
    return [result.page_content for result in results]

# 뉴스 크롤링 실행
search_query = input("검색할 기업명을 입력하세요: ")
search_days = int(input("며칠 동안의 기사를 검색할까요? "))
today = datetime.today()
start_date = today - timedelta(days=search_days)
start_date_str = start_date.strftime('%Y%m%d')
end_date_str = today.strftime('%Y%m%d')

news_data = 네이버_뉴스_크롤링(search_query, start_date_str, end_date_str)
if news_data:
    titles, contents = news_data
    texts = [f"{title} {content}" for title, content in zip(titles, contents)]
    vector_store = create_embeddings(texts)
    
    user_query = input("요약할 내용을 입력하세요: ")
    relevant_sentences = retrieve_relevant_sentences(user_query, vector_store)
    print("\n".join(relevant_sentences))
















