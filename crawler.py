import requests
import os
import re
import time
from bs4 import BeautifulSoup

def clean_filename(filename):
    """移除 Windows/Mac 不允許在檔名中出現的特殊字元"""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def clean_content_tags(html_content):
    """強效過濾：專門清洗 API 回傳的「內文」雜訊"""
    if not html_content:
        return ""
        
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. 移除「雜訊區塊」 
    junk_selectors = [
        '.wp-block-outermost-social-sharing', # Share 按鈕
        '.kb-table-of-content-nav',           # 目錄
        '.kt-blocks-info-box-link-wrap',      # 查核結果大圖示
        '.kb-dynamic-list'                    # 分類標籤
    ]
    for selector in junk_selectors:
        for el in soup.select(selector):
            el.decompose()
            
    # 2. 移除「重複的行動版/電腦版隱藏區塊」
    for hidden in soup.select('.kb-v-md-hidden, .kb-v-sm-hidden, .kb-v-lg-hidden'):
        hidden.decompose()
        
    # 3. 過濾不要的廣告或宣告段落
    for p in soup.find_all('p'):
        text = p.get_text(strip=True)
        if "事實查核需要你的一份力量" in text or "本中心查核作業獨立進行" in text or "查核結果說明：" in text:
            p.decompose()

    # 4. 只精準抓取標題、段落與清單
    elements = soup.find_all(['p', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
    
    return "\n\n".join([el.get_text(strip=True) for el in elements if el.get_text(strip=True)])

def crawl_tfc_api(total_pages=1):
    api_url = "https://tfc-taiwan.org.tw/wp-json/wp/v2/fact-check-reports"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    output_folder = "tfc_reports_api"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    article_count = 0
    
    for page in range(1, total_pages + 1):
        print(f"\n---  正在透過 API 請求第 {page} 頁資料 ---")
        
        params = {"page": page, "per_page": 9}
        
        try:
            response = requests.get(api_url, headers=headers, params=params)
            
            if response.status_code == 400:
                print(" 已經到底了，沒有更多文章可以抓取。")
                break
                
            response.raise_for_status()
            articles = response.json()
            
            if not articles:
                print(" 這一頁是空的。")
                break
                
            for article in articles:
                article_count += 1
                
                # 從 JSON 取出原始資料
                raw_title = article.get('title', {}).get('rendered', '未命名標題')
                link = article.get('link', '')
                date = article.get('date', '')[:10]
                raw_content = article.get('content', {}).get('rendered', '')
                
                # 標題用簡單的 BeautifulSoup 解析純文字就好
                title = BeautifulSoup(raw_title, 'html.parser').get_text(strip=True)
                
                # 內文過濾雜訊
                full_text = clean_content_tags(raw_content)
                
                print(f"   正在存檔 ({article_count}): {title[:20]}...")
                
                # 寫入檔案
                safe_title = clean_filename(title)
                filename = os.path.join(output_folder, f"[{date}] {safe_title}.txt")
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"標題：{title}\n")
                    f.write(f"發布日期：{date}\n")
                    f.write(f"來源網址：{link}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(full_text)
                    
            time.sleep(1)
            
        except Exception as e:
            print(f" 發生錯誤: {e}")
            break

    print(f"\n 任務完成！共下載 {article_count} 篇文章，存放於 {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    # 抓取第幾頁
    crawl_tfc_api(total_pages=460)