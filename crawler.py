import requests
from bs4 import BeautifulSoup
import time
import os
import re

def clean_filename(filename):
    """移除 Windows/Mac 不允許在檔名中出現的特殊字元"""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def get_total_pages(url, headers):
    """自動偵測網站的總頁數"""
    print("🕵️‍♂️ 正在偵測網站總頁數...")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 策略 1：找尋帶有 data-max-num-pages 的容器
        query_card = soup.select_one('.wp-block-kadence-query-card')
        if query_card and query_card.has_attr('data-max-num-pages'):
            max_pages = int(query_card['data-max-num-pages'])
            print(f"✅ 成功偵測到總頁數：{max_pages} 頁")
            return max_pages
            
        # 策略 2：找分頁按鈕裡面的最大數字 (備用方案)
        page_links = soup.select('.page-numbers[data-page]')
        if page_links:
            max_pages = max([int(link['data-page']) for link in page_links if link['data-page'].isdigit()])
            print(f"✅ 成功偵測到總頁數：{max_pages} 頁")
            return max_pages
            
    except Exception as e:
        print(f"⚠️ 偵測總頁數失敗：{e}")
        
    print("⚠️ 找不到分頁資訊，預設只抓取 1 頁。")
    return 1

def crawl_tfc_comprehensive(scrape_all=False, limit_pages=3): 
    list_base_url = "https://tfc-taiwan.org.tw/fact-check-reports-all/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # 1. 建立存放文章的資料夾
    output_folder = "tfc_reports"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 2. 決定要爬幾頁
    if scrape_all:
        total_pages = get_total_pages(list_base_url, headers)
        print(f"🚨 警告：準備爬取全部共 {total_pages} 頁，這可能會花費數小時！")
    else:
        total_pages = limit_pages
        print(f"ℹ️ 測試模式：僅設定爬取前 {total_pages} 頁。")

    article_count = 0

    # 3. 開始分頁迴圈
    for page in range(1, total_pages + 1):
        print(f"\n--- 📑 正在處理第 {page} / {total_pages} 頁 ---")
        
        current_url = list_base_url if page == 1 else f"{list_base_url}?pg={page}"
            
        try:
            response = requests.get(current_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            article_cards = soup.select('li.kb-query-item')
            if not article_cards:
                print("停止：找不到文章卡片，可能是已經到最後一頁了。")
                break

            for card in article_cards:
                link_tag = card.select_one('a.kb-section-link-overlay')
                title_tag = card.select_one('.kb-dynamic-html')
                
                if link_tag and title_tag:
                    title = title_tag.get_text(strip=True)
                    link = link_tag.get('href')
                    article_count += 1
                    
                    print(f"  👉 正在處理 ({article_count})：{title[:20]}...")
                    
                    # 進入文章內頁抓取內容
                    inner_res = requests.get(link, headers=headers)
                    inner_soup = BeautifulSoup(inner_res.text, 'html.parser')
                    content_area = inner_soup.select_one('.entry-content')
                    
                    if content_area:
                        # 清除雜訊 (目錄、分享按鈕等)
                        for unwanted in content_area.select('.kb-table-of-content-nav, .wp-block-outermost-social-sharing, .kt-blocks-info-box-link-wrap'):
                            unwanted.decompose()
                        for p in content_area.find_all('p'):
                            text = p.get_text(strip=True)
                            if "事實查核需要你的一份力量" in text or "本中心查核作業獨立進行" in text or "查核結果說明：" in text:
                                p.decompose()

                        tags_to_find = ['p', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']
                        elements = content_area.find_all(tags_to_find)
                    else:
                        elements = inner_soup.find_all(['p', 'h2', 'h3'])
                    
                    full_text = "\n\n".join([el.get_text(strip=True) for el in elements if el.get_text(strip=True)])

                    # 寫入個別檔案
                    safe_title = clean_filename(title)
                    filename = os.path.join(output_folder, f"{safe_title}.txt")
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"標題：{title}\n")
                        f.write(f"來源：{link}\n")
                        f.write("-" * 50 + "\n\n")
                        f.write(full_text)
                    
                    # ⚠️ 禮儀暫停，保護 IP
                    time.sleep(1.5)

        except Exception as e:
            print(f"❌ 處理第 {page} 頁時發生錯誤：{e}")
            break

    print(f"\n✅ 任務完成！共下載 {article_count} 篇文章，存放於 {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    # 如果你想一次抓完 400 多頁，把 scrape_all 改成 True
    # 如果只是測試，維持 False，它只會抓 limit_pages 裡面設定的頁數
    crawl_tfc_comprehensive(scrape_all=True, limit_pages=2)