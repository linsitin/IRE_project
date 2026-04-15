import requests
from bs4 import BeautifulSoup
import time
import os
import re

def clean_filename(filename):
    """移除 Windows/Mac 不允許在檔名中出現的特殊字元"""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def crawl_tfc_comprehensive(total_pages=2): # 預設先抓 2 頁測試，你可以改成更大的數字
    list_base_url = "https://tfc-taiwan.org.tw/fact-check-reports-all/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # 1. 建立存放文章的資料夾
    output_folder = "tfc_reports"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 已建立資料夾：{output_folder}")

    article_count = 0

    # 2. 開始分頁迴圈
    for page in range(1, total_pages + 1):
        print(f"\n--- 📑 正在處理第 {page} 頁 ---")
        
        # 根據 Kadence 系統常見的分頁網址格式（或是標準 WordPress 格式）
        # 如果第一頁不需要參數，我們判斷一下
        if page == 1:
            current_url = list_base_url
        else:
            # TFC 網站的分頁通常使用 ?page=X 或 ?query-141955-page=X
            current_url = f"{list_base_url}?pg={page}"
            
        try:
            response = requests.get(current_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 抓取該頁所有的文章卡片
            article_cards = soup.select('li.kb-query-item')
            
            if not article_cards:
                print("停止：找不到文章卡片，可能是已經到最後一頁了。")
                break

            for card in article_cards:
                # 取得標題與連結
                link_tag = card.select_one('a.kb-section-link-overlay')
                title_tag = card.select_one('.kb-dynamic-html')
                
                if link_tag and title_tag:
                    title = title_tag.get_text(strip=True)
                    link = link_tag.get('href')
                    article_count += 1
                    
                    print(f"  👉 正在處理：{title[:20]}...")
                    
                    # 3. 進入文章內頁抓取內容
                    inner_res = requests.get(link, headers=headers)
                    inner_soup = BeautifulSoup(inner_res.text, 'html.parser')
                    
                    content_area = inner_soup.select_one('.entry-content')
                    paragraphs = content_area.find_all('p') if content_area else inner_soup.find_all('p')
                    full_text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

                    # 4. 寫入個別檔案
                    safe_title = clean_filename(title)
                    filename = os.path.join(output_folder, f"{safe_title}.txt")
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"標題：{title}\n")
                        f.write(f"來源：{link}\n")
                        f.write("-" * 30 + "\n\n")
                        f.write(full_text)
                    
                    # ⚠️ 禮儀暫停，避免被鎖 IP
                    time.sleep(1.5)

        except Exception as e:
            print(f"❌ 處理第 {page} 頁時發生錯誤：{e}")
            break

    print(f"\n✅ 任務完成！共下載 {article_count} 篇文章，存放於 {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    # 你可以在這裡決定要抓幾頁，例如想抓前 5 頁就填 5
    crawl_tfc_comprehensive(total_pages=2)