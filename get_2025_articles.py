import os
import re
from collections import Counter

try:
    print("⏳ 嘗試載入 CKIP 套件中...")
    from ckip_transformers.nlp import CkipWordSegmenter
    print("✅ CKIP 套件載入成功！")
except ImportError:
    print("❌ 找不到 ckip-transformers 套件！請在終端機輸入：pip install ckip-transformers")
    exit()

def get_2025_articles(folder_path):
    articles = []
    
    if not os.path.exists(folder_path):
        print(f"❌ 找不到資料夾：{os.path.abspath(folder_path)}")
        return articles
        
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    print(f"📂 開始利用「報告編號 (3416~3962)」精準篩選 2025 年資料...")
    
    for filename in file_list:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 【全新精準篩選】利用正規表達式尋找報告編號
            match = re.search(r'報告編號[：:]\s*(\d+)', content)
            if match:
                report_num = int(match.group(1))
                # 如果編號不在 3416 到 3962 之間，就跳過這篇文章
                if not (3416 <= report_num <= 3962):
                    continue 
            else:
                # 如果這篇文章找不到報告編號，跳過以防萬一
                continue
            
            text_parts = content.split("--------------------------------------------------")
            main_text = text_parts[-1].strip() if len(text_parts) > 1 else content
            
            # 清洗 Metadata 雜訊
            main_text = re.sub(r'(發佈|更新|報告編號|查核記者|責任編輯|記者|編輯)[：:].+', '', main_text)
            articles.append(main_text)
            
    print(f"✅ 篩選完畢！共抓出 {len(articles)} 篇 2025 年的文章。")
    return articles

def run_word_frequency_analysis():
    print("\n--- 🚀 詞頻統計程式開始執行 ---")
    texts = get_2025_articles("tfc_reports")
    
    if len(texts) == 0:
        print("❌ 找不到符合條件的文章。")
        return

    print("🤖 正在載入 WS (斷詞) 模型 (albert-base)...")
    ws_driver = CkipWordSegmenter(model="albert-base", device=0)
    
    # 停用詞表
    stopwords = {
        "我們", "你們", "他們", "這個", "那個", "可以", "因為", "所以", 
        "如果", "就是", "還是", "沒有", "一個", "什麼", "這些", "那些",
        "表示", "指出", "影片", "網傳", "照片", "內容", "訊息", "圖片",
        "查核", "中心", "台灣", "發現", "結果", "進行", "出現", "相關",
        "報導", "部分", "不是", "目前", "流傳", "已經", "可能", "看到"
    }
    
    all_words = []
    
    print("✂️ 開始對所有文章進行斷詞，請稍候...")
    ws_results = ws_driver(texts, batch_size=2)
    
    for word_list in ws_results:
        for word in word_list:
            if len(word) > 1 and re.match(r'^[\w\u4e00-\u9fa5]+$', word) and word not in stopwords:
                all_words.append(word)

    print("\n" + "="*50)
    print("📊 2025 年度事實查核報告：全文本高頻詞彙 Top 20")
    print("="*50)
    
    counter = Counter(all_words)
    top_words = counter.most_common(20)
    
    for rank, (word, count) in enumerate(top_words, start=1):
        print(f"  第 {rank:2d} 名： 【{word:^6}】 (共出現 {count} 次)")
        
    print("-" * 50)
    
    print("\n[進階查詢]")
    target_word = "詐騙"
    target_count = counter.get(target_word, 0)
    print(f"🔍 你特別關注的詞彙【{target_word}】，在 2025 年共出現了 {target_count} 次！")

if __name__ == "__main__":
    run_word_frequency_analysis()