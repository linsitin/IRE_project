import os
import re
from collections import Counter

try:
    print("⏳ 嘗試載入 CKIP 套件中...")
    # 這次同時載入 Word Segmenter (WS) 與 NER
    from ckip_transformers.nlp import CkipWordSegmenter, CkipNerChunker
    print("✅ CKIP 套件載入成功！")
except ImportError:
    print("❌ 找不到 ckip-transformers 套件！請在終端機輸入：pip install ckip-transformers")
    exit()

def get_2025_articles(folder_path):
    articles = []
    
    if not os.path.exists(folder_path):
        print(f"❌ 找不到資料夾：{os.path.abspath(folder_path)}")
        return articles
        
    print(f"📂 成功找到資料夾，開始讀取文章...")
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    for filename in file_list:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            if "2025" not in content:
                continue 
            
            text_parts = content.split("--------------------------------------------------")
            main_text = text_parts[-1].strip() if len(text_parts) > 1 else content
            
            # 【資料清洗】移除正文中的 Metadata 雜訊
            main_text = re.sub(r'(發佈|更新|報告編號|查核記者|責任編輯|記者|編輯)[：:].+', '', main_text)
            
            articles.append({"filename": filename, "text": main_text})
            
    return articles

def run_yearly_trend_analysis():
    print("\n--- 🚀 程式開始執行 ---")
    articles = get_2025_articles("tfc_reports")
    
    if len(articles) == 0:
        print("❌ 因為沒有符合條件的文章，程式提早結束。")
        return

    print("🤖 正在載入 WS (斷詞) 與 NER (實體辨識) 模型 (albert-base)...")
    ws_driver = CkipWordSegmenter(model="albert-base")
    ner_driver = CkipNerChunker(model="albert-base")
    
    # ==========================================
    # 🔬 新增：NLP 分析觀察區 (僅取第一篇文章示範)
    # ==========================================
    sample_text = articles[0]["text"][:200] + "..." # 只取前200字避免洗版
    print("\n" + "▼"*50)
    print(f"🔬 【NLP 底層解析示範】文章：{articles[0]['filename']}")
    print("▼"*50)
    print(f"\n📝 1. 原始文本：\n{sample_text}\n")
    
    # 測試斷詞
    ws_res = ws_driver([sample_text])
    print(f"✂️  2. 斷詞結果 (WS)：\n{' | '.join(ws_res[0])}\n")
    
    # 測試實體辨識
    ner_res_sample = ner_driver([sample_text])
    print(f"🏷️  3. 實體辨識結果 (NER)：")
    for entity in ner_res_sample[0]:
        print(f"   ➤ 抓到實體：【{entity.word}】 (標籤：{entity.ner})")
    print("▲"*50 + "\n")
    # ==========================================

    trends_2025 = []
    
    # 過濾名單
    exclude_list = [
        "陳偉婷", "陳培煌", "曾慧雯", "邱劭安", "許雲凱", "陳柏宇", "馬麗昕", 
        "查核中心", "台灣事實查核中心", "事實查核中心", "查核記者", "責任編輯"
    ]
    
    print("🔍 開始進行所有文章的實體統計，請稍候...")
    texts = [art["text"] for art in articles]
    
    # 全文跑 NER
    ner_results = ner_driver(texts, batch_size=2)
    
    for entity_list in ner_results:
        for entity in entity_list:
            if entity.ner in ["PERSON", "ORG"]:
                if len(entity.word) > 1 and entity.word not in exclude_list: 
                    trends_2025.append(entity.word)

    print("\n" + "="*50)
    print("📈 2025 年度事實查核報告：最常遭造謠或牽涉之實體 Top 10")
    print("="*50)
    
    counter = Counter(trends_2025)
    top_entities = counter.most_common(10)
    
    for rank, (word, count) in enumerate(top_entities, start=1):
        print(f"  第 {rank} 名： {word} (出現 {count} 次)")
    print("-" * 50)
    print("🎉 程式順利執行完畢！")

if __name__ == "__main__":
    run_yearly_trend_analysis()