import os
import re
from collections import defaultdict, Counter
from ckip_transformers.nlp import CkipNerChunker

def get_article_data(folder_path):
    """讀取資料夾內的所有文章，萃取日期與純文字內容"""
    articles = []
    print(f"📂 正在讀取 {folder_path} 內的文章...")
    
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
            
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 利用正規表達式抓取「發佈：YYYY-MM-DD」
            date_match = re.search(r'發佈：(\d{4}-\d{2})', content)
            month = date_match.group(1) if date_match else "未知時間"
            
            # 清理文字，移除開頭的 Metadata
            text_parts = content.split("--------------------------------------------------")
            main_text = text_parts[-1].strip() if len(text_parts) > 1 else content
            
            articles.append({"filename": filename, "month": month, "text": main_text})
            
    print(f"✅ 共讀取 {len(articles)} 篇文章")
    return articles

def run_trend_analysis():
    # 1. 載入模型 (第一次執行會自動下載模型檔，需要一點時間)
    print("🤖 正在載入 CKIP NER 模型 (albert-base)...")
    ner_driver = CkipNerChunker(model="albert-base")
    
    # 2. 準備資料
    articles = get_article_data("tfc_reports")
    trends = defaultdict(list)
    
    # 3. 處理每篇文章
    print("🔍 開始進行命名實體辨識 (NER)...")
    texts = [art["text"] for art in articles]
    
    # 批次執行 NER (設定 batch_size 可防記憶體爆炸)
    ner_results = ner_driver(texts, batch_size=4)
    
    for i, entity_list in enumerate(ner_results):
        month = articles[i]["month"]
        for entity in entity_list:
            # 只提取人名 (PERSON) 與組織機構 (ORG)
            if entity.ner in ["PERSON", "ORG"]:
                # 過濾掉太短的雜訊
                if len(entity.word) > 1: 
                    trends[month].append(entity.word)

    # 4. 統計與輸出結果
    print("\n" + "="*40)
    print("📈 各月份造謠趨勢實體分析結果")
    print("="*40)
    
    # 依照月份排序輸出
    for month in sorted(trends.keys()):
        counter = Counter(trends[month])
        # 抓取該月出現最多次的前 5 個實體
        top_entities = counter.most_common(5)
        
        print(f"【{month}】熱門實體：")
        for word, count in top_entities:
            print(f"  - {word} (出現 {count} 次)")
        print("-" * 20)

if __name__ == "__main__":
    run_trend_analysis()