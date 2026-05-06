import os
import re
import numpy as np
import unicodedata
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from ckip_transformers.nlp import CkipWordSegmenter
except ImportError:
    print("請安裝 ckip-transformers")
    exit()

def get_display_width(s):
    """計算字串的視覺寬度（中文字算 2 格，英文/數字算 1 格）"""
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in str(s))

def pad_str(s, total_width, align='center'):
    """根據視覺寬度，自動補齊空白字元"""
    s = str(s)
    current_width = get_display_width(s)
    padding = total_width - current_width
    if padding <= 0: return s
    
    if align == 'center':
        left_pad = padding // 2
        right_pad = padding - left_pad
        return ' ' * left_pad + s + ' ' * right_pad
    elif align == 'left':
        return s + ' ' * padding
    elif align == 'right':
        return ' ' * padding + s

def get_2021_articles(folder_path):
    articles = []
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    for filename in file_list:
        if "[2021" in filename:
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                text_parts = content.split("==================================================")
                if len(text_parts) > 1:
                    main_text = text_parts[-1].strip()
                else:
                    main_text = content.strip()
                    
                main_text = re.sub(r'(發佈|更新|查核記者|責任編輯|記者|編輯)[：:].+', '', main_text)
                if main_text: 
                    articles.append(main_text)
    return articles

def run_hybrid_trend_analysis_top10():
    print("\n---  TF-IDF (Top 10) + DF 趨勢分析 ---")
    
    texts = get_2021_articles("tfc_reports_api")
    total_docs = len(texts)
    if total_docs == 0: return

    print(" 載入 CKIP 斷詞模型 ( 已啟用 GPU 硬體加速 device=0)...")
    ws_driver = CkipWordSegmenter(model="albert-base", device=0)
    
    print(f" 1. 開始對全部 {total_docs} 篇文章進行斷詞...")
    ws_results = ws_driver(texts, batch_size=16)
    
    stopwords = {"我們", "可以", "因為", "所以", "這個", "那個", "沒有", "什麼", 
                 "表示", "指出", "影片", "網傳", "照片", "內容", "訊息", "圖片",
                 "查核", "中心", "台灣", "發現", "結果", "進行", "出現", "相關",
                 "報導", "部分", "不是", "目前", "流傳", "已經", "可能", "看到"}
    
    corpus = []
    for word_list in ws_results:
        filtered = [w for w in word_list if len(w) > 1 and re.match(r'^[\w\u4e00-\u9fa5]+$', w) and w not in stopwords]
        corpus.append(" ".join(filtered))

    print(" 2. 計算 TF-IDF 矩陣...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = np.array(vectorizer.get_feature_names_out())

    print(" 3. 嚴格提取每篇文章的 Top 10 核心關鍵字，並進行大會師統計...")
    trend_counter = Counter()
    
    for i in range(total_docs):
        row = tfidf_matrix[i].toarray()[0]
        top_indices = np.argsort(row)[-10:] 
        top_words = [feature_names[idx] for idx in top_indices if row[idx] > 0]
        trend_counter.update(top_words)

    print("\n" + "="*70)
    print(" 2021 年度事實查核報告：核心話題趨勢 Top 20")
    print("(指標：該詞彙成為單篇文章『Top 10 核心關鍵字』的總篇數)")
    print("="*70)
    
    # 完美對齊的表頭
    print(f"{pad_str('排名', 8)} | {pad_str('核心話題關鍵字', 16)} | {pad_str('成為核心的篇數', 16)} | {pad_str('話題強度 (%)', 12)}")
    print("-" * 70)
    
    top_trends = trend_counter.most_common(20)
    
    for rank, (word, count) in enumerate(top_trends, start=1):
        power = (count / total_docs) * 100
        
        # 呼叫 pad_str 把每一欄都精準鎖定寬度
        rank_str = pad_str(f"第 {rank} 名", 8)
        word_str = pad_str(f"【 {word} 】", 16)
        count_str = pad_str(count, 16)
        power_str = pad_str(f"{power:.1f} %", 12)
        
        print(f"{rank_str} | {word_str} | {count_str} | {power_str}")
        
    print("-" * 70)

if __name__ == "__main__":
    run_hybrid_trend_analysis_top10()