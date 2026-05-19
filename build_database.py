import os
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from ckip_transformers.nlp import CkipWordSegmenter
except ImportError:
    print("請安裝 ckip-transformers")
    exit()

def build_database_max_df(folder_path="tfc_reports_api", output_file="tfc_database_maxdf.pkl"):
    print("==================================================")
    print("  開始建立查核文章語料庫 ")
    print("==================================================")
    
    articles = []
    titles = []
    dates = []
    filepaths = []
    
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    print(f" 1. 讀取 {len(file_list)} 篇本機文章...")
    for filename in file_list:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            title = content.split('\n')[0].replace('標題：', '').strip()
            date_match = re.search(r'\[(\d{4}-\d{2}-\d{2})\]', filename)
            date = date_match.group(1) if date_match else "未知日期"
            
            text_parts = content.split("==================================================")
            main_text = text_parts[-1].strip() if len(text_parts) > 1 else content.strip()
            main_text = re.sub(r'(發佈|更新|查核記者|責任編輯|記者|編輯)[：:].+', '', main_text)
            
            if main_text:
                articles.append(main_text)
                titles.append(title)
                dates.append(date)
                filepaths.append(filepath)

    print(" 2. 載入 CKIP 斷詞模型 (啟動 GPU)...")
    ws_driver = CkipWordSegmenter(model="albert-base", device=0)
    
    print(" 3. 開始深度斷詞 (請耐心等候進度條跑完)...")
    ws_results = ws_driver(articles, batch_size=16)

    # 不保留最基本的人工語氣停用詞
    # stopwords = {"我們", "可以", "因為", "所以", "這個", "那個", "沒有", "什麼", 
    #              "表示", "指出", "影片", "網傳", "照片", "內容", "訊息", "圖片",
    #              "查核", "中心", "台灣", "發現", "結果", "進行", "出現", "相關",
    #              "報導", "部分", "不是", "目前", "流傳", "已經", "可能", "看到"}

    print(" 4. 進行初步清洗 (僅過濾單字與基本格式)...")
    corpus = []
    for word_list in ws_results:
        filtered = [w for w in word_list if len(w) > 1 and re.match(r'^[\w\u4e00-\u9fa5]+$', w) ]
        corpus.append(" ".join(filtered))

    # 真實 DF (涵蓋率) 
    print("\n [觀測] 正在計算全資料庫的 DF 排行...")
    from collections import Counter
    df_counter = Counter()
    for doc in corpus:
        # 將單篇文章拆回字詞，並用 set 去重複 (因為 DF 只算篇數，不算單篇出現次數)
        unique_words_in_doc = set(doc.split())
        df_counter.update(unique_words_in_doc)

    total_docs = len(corpus)
    print("\n 【全資料庫最常出現的 Top 100 詞彙真實涵蓋率】")
    for rank, (word, count) in enumerate(df_counter.most_common(100), 1):
        percentage = (count / total_docs) * 100
        print(f"第 {rank:2d} 名 | 【{word:^6}】 | 出現 {count:4d} 篇 | 涵蓋率: {percentage:5.1f}%")
    print("==================================================\n")

    # 自己當 max_df
    max_df_threshold = 0.5  # 設定 50% 門檻 (你隨時可以改成 0.5 或 0.4)
    min_count_to_kill = total_docs * max_df_threshold
    
    # 從我們的觀測儀裡面，把出現次數超過門檻的詞全部抓出來，列入名單！
    auto_stop_words = [word for word, count in df_counter.items() if count > min_count_to_kill]
    
    print(f" 自行處理 {len(auto_stop_words)} 個超過 {max_df_threshold*100}% 的高頻廢詞：")
    if len(auto_stop_words) > 0:
        # 照字數排個版，讓畫面好看一點
        auto_stop_words.sort(key=lambda x: (len(x), x))
        print("、".join(auto_stop_words))
    else:
        print(" 沒有詞彙超過門檻。")

    print("\n 5. 建立 TF-IDF 語意矩陣 ( 強制停用詞名單 )...")
    
    # 既然底層套件無法執行，我們就直接把名單塞給它的 stop_words 參數，強迫執行！
    vectorizer = TfidfVectorizer(
        stop_words=auto_stop_words if len(auto_stop_words) > 0 else None, 
        token_pattern=r"[^\s]+"
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 這裡我們就不需要再去印 vectorizer.stop_words_ 了，因為上面我們已經自己印出來了
    
    print(f"\n  6. 將所有資料打包儲存為 {output_file}...")
    db_data = {
        'titles': titles,
        'dates': dates,
        'filepaths': filepaths,
        'tfidf_matrix': tfidf_matrix,
        'vectorizer': vectorizer,
        'corpus': [doc.split() for doc in corpus] # 把斷好詞的清單存起來
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(db_data, f)
        
    print(" 資料庫建立完成！")

if __name__ == "__main__":
    build_database_max_df()