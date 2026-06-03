import pickle
import numpy as np
from gensim.models import FastText
import os
import re

try:
    from ckip_transformers.nlp import CkipWordSegmenter
except ImportError:
    print("請確認已安裝 ckip-transformers")
    exit()

def train_and_save_fasttext():
    # 1. 讀取資料庫
    db_file = "tfc_database_maxdf0.5.pkl" # 請確認檔名正確 (例如 0.5 或 0.33)
    print(f" 讀取 {db_file}...")
    with open(db_file, 'rb') as f:
        db_data = pickle.load(f)
    
    # db_data['corpus'] 已經是你的「內文 (全文-標題)」了
    corpus_tokens = db_data['corpus'] 
    raw_titles = db_data['titles']  # 原始帶有【錯誤】的標題
    filepaths = db_data['filepaths']
    dates = db_data['dates']

    print(f" 讀取到 {len(corpus_tokens)} 篇文章。")

    #  處理：標題物理剝離 (將查核標籤移除)
    clean_titles = []
    print(" 正在清理標題，抽離【查核結果】標籤...")
    for title in raw_titles:
        # 正則表達式：尋找開頭是【】的字串，並只取後面的乾淨標題
        match = re.match(r'^【(.*?)】(.*)', title)
        if match:
            clean_titles.append(match.group(2).strip())
        else:
            clean_titles.append(title.strip())

    # 呼叫 CKIP 處理『乾淨的標題』
    print(" 載入 CKIP 模型 (為了處理乾淨標題的斷詞)...")
    ws_driver = CkipWordSegmenter(model="albert-base", device=0)
    
    print(" 正在使用 CKIP 進行標題斷詞...")
    title_ws_results = ws_driver(clean_titles, batch_size=16)
    
    title_tokens = []
    for word_list in title_ws_results:
        filtered = [w for w in word_list if w.strip()]
        title_tokens.append(filtered)
    print(" 標題斷詞完成！")

    # 2. 訓練模型
    print("\n 訓練 FastText 語意模型中...")
    model = FastText(sentences=corpus_tokens, vector_size=100, window=5, min_count=2, workers=os.cpu_count(), epochs=10)
    
    # 3. 方法一：計算【內文 (全文-標題)】向量 
    print(" 計算【內文】向量中...")
    doc_vectors = []
    for tokens in corpus_tokens:
        word_vectors = [model.wv[w] for w in tokens if w in model.wv]
        if word_vectors:
            doc_vectors.append(np.mean(word_vectors, axis=0))
        else:
            doc_vectors.append(np.zeros(100))
            
    # 4. 方法二：計算【乾淨標題】向量 
    print(" 計算【標題】向量中 (已剔除標籤)...")
    title_vectors = []
    for tokens in title_tokens:
        word_vectors = [model.wv[w] for w in tokens if w in model.wv]
        if word_vectors:
            title_vectors.append(np.mean(word_vectors, axis=0))
        else:
            title_vectors.append(np.zeros(100))
    
    # 5. 存檔
    output = {
        'model': model,
        'doc_vectors': np.array(doc_vectors),       # 給方法一用的內文向量
        'title_vectors': np.array(title_vectors),   # 給方法二用的乾淨標題向量
        'titles': raw_titles,                       # 顯示用的原始標題 (保留 UI 呈現)
        'clean_titles': clean_titles,
        'filepaths': filepaths,
        'dates': dates
    }
    with open("fasttext_model_data.pkl", "wb") as f:
        pickle.dump(output, f)
    print(" FastText 內文與乾淨標題向量已儲存至 fasttext_model_data.pkl")

if __name__ == "__main__":
    train_and_save_fasttext()