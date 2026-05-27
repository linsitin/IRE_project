import pickle
import numpy as np
from gensim.models import FastText
import os

try:
    from ckip_transformers.nlp import CkipWordSegmenter
except ImportError:
    print(" 請確認已安裝 ckip-transformers")
    exit()

def train_and_save_fasttext():
    # 1. 讀取資料庫 (使用你目前的資料庫)
    db_file = "tfc_database_maxdf0.5.pkl" # 請確認檔名正確
    print(f" 讀取 {db_file}...")
    with open(db_file, 'rb') as f:
        db_data = pickle.load(f)
    
    corpus_tokens = db_data['corpus'] 
    titles = db_data['titles']
    filepaths = db_data['filepaths']
    dates = db_data['dates']

    print(f" 讀取到 {len(corpus_tokens)} 篇文章。")

    # 呼叫 CKIP 處理標題
    print(" 載入 CKIP 模型 (為了處理標題斷詞)...")
    ws_driver = CkipWordSegmenter(model="albert-base", device=0)
    
    print(" 正在使用 CKIP 進行標題斷詞...")
    # 因為標題很短，這一步應該會在幾秒到一分鐘內完成
    title_ws_results = ws_driver(titles, batch_size=16)
    
    title_tokens = []
    for word_list in title_ws_results:
        # 過濾空白字元
        filtered = [w for w in word_list if w.strip()]
        title_tokens.append(filtered)
    print(" 標題斷詞完成！")
    # ==========================================

    # 2. 訓練模型
    print(" 訓練 FastText 模型中 (全文)...")
    model = FastText(sentences=corpus_tokens, vector_size=100, window=5, min_count=2, workers=os.cpu_count(), epochs=10)
    
    # 3. 計算【全文】向量
    print(" 計算【全文】向量中...")
    doc_vectors = []
    for tokens in corpus_tokens:
        word_vectors = [model.wv[w] for w in tokens if w in model.wv]
        if word_vectors:
            doc_vectors.append(np.mean(word_vectors, axis=0))
        else:
            doc_vectors.append(np.zeros(100))
            
    # 4. 計算【標題】向量 (使用剛剛 CKIP 斷好的 title_tokens)
    print(" 計算【標題】向量中...")
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
        'doc_vectors': np.array(doc_vectors),
        'title_vectors': np.array(title_vectors), # 將算好的標題向量存進去
        'titles': titles,
        'filepaths': filepaths,
        'dates': dates
    }
    with open("fasttext_model_data.pkl", "wb") as f:
        pickle.dump(output, f)
    print(" 模型、全文向量與標題向量已全部更新並儲存至 fasttext_model_data.pkl")

if __name__ == "__main__":
    train_and_save_fasttext()