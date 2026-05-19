import pickle
import numpy as np
from gensim.models import FastText
import os

def train_and_save_fasttext():
    # 讀取斷詞資料 (這是在 build_database.py 裡面準備好的)
    db_file = "tfc_database_maxdf0.5.pkl"
    with open(db_file, 'rb') as f:
        db_data = pickle.load(f)
    
    corpus_tokens = db_data['corpus'] 
    titles = db_data['titles']
    filepaths = db_data['filepaths']
    dates = db_data['dates']

    print(f" 讀取到 {len(corpus_tokens)} 篇文章。")

    # 1. 訓練模型
    print(" 訓練 FastText 中...")
    model = FastText(sentences=corpus_tokens, vector_size=100, window=5, min_count=2, workers=os.cpu_count(), epochs=10)
    
    # 2. 計算所有文章向量 (將斷詞後的詞向量取平均)
    print(" 計算文章向量中...")
    doc_vectors = []
    for tokens in corpus_tokens:
        word_vectors = [model.wv[w] for w in tokens if w in model.wv]
        if word_vectors:
            doc_vectors.append(np.mean(word_vectors, axis=0))
        else:
            doc_vectors.append(np.zeros(100)) # 若無任何詞在模型中，給零向量
    
    # 3. 存檔
    output = {
        'model': model,
        'doc_vectors': np.array(doc_vectors),
        'titles': titles,
        'filepaths': filepaths,
        'dates': dates
    }
    with open("fasttext_model_data.pkl", "wb") as f:
        pickle.dump(output, f)
    print(" 模型與向量矩陣已儲存至 fasttext_model_data.pkl")

if __name__ == "__main__":
    train_and_save_fasttext()