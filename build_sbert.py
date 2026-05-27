import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer

def build_sbert_vectors():
    # 這裡我們讀取資料庫是為了拿原始的文章內容
    db_file = "tfc_database_maxdf0.5.pkl" # 請確認你的資料庫檔名
    print(f" 讀取 {db_file} 以獲取文章內容...")
    with open(db_file, 'rb') as f:
        db_data = pickle.load(f)
        
    filepaths = db_data['filepaths']
    
    raw_articles = []
    print(" 正在從原始檔案讀取內文...")
    for path in filepaths:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            # 這裡進行簡單的清理（跟當初 build_database 一樣的邏輯）
            text_parts = content.split("==================================================")
            main_text = text_parts[-1].strip() if len(text_parts) > 1 else content.strip()
            raw_articles.append(main_text)

    print(" 正在載入 SentenceTransformer 模型...")
    # 使用 Text2Vec-base-Chinese (效果非常好)
    model = SentenceTransformer('shibing624/text2vec-base-chinese')
    
    print(f" 正在為 {len(raw_articles)} 篇文章計算深度語意向量...")
    # SBERT 內部會自己處理 Tokenization，直接餵原始字串即可
    sbert_doc_vectors = model.encode(raw_articles, show_progress_bar=True)
    
    print(" 存檔中...")
    output = {
        'sbert_doc_vectors': np.array(sbert_doc_vectors)
    }
    with open("sbert_model_data.pkl", "wb") as f:
        pickle.dump(output, f)
    print(" SBERT 向量已儲存至 sbert_model_data.pkl")

if __name__ == "__main__":
    build_sbert_vectors()