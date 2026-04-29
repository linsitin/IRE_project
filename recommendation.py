import os
import torch
from sentence_transformers import SentenceTransformer, util

def get_all_texts(folder_path):
    """讀取所有文章標題與內容"""
    titles = []
    texts = []
    
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
            
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # 抓取第一行的標題
            title = content.split('\n')[0].replace('標題：', '').strip()
            
            # 抓取純內文
            text_parts = content.split("--------------------------------------------------")
            main_text = text_parts[-1].strip() if len(text_parts) > 1 else content
            
            titles.append(title)
            # 將標題與內文合併作為運算基礎
            texts.append(title + " " + main_text)
            
    return titles, texts

def run_recommendation(target_index=0):
    # 1. 載入 BGE-m3 模型 (目前繁中最強大的開源語義向量模型)
    print("🧠 正在載入 BGE-m3 語義模型 (檔案較大，請稍候)...")
    # 如果電腦記憶體不夠，可將 'BAAI/bge-m3' 換成較輕量的 'paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer('BAAI/bge-m3')
    
    # 2. 讀取資料
    print("📖 讀取文章中...")
    titles, texts = get_all_texts("tfc_reports")
    
    if not titles:
        print("❌ 找不到文章，請確認 tfc_reports 資料夾內有 .txt 檔")
        return
        
    if target_index >= len(titles):
        print("❌ 目標索引超出範圍")
        return

    # 3. 將所有文章轉換為向量 (Embeddings)
    print(f"🔢 正在將 {len(texts)} 篇文章轉換為向量 (此步驟可能會吃 GPU/CPU 資源)...")
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    # 4. 計算相似度
    target_embedding = embeddings[target_index]
    # 計算目標文章與「所有文章」的 Cosine Similarity
    cosine_scores = util.cos_sim(target_embedding, embeddings)[0]
    
    # 5. 找出最高分的前 N 篇 (扣除自己)
    top_k = 3 
    # 使用 torch.topk 抓取分數最高的前 K+1 個 (包含自己)
    top_results = torch.topk(cosine_scores, k=min(top_k + 1, len(titles)))
    
    print("\n" + "="*50)
    print(f"🎯 您正在閱讀：【{titles[target_index]}】")
    print("="*50)
    print("💡 系統為您推薦以下相似查核報告：\n")
    
    for score, idx in zip(top_results[0], top_results[1]):
        # 略過自己這篇文章
        if idx.item() == target_index:
            continue
            
        similarity = score.item() * 100
        print(f"👉 推薦文章：{titles[idx.item()]}")
        print(f"   [相似度分數：{similarity:.1f}%]\n")

if __name__ == "__main__":
    # 你可以修改 target_index 的數字 (0, 1, 2...) 來測試不同文章的推薦結果
    run_recommendation(target_index=0)