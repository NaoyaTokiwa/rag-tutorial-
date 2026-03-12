# 🎓 RAG 入門チュートリアル：Streamlit + OpenAI + ChromaDB
PDF 資料をアップロードして**要約・質問回答**ができる RAG（Retrieval Augmented Generation） アプリをゼロから作るチュートリアルです。

↓　起動時のUI
<img width="1428" height="776" alt="image" src="https://github.com/user-attachments/assets/69877b86-851c-48e6-8491-a59efae765c0" />

↓領収書のPDFをアップロードし、要約指示した際の出力結果
<img width="1427" height="758" alt="image" src="https://github.com/user-attachments/assets/10cdb729-a6cf-45b0-a1e0-a3276768d216" />


## 機能概要

- PDF アップロード
- PDF 内容のベクトル化（ChromaDB）
- PDF に基づく QA / 要約

## 🚀 5分で動かす

```bash
git clone https://github.com/NaoyaTokiwa/rag-tutorial-.git
cd rag-tutorial-
python3 -m venv .venv
source .venv/bin/activate  
pip install streamlit openai chromadb pypdf
export OPENAI_API_KEY="sk-..."  # OpenAI Platform で取得
streamlit run app.py
```

## 📱 デモ機能
1.	PDF アップロード（複数可）
2.	自動インデックス作成（テキスト抽出→チャンク分割→ベクトル化）
3.	RAG チャット（PDF内容に基づく要約・QA）

```
例質問:
「この資料を3行で要約して」
「仕様書の○○についてどこに書いてある？」
```

## 🏗️ アプリ構成（app.py 完全解説）
全体アーキテクチャ図

```
[PDF アップロード] 
  ↓
[1.テキスト抽出(PDF→TXT)] → [2.チャンク分割(800文字)] → [3.ベクトル化(OpenAI)]
                                                           ↓
[ChromaDB 保存・検索] ←←← [4.類似検索] ← [質問ベクトル化]
                                                           ↓  
[5.LLM回答生成(gpt-4o-mini)] ← [検索ヒット文章]
  ↓
[日本語回答表示]

```

## コード詳細（初心者向け）
1. 必要なライブラリ

```python
import streamlit as st     # Web UI（チャット画面）
from openai import OpenAI  # OpenAI API（ベクトル+回答）
import chromadb           # ベクトルDB（高速検索）
from pypdf import PdfReader # PDF→テキスト
```

2. RAG 6ステップ実装
① PDF→テキスト
```python
def pdf_to_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([page.extract_text() for page in reader.pages])
```

② テキスト→チャンク
```python
def split_text(text, chunk_size=800, overlap=200):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]

```

③ チャンク→ベクトル
```python
def embed_texts(texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]  # 文章→1536次元ベクトル
```

④ DB保存
```python
collection.add(ids=..., documents=chunks, embeddings=...)
```

⑤ 質問→類似検索
```python
def retrieve_relevant_chunks(query, k=4):
    q_emb = embed_texts([query])
    return collection.query(query_embeddings=[q_emb], n_results=4)["documents"]
```

⑥ LLM回答生成
```python
def generate_answer(query, chunks):
    context = "\n\n".join(chunks)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"コンテキスト:\n{context}\n質問:{query}"}]
    )
    return resp.choices.message.content
```

3. Streamlit UI
```python
サイドバー: PDFアップロード + インデックスボタン
メイン: st.chat_input + st.chat_message（チャットUI）
st.session_state: DB/履歴保持
```

## 🔧 カスタマイズ例
```
・複数PDF: 実装済み
・要約モード: prompt変更
・ソース引用: metadata活用
・LangChain: フレームワーク化
```

## ⚠️ 注意事項
```
✅ OpenAI API課金必須（$5で十分）
✅ .gitignore で .venv除外
✅ 小容量PDF推奨（コスト節約）
```
## 📚 参考資料
•	Streamlit
•	ChromaDB
•	OpenAI API
```

