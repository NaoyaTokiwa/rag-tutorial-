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
import chromadb            # ベクトルデータベース（高速類似検索）
from openai import OpenAI  # OpenAI API（ベクトル化 + 回答生成）
from pypdf import PdfReader # PDF → テキスト変換
```

2. RAG 6ステップ実装
① PDF→テキスト
```python
def pdf_to_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))  # PDF をメモリ上で読み込み
    texts = [page.extract_text() for page in reader.pages]  # 全ページ抽出
    return "\n".join(texts)
```

② テキスト → チャンク（800文字分割）
```python
def split_text(text, chunk_size=800, overlap=200):
    # 長い文章を小分け（検索精度向上 + LLM トークン節約）
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks
```

③ チャンク → ベクトル（OpenAI embeddings）
```python
def embed_texts(texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]  # 各文章 → 1536次元ベクトル
```

④ ChromaDB 保存
```python
collection.add(ids=..., documents=chunks, embeddings=..., metadatas=...)
# 「文章 + ベクトル」を DB に格納 → 高速検索可能に！
```

⑤ 質問 → 類似チャンク検索
```python
def retrieve_relevant_chunks(query, k=4):
    q_emb = embed_texts([query])  # 質問をベクトル化
    results = collection.query(query_embeddings=[q_emb], n_results=4)
    return results["documents"]  # 上位4件の関連文章を返す
```

⑥ LLMで回答生成
```python
def generate_answer(query, context_chunks):
    prompt = f"コンテキスト: {'\n'.join(context_chunks)}\n質問: {query}"
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[...])
    return resp.choices.message.content
```

3. Streamlit UI 構成
```
左サイドバー: PDFアップロード + インデックス作成ボタン
メイン: RAGチャット（st.chat_input + st.chat_message）
セッション状態: コレクション + チャット履歴を保持
```

## 重要ポイント:
•	 st.session_state : ページ更新してもデータ保持
•	 st.spinner : 処理中ローディング表示
•	 disabled=not uploaded_files : 条件分岐ボタン制御

## 🔧 拡張アイデア
```
✅ 複数PDF対応（実装済）
✅ 要約モード追加
✅ ソース引用表示（metadata活用）
✅ FastAPI 分離（本格Web化）
✅ LangChain 導入（RAGフレームワーク）
```

## ⚠️ 注意事項
```
✅ OpenAI API課金必須（$5で十分）
✅ .gitignore で .venv除外
✅ 小容量PDF推奨（コスト節約）
```
## 📚 参考資料
* [Streamlit 公式](https://streamlit.io/)
* [ChromaDB 公式](https://docs.trychroma.com/docs/overview/introduction)

```

