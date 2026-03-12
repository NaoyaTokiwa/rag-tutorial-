# RAG チュートリアル（Streamlit + OpenAI）

このリポジトリは、PDF ベースの RAG（Retrieval Augmented Generation）アプリを題材にした
Python / Streamlit 学習用チュートリアルです。

## 機能概要

- PDF アップロード
- PDF 内容のベクトル化（ChromaDB）
- PDF に基づく QA / 要約

## 動かし方（ローカル）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install streamlit openai chromadb pypdf
export OPENAI_API_KEY="sk-..."
streamlit run app.py
