import os
import io
import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

# ====== 設定 ======
EMBED_MODEL = "text-embedding-3-small"   # 軽量埋め込みモデル名（OpenAI）
CHAT_MODEL = "gpt-4o-mini"               # 2026年現在の軽量モデル名に更新

# ====== OpenAI クライアント ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("環境変数 OPENAI_API_KEY が設定されていません。")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ====== ChromaDB（インメモリでOK） ======
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.Client(Settings(is_persistent=False))
if "collection" not in st.session_state:
    st.session_state.collection = st.session_state.chroma_client.create_collection(
        name="docs"
    )

# ====== ユーティリティ ======
def pdf_to_text(file_bytes: bytes) -> str:
    """PDF バイト列 → 文字列"""
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)

def split_text(text: str, chunk_size: int = 800, overlap: int = 200):
    """シンプルなチャンク分割（文字数ベース）"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    """OpenAI で埋め込み取得"""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [d.embedding for d in resp.data]

def build_vector_store_from_pdf(file) -> None:
    """PDF からテキスト抽出 → チャンク → 埋め込み → Chroma に追加"""
    bytes_data = file.read()
    text = pdf_to_text(bytes_data)
    chunks = split_text(text)

    embeddings = embed_texts(chunks)
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    metadatas = [{"source": file.name, "index": i} for i in range(len(chunks))]

    st.session_state.collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
    )

def retrieve_relevant_chunks(query: str, k: int = 4):
    """質問文に近いチャンクを上位 k 件取り出す"""
    q_emb = embed_texts([query])[0]
    results = st.session_state.collection.query(
        query_embeddings=[q_emb],
        n_results=k,
    )
    docs = results["documents"][0]
    return docs

def generate_answer(query: str, context_chunks: list[str]) -> str:
    """RAG プロンプトを組んで LLM で回答生成"""
    context = "\n\n".join(context_chunks)
    system_prompt = (
        "あなたは与えられたコンテキストに基づいて日本語で丁寧に答えるアシスタントです。"
        "コンテキストに書かれていないことは推測しすぎず、その旨を伝えてください。"
    )
    user_content = (
        f"【コンテキスト】\n{context}\n\n"
        f"【質問】\n{query}\n\n"
        "上のコンテキストの内容だけを根拠に回答してください。"
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content

# ====== Streamlit UI ======
st.set_page_config(page_title="シンプル RAG デモ", page_icon="💬")

st.title("📄 シンプル RAG チャット（PDF 対応）")

st.sidebar.header("1. PDF をアップロード")
uploaded_files = st.sidebar.file_uploader(
    "社内資料や仕様書などの PDF をアップロードしてください（複数可）",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.sidebar.button("インデックス作成", disabled=not uploaded_files):
    with st.spinner("PDF からインデックスを作成中..."):
        # 既存コレクションを削除（修正箇所）
        try:
            st.session_state.chroma_client.delete_collection("docs")
        except:
            pass  # 存在しない場合は無視
        
        # 新しくコレクション作成
        st.session_state.collection = st.session_state.chroma_client.create_collection(
            name="docs"
        )
        
        for f in uploaded_files:
            build_vector_store_from_pdf(f)
    st.sidebar.success("インデックス作成完了！ 質問してみてください。")

st.sidebar.markdown("---")
st.sidebar.write("※ アップロードした PDF の内容に基づいて回答します。")

st.markdown("### 💬 質問してみよう（RAG）")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(content)

query = st.chat_input("PDF の内容について質問してください（例: この資料を3行で要約して）")

if query:
    st.session_state.chat_history.append(("user", query))
    with st.chat_message("user"):
        st.write(query)

    if st.session_state.collection.count() == 0:
        with st.chat_message("assistant"):
            msg = "まず左のサイドバーから PDF をアップロードして「インデックス作成」を押してください。"
            st.write(msg)
            st.session_state.chat_history.append(("assistant", msg))
    else:
        with st.chat_message("assistant"):
            with st.spinner("関連する文書を検索し、回答を生成しています..."):
                chunks = retrieve_relevant_chunks(query, k=4)
                answer = generate_answer(query, chunks)
            st.write(answer)
            st.session_state.chat_history.append(("assistant", answer))
