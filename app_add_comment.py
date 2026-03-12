"""
全体の流れ（初心者向けまとめ）
1. PDFアップロード → 2. インデックス作成ボタン → 3. 質問入力
    ↓（内部処理）
PDF→テキスト→チャンク→ベクトル→DB保存（事前準備）
質問→ベクトル→類似検索→LLM回答生成→画面表示
"""

# 標準ライブラリ（Python に最初から入っているもの）
import os      # 環境変数（OpenAI キーなど）を取得するために使う
import io      # メモリ上でファイルを扱うためのツール（PDF 読み込み用）

# 外部ライブラリ（pip install で入れたもの）
import streamlit as st   # Web アプリの画面と操作を作るライブラリ
from openai import OpenAI   # OpenAI（ChatGPT を作っている会社）の API を呼び出す
import chromadb   # ベクトルデータベース（文章を数値化して高速検索するための箱）
from chromadb.config import Settings   # ChromaDB の設定をカスタマイズ
from pypdf import PdfReader   # PDF ファイルから中身のテキストを取り出す

# ====== 設定（アプリ全体で使う定数） ======
EMBED_MODEL = "text-embedding-3-small"   # RAG で使う「文章→数値ベクトル変換」モデル名
CHAT_MODEL = "gpt-4o-mini"               # 回答を生成する ChatGPT モデル名（安くて速い）

# ====== OpenAI クライアント（OpenAI と通信するための「電話帳」） ======
# 環境変数から API キーを取得（ターミナルで export OPENAI_API_KEY="sk-..." と設定したもの）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # キーがないと即座にエラー表示してアプリを止める（セキュリティのため）
    st.error("環境変数 OPENAI_API_KEY が設定されていません。")
    st.stop()

# OpenAI クライアントを作成（これで API を呼び出せるようになる）
client = OpenAI(api_key=OPENAI_API_KEY)

# ====== ChromaDB（文章のベクトルを保存・検索するデータベース） ======
# Streamlit の「セッション状態」を使って、ページ更新してもデータが消えないようにする
if "chroma_client" not in st.session_state:
    # インメモリ（RAM 上）の ChromaDB クライアントを作成（永続化しない＝アプリ終了で消える）
    st.session_state.chroma_client = chromadb.Client(Settings(is_persistent=False))
if "collection" not in st.session_state:
    # 「docs」という名前のコレクション（フォルダのようなもの）を作成
    # コレクションの中に文章のベクトルが保存される
    st.session_state.collection = st.session_state.chroma_client.create_collection(
        name="docs"
    )

# ====== ユーティリティ（便利関数：RAG の各ステップを担当） ======
def pdf_to_text(file_bytes: bytes) -> str:
    """RAGステップ1: PDF ファイル → プレーンテキストに変換"""
    # アップロードされた PDF バイト列をメモリ上で PDFReader に渡す
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        # 各ページのテキストを抽出（空なら無視）
        texts.append(page.extract_text() or "")
    # 全ページのテキストを1つにまとめて返す
    return "\n".join(texts)

def split_text(text: str, chunk_size: int = 800, overlap: int = 200):
    """RAGステップ2: 長い文章を「小さく分割」（チャンク化）"""
    # なぜ分割？→ LLM の入力制限があり、検索精度も上がるため
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size   # 800文字ごとに切る
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap   # 重複200文字で次のチャンク開始（文脈つなぎ）
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    """RAGステップ3: 文章のリスト → 数値ベクトルに変換（OpenAI API 呼び出し）"""
    # OpenAI の埋め込み API を呼び出して、各文章を1536次元のベクトルに変換
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    # レスポンスからベクトルリストを取り出して返す
    return [d.embedding for d in resp.data]

def build_vector_store_from_pdf(file) -> None:
    """RAGステップ1〜4: PDF1ファイル分を「テキスト化→分割→ベクトル化→DB保存」の一連処理"""
    bytes_data = file.read()   # アップロードファイルの内容を読み込み
    text = pdf_to_text(bytes_data)   # PDF → テキスト
    chunks = split_text(text)   # テキスト → チャンクリスト

    embeddings = embed_texts(chunks)   # チャンク → ベクトルリスト
    ids = [f"chunk-{i}" for i in range(len(chunks))]   # 各チャンクにユニークID付与
    metadatas = [{"source": file.name, "index": i} for i in range(len(chunks))]   # 元ファイル名やページ番号を記録

    # ChromaDB に「文章・ベクトル・メタデータ」を一括追加（これで検索可能に！）
    st.session_state.collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
    )

def retrieve_relevant_chunks(query: str, k: int = 4):
    """RAGステップ5: 質問文から「関連する文章チャンク」を上位4件検索・抽出"""
    q_emb = embed_texts([query])[0]   # 質問文をベクトル化
    # ChromaDB で類似ベクトル検索（コサイン類似度で近い上位 k 件）
    results = st.session_state.collection.query(
        query_embeddings=[q_emb],
        n_results=k,
    )
    docs = results["documents"][0]   # ヒットした文章チャンクを返す
    return docs

def generate_answer(query: str, context_chunks: list[str]) -> str:
    """RAGステップ6: 検索したチャンク＋質問を LLM に渡して最終回答生成"""
    context = "\n\n".join(context_chunks)   # 検索ヒット文章を1つのコンテキストにまとめる
    system_prompt = (
        "あなたは与えられたコンテキストに基づいて日本語で丁寧に答えるアシスタントです。"
        "コンテキストに書かれていないことは推測しすぎず、その旨を伝えてください。"
    )
    user_content = (
        f"【コンテキスト】\n{context}\n\n"
        f"【質問】\n{query}\n\n"
        "上のコンテキストの内容だけを根拠に回答してください。"
    )

    # OpenAI Chat API を呼び出して回答生成（システムプロンプト＋ユーザ質問）
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content   # 生成された回答を返す

# ====== Streamlit UI（Web 画面部分） ======
st.set_page_config(page_title="シンプル RAG デモ", page_icon="💬")   # タブのタイトルとアイコン設定

st.title("📄 シンプル RAG チャット（PDF 対応）")   # ページタイトル

st.sidebar.header("1. PDF をアップロード")   # 左サイドバーにヘッダー
uploaded_files = st.sidebar.file_uploader(
    "社内資料や仕様書などの PDF をアップロードしてください（複数可）",
    type=["pdf"],   # PDF ファイルのみ許可
    accept_multiple_files=True,   # 複数ファイルOK
)

# 「インデックス作成」ボタン（PDF がアップロードされている時のみ有効）
if st.sidebar.button("インデックス作成", disabled=not uploaded_files):
    with st.spinner("PDF からインデックスを作成中..."):   # 処理中表示
        # 既存コレクションを削除（修正箇所：同じ名前があっても上書き可能に）
        try:
            st.session_state.chroma_client.delete_collection("docs")
        except:
            pass  # 存在しない場合は無視
        
        # 新しくコレクション作成
        st.session_state.collection = st.session_state.chroma_client.create_collection(
            name="docs"
        )
        
        # アップロードされた全 PDF に対して処理実行
        for f in uploaded_files:
            build_vector_store_from_pdf(f)
    st.sidebar.success("インデックス作成完了！ 質問してみてください。")   # 成功メッセージ

st.sidebar.markdown("---")   # 区切り線
st.sidebar.write("※ アップロードした PDF の内容に基づいて回答します。")   # 説明

st.markdown("### 💬 質問してみよう（RAG）")   # メインエリアの見出し

# チャット履歴をセッション状態で保持（ページ更新しても会話が残る）
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 過去の会話を表示（ユーザーと AI の吹き出し形式）
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(content)

# 質問入力欄（Enter または送信ボタンで発火）
query = st.chat_input("PDF の内容について質問してください（例: この資料を3行で要約して）")

if query:
    # ユーザー質問を履歴に追加
    st.session_state.chat_history.append(("user", query))
    with st.chat_message("user"):
        st.write(query)

    # コレクションが空ならエラーメッセージ
    if st.session_state.collection.count() == 0:
        with st.chat_message("assistant"):
            msg = "まず左のサイドバーから PDF をアップロードして「インデックス作成」を押してください。"
            st.write(msg)
            st.session_state.chat_history.append(("assistant", msg))
    else:
        # RAG 本番実行！
        with st.chat_message("assistant"):
            with st.spinner("関連する文書を検索し、回答を生成しています..."):
                # ステップ5: 検索
                chunks = retrieve_relevant_chunks(query, k=4)
                # ステップ6: 生成
                answer = generate_answer(query, chunks)
            st.write(answer)
            st.session_state.chat_history.append(("assistant", answer))
