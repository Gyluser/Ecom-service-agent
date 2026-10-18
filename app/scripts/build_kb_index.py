"""离线构建知识库向量索引。

用法：
  # 默认走 settings.rag_backend
  python app/scripts/build_kb_index.py

  # 显式指定后端，便于一份知识库同时构建两种后端的索引做对比
  python app/scripts/build_kb_index.py --backend numpy
  python app/scripts/build_kb_index.py --backend chroma

流程：
  1. 扫描 knowledge/ 下的所有 .md 文件，按 Markdown 结构和业务语义单元切分。
  2. 调用 OpenAI Embeddings 将每个 chunk 向量化。
  3. 通过 backend.upsert 持久化索引（NumpyBackend 写 JSON / ChromaBackend 写 PersistentClient）。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.config.settings import settings  # noqa: E402
from app.agent.rag.backends import create_backend  # noqa: E402
from app.agent.rag.chunker import chunk_markdown_dir  # noqa: E402
from app.agent.rag.embedder import Embedder  # noqa: E402


def _rooted_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _build_backend(
    name: str,
    *,
    index_path: Path,
    chroma_persist_dir: Path,
    chroma_collection: str,
):
    if name == "numpy":
        return create_backend(
            "numpy",
            index_path=index_path,
        ), str(index_path)
    if name == "chroma":
        return create_backend(
            "chroma",
            persist_dir=chroma_persist_dir,
            collection_name=chroma_collection,
        ), f"{chroma_persist_dir} (collection={chroma_collection})"
    raise ValueError(f"未知后端: {name}")


def main():
    parser = argparse.ArgumentParser(description="构建知识库向量索引")
    parser.add_argument(
        "--backend",
        choices=["numpy", "chroma"],
        default=settings.rag_backend,
        help=f"向量后端（默认: {settings.rag_backend}）",
    )
    parser.add_argument(
        "--kb-dir",
        default=None,
        help="覆盖知识库 Markdown 目录；相对路径以项目根目录为基准",
    )
    parser.add_argument(
        "--index-path",
        default=None,
        help="覆盖 Numpy JSON 索引路径；相对路径以项目根目录为基准",
    )
    parser.add_argument(
        "--chroma-persist-dir",
        default=None,
        help="覆盖 Chroma 持久化目录；相对路径以项目根目录为基准",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=f"覆盖 Chroma collection 名称（默认: {settings.chroma_collection}）",
    )
    args = parser.parse_args()

    kb_dir = _rooted_path(args.kb_dir or settings.kb_dir)
    index_path = _rooted_path(args.index_path or settings.kb_index_path)
    chroma_persist_dir = _rooted_path(
        args.chroma_persist_dir or settings.chroma_persist_dir
    )
    chroma_collection = args.collection or settings.chroma_collection

    if not kb_dir.exists():
        print(f"❌ 知识库目录不存在: {kb_dir}")
        sys.exit(1)

    backend, target_desc = _build_backend(
        args.backend,
        index_path=index_path,
        chroma_persist_dir=chroma_persist_dir,
        chroma_collection=chroma_collection,
    )

    print("=" * 60)
    print("  并夕夕 · 知识库索引构建")
    print(f"  后端      : {args.backend}")
    print(f"  源目录    : {kb_dir}")
    print(f"  索引目标  : {target_desc}")
    print(f"  Embedding : {settings.embedding_model}")
    print(f"  模式/设备 : {settings.embedding_backend} / {settings.embedding_device}")
    print("=" * 60)

    print("\n[1/3] 扫描并切分 markdown 文档...")
    chunks = chunk_markdown_dir(kb_dir)
    if not chunks:
        print("❌ 未发现任何文档，请检查 knowledge/ 目录")
        sys.exit(1)

    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.doc] = by_doc.get(c.doc, 0) + 1
    for doc, n in by_doc.items():
        print(f"   - {doc}: {n} chunk")
    print(f"   合计 {len(chunks)} 个 chunk")

    print(f"\n[2/3] 调用 {settings.embedding_model} 批量向量化...")
    embedder = Embedder(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
        backend=settings.embedding_backend,
        model_path=settings.embedding_model_path,
        device=settings.embedding_device,
    )
    vectors = embedder.encode([c.text for c in chunks])
    dim = len(vectors[0]) if vectors else 0
    print(f"   完成，向量维度 = {dim}")

    print(f"\n[3/3] 写入 {args.backend} 索引...")
    backend.upsert(
        chunks=chunks,
        vectors=vectors,
        embedding_model=settings.embedding_model,
    )
    print(f"   已写入 {backend.size()} 条向量")

    print("\n索引构建完成。")


if __name__ == "__main__":
    main()
