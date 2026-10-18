"""知识库分块：Markdown 结构 + 业务语义单元 + 长文本递归切分。

当前知识库是结构化的 Markdown，但不同文档的检索单元并不相同：
- FAQ：一个问题及其答案是一个 chunk；
- 会员权益：一个会员等级的权益是一个 chunk；
- 退换货政策：一个规则、材料要求或流程是一个 chunk；
- 配送说明：一个配送/运费规则组是一个 chunk；
- 其他 Markdown：优先按 H3，再按 H2 切分。

所有 chunk 都保留 H1/H2/H3 标题路径，并带有父级 ID、来源和类型，
便于检索、结果解释、父子上下文扩展以及后续人工标注。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_CHUNK_CHARS = 1200
CHUNKING_VERSION = "hierarchical-v2"
LEGACY_CHUNKING_VERSION = "legacy-h2-v1"


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    section: str
    text: str
    parent_id: str = ""
    section_path: list[str] = field(default_factory=list)
    source: str = ""
    chunk_type: str = "section"
    # 旧索引没有这个字段，使用 legacy 值可以识别旧分块结果。
    chunking_version: str = LEGACY_CHUNKING_VERSION

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc": self.doc,
            "section": self.section,
            "text": self.text,
            "parent_id": self.parent_id,
            "section_path": list(self.section_path),
            "source": self.source,
            "chunk_type": self.chunk_type,
            "chunking_version": self.chunking_version,
        }


@dataclass
class _MarkdownNode:
    level: int
    title: str
    lines: list[str] = field(default_factory=list)
    children: list["_MarkdownNode"] = field(default_factory=list)


@dataclass
class _ChunkUnit:
    path: list[str]
    text: str
    chunk_type: str


def chunk_markdown_dir(kb_dir: Path) -> list[Chunk]:
    """扫描目录下所有 .md 文件，逐一切分并汇总。"""
    chunks: list[Chunk] = []
    for md_path in sorted(kb_dir.glob("*.md")):
        chunks.extend(_chunk_one_file(md_path))
    return chunks


def _chunk_one_file(md_path: Path) -> list[Chunk]:
    root = _parse_markdown(md_path.read_text(encoding="utf-8"))
    h1 = next((node for node in root.children if node.level == 1), None)
    doc_title = h1.title if h1 else md_path.stem
    h2_nodes = h1.children if h1 else [node for node in root.children if node.level == 2]

    units: list[_ChunkUnit] = []
    overview = _node_body(h1) if h1 else ""
    if overview:
        units.append(
            _ChunkUnit(
                path=[doc_title, "概览"],
                text=overview,
                chunk_type="overview",
            )
        )

    for h2 in h2_nodes:
        if h2.level != 2:
            continue
        path = [doc_title, h2.title]
        body = _node_body(h2)
        children = [child for child in h2.children if child.level == 3]

        if md_path.stem == "常见问题FAQ":
            units.extend(_faq_units(path, children, body))
        elif md_path.stem == "会员权益":
            units.extend(_membership_units(path, h2, children, body))
        elif md_path.stem == "退换货政策":
            units.extend(_policy_units(path, children, body))
        elif md_path.stem == "配送说明":
            units.extend(_delivery_units(path, h2, body))
        else:
            units.extend(_generic_units(path, children, body))

    return _materialize_units(doc=md_path.stem, source=md_path.as_posix(), units=units)


def _parse_markdown(raw: str) -> _MarkdownNode:
    """解析 Markdown 标题树，保留 H1/H2/H3 层级。"""
    root = _MarkdownNode(level=0, title="")
    stack = [root]

    for line in raw.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack[-1].level >= level:
                stack.pop()
            node = _MarkdownNode(level=level, title=title)
            stack[-1].children.append(node)
            stack.append(node)
        else:
            stack[-1].lines.append(line)
    return root


def _node_body(node: _MarkdownNode | None) -> str:
    return "\n".join(node.lines).strip() if node else ""


def _faq_units(
    path: list[str],
    children: list[_MarkdownNode],
    body: str,
) -> list[_ChunkUnit]:
    """FAQ 按 H3 的问题+答案切分；兼容无 H3 的 Q/A 列表。"""
    if children:
        return [
            _ChunkUnit(
                path=path + [child.title],
                text=_join_title_and_body(child.title, _node_body(child)),
                chunk_type="faq",
            )
            for child in children
            if _node_body(child)
        ]

    pairs = _split_qa_pairs(body)
    if pairs:
        return [
            _ChunkUnit(path=path + [question], text=text, chunk_type="faq")
            for question, text in pairs
        ]
    return [_ChunkUnit(path=path, text=body, chunk_type="section")] if body else []


def _membership_units(
    path: list[str],
    h2: _MarkdownNode,
    children: list[_MarkdownNode],
    body: str,
) -> list[_ChunkUnit]:
    """会员权益按具体等级切分，会员 FAQ 按 Q/A 切分。"""
    if h2.title == "五、常见问题":
        pairs = _split_qa_pairs(body)
        return [
            _ChunkUnit(path=path + [question], text=text, chunk_type="faq")
            for question, text in pairs
        ] or ([_ChunkUnit(path=path, text=body, chunk_type="section")] if body else [])

    if children:
        return [
            _ChunkUnit(
                path=path + [child.title],
                text=_node_body(child),
                chunk_type="member_tier" if "会员" in child.title else "subsection",
            )
            for child in children
            if _node_body(child)
        ]
    return [_ChunkUnit(path=path, text=body, chunk_type="section")] if body else []


def _policy_units(
    path: list[str],
    children: list[_MarkdownNode],
    body: str,
) -> list[_ChunkUnit]:
    """退换货政策按具体规则、材料要求和流程切分。"""
    if children:
        return [
            _ChunkUnit(
                path=path + [child.title],
                text=_node_body(child),
                chunk_type="policy_rule",
            )
            for child in children
            if _node_body(child)
        ]
    return [_ChunkUnit(path=path, text=body, chunk_type="policy_process")] if body else []


def _delivery_units(path: list[str], h2: _MarkdownNode, body: str) -> list[_ChunkUnit]:
    """配送说明按配送主题切分，运费规则单独拆出关键规则组。"""
    if h2.title == "三、运费规则":
        bullets = [line.strip() for line in body.splitlines() if line.strip()]
        ordinary = [line for line in bullets if "包邮门槛" in line]
        large = [line for line in bullets if "大件商品" in line]
        remote_and_member = [
            line
            for line in bullets
            if "偏远地区" in line or "会员特权" in line
        ]
        units: list[_ChunkUnit] = []
        if ordinary:
            units.append(
                _ChunkUnit(
                    path=path + ["普通包邮规则"],
                    text="\n".join(ordinary),
                    chunk_type="delivery_rule",
                )
            )
        if large:
            units.append(
                _ChunkUnit(
                    path=path + ["大件商品运费"],
                    text="\n".join(large),
                    chunk_type="delivery_rule",
                )
            )
        if remote_and_member:
            units.append(
                _ChunkUnit(
                    path=path + ["偏远地区与会员包邮例外"],
                    text="\n".join(remote_and_member),
                    chunk_type="delivery_rule",
                )
            )
        return units

    body = body.strip()
    return [_ChunkUnit(path=path, text=body, chunk_type="delivery_section")] if body else []


def _generic_units(
    path: list[str],
    children: list[_MarkdownNode],
    body: str,
) -> list[_ChunkUnit]:
    if children:
        return [
            _ChunkUnit(path=path + [child.title], text=_node_body(child), chunk_type="subsection")
            for child in children
            if _node_body(child)
        ]
    return [_ChunkUnit(path=path, text=body, chunk_type="section")] if body else []


def _split_qa_pairs(text: str) -> list[tuple[str, str]]:
    """切分形如 '- **Q：...**\\n  A：...' 的问答列表。"""
    lines = text.splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^\s*-\s*\*\*Q[：:]", line)
    ]
    pairs: list[tuple[str, str]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        block = "\n".join(lines[start:end]).strip()
        match = re.match(r"^\s*-\s*\*\*Q[：:]\s*(.+?)\*\*", lines[start])
        question = match.group(1).strip() if match else f"问题{position + 1}"
        if block:
            pairs.append((question, block))
    return pairs


def _join_title_and_body(title: str, body: str) -> str:
    return f"{title}\n{body}".strip() if body else title


def _materialize_units(doc: str, source: str, units: list[_ChunkUnit]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for unit in units:
        text = unit.text.strip()
        if not text:
            continue
        pieces = _split_long(text)
        parent_path = unit.path[:-1] if len(unit.path) > 1 else unit.path
        parent_id = _path_id(doc, parent_path)
        base_id = _path_id(doc, unit.path)

        for part_index, piece in enumerate(pieces):
            chunk_id = base_id
            if len(pieces) > 1:
                chunk_id = f"{base_id}::part-{part_index:02d}"
            path_text = " · ".join(unit.path)
            body = f"【{path_text}】\n{piece}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc=doc,
                    section=unit.path[-1],
                    text=body,
                    parent_id=parent_id,
                    section_path=list(unit.path),
                    source=source,
                    chunk_type=unit.chunk_type if len(pieces) == 1 else "long_text_part",
                    chunking_version=CHUNKING_VERSION,
                )
            )
    return chunks


def _path_id(doc: str, path: list[str]) -> str:
    """生成稳定、可读、可用于向量库 ID 的路径 ID。"""
    # section_path 的第一个元素是 H1 文本，doc 使用文件名；ID 中保留 doc
    # 作为统一前缀，避免把品牌名变化直接变成索引 ID 的变化。
    meaningful = [part.strip() for part in path[1:] if part.strip()]
    return "::".join([doc, *meaningful]) if meaningful else doc


def _split_long(text: str) -> list[str]:
    """按段落→行→中文句子递归细分，并在边界处贪心合并。"""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text.strip()]

    fragments: list[str] = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    for paragraph in paragraphs:
        if len(paragraph) <= MAX_CHUNK_CHARS:
            fragments.append(paragraph)
            continue
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        for line in lines:
            if len(line) <= MAX_CHUNK_CHARS:
                fragments.append(line)
                continue
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[。！？；])", line)
                if sentence.strip()
            ]
            fragments.extend(sentences or [line])

    packed: list[str] = []
    buffer = ""
    for fragment in fragments:
        if len(fragment) > MAX_CHUNK_CHARS:
            if buffer:
                packed.append(buffer.strip())
                buffer = ""
            for start in range(0, len(fragment), MAX_CHUNK_CHARS):
                packed.append(fragment[start : start + MAX_CHUNK_CHARS].strip())
            continue
        candidate = f"{buffer}\n\n{fragment}" if buffer else fragment
        if buffer and len(candidate) > MAX_CHUNK_CHARS:
            packed.append(buffer.strip())
            buffer = fragment
        else:
            buffer = candidate
    if buffer:
        packed.append(buffer.strip())
    return packed
