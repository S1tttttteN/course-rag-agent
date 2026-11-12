"""解析本地文件并切分正文，由调用方决定文件所属课程和存储路径。"""

from codecs import BOM_UTF16_BE, BOM_UTF16_LE, BOM_UTF32_BE, BOM_UTF32_LE
from pathlib import Path
from zipfile import ZipFile

import docx2txt
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}
MAX_CHARACTERS = 2_000_000


def _read_plain_text(path: Path) -> str:
    """按字节序标记或常用中文编码读取纯文本。

    Args:
        path: 待解析的纯文本文件路径。

    Returns:
        去除字节序标记并统一换行后的正文。

    Raises:
        ValueError: 文件无法按识别出的编码解码。
    """
    with path.open("rb") as file:
        prefix = file.read(4)
    # UTF-32 LE 的标记以 UTF-16 LE 的标记开头，优先识别较长的标记。
    if prefix.startswith((BOM_UTF32_LE, BOM_UTF32_BE)):
        encodings = ("utf-32",)
    elif prefix.startswith((BOM_UTF16_LE, BOM_UTF16_BE)):
        encodings = ("utf-16",)
    else:
        encodings = ("utf-8-sig", "gb18030", "big5")

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文本编码，请另存为 UTF-8 后上传。")


def load_and_split(
    path: Path, source: str, chunk_size=500, chunk_overlap=100
) -> list[Document]:
    """解析本地资料并按文本边界切分正文。

    Args:
        path: 已保存原件的本地路径。
        source: 写入片段元数据的来源名称。
        chunk_size: 目标片段长度，单位为字符。
        chunk_overlap: 相邻片段的重叠字符数。

    Returns:
        带来源、起始位置和可用页码元数据的文本片段列表。

    Raises:
        ValueError: 文件格式、编码、正文或大小不符合解析要求。
    """
    suffix = path.suffix.lower()
    pages = []
    if suffix == ".pdf":
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("PDF 已加密，请先解密后上传。")
        total = 0
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            total += len(text)
            if total > MAX_CHARACTERS:
                raise ValueError("正文超过 200 万字，请拆分文件后上传。")
            if text.strip():
                pages.append(
                    Document(
                        page_content=text,
                        metadata={"source": source, "page": index + 1},
                    )
                )
    else:
        if suffix == ".docx":
            with ZipFile(path) as archive:
                if (
                    sum(entry.file_size for entry in archive.infolist())
                    > 100 * 1024 * 1024
                ):
                    raise ValueError("Word 解压后过大，请拆分文件。")
            text = docx2txt.process(str(path))
        elif suffix in {".html", ".htm"}:
            soup = BeautifulSoup(path.read_bytes(), "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = "\n".join(
                line.strip()
                for line in soup.get_text("\n").splitlines()
                if line.strip()
            )
        elif suffix in {".txt", ".md"}:
            text = _read_plain_text(path)
        else:
            raise ValueError("不支持的文件格式。")
        if len(text) > MAX_CHARACTERS:
            raise ValueError("正文超过 200 万字，请拆分文件后上传。")
        if text.strip():
            pages.append(Document(page_content=text, metadata={"source": source}))
    if not pages:
        raise ValueError("没有可读取的正文。扫描 PDF 需要先进行文字识别。")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    ).split_documents(pages)
    if len(chunks) > 10000:
        raise ValueError("资料片段过多，请拆分文件。")
    return chunks
