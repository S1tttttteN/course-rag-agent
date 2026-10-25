"""File-based loading, independent of course paths and web requests."""
from pathlib import Path
from zipfile import ZipFile

import docx2txt
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}
MAX_CHARACTERS = 2_000_000


def load_and_split(path: Path, source: str, chunk_size=500, chunk_overlap=100) -> list[Document]:
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
                pages.append(Document(page_content=text, metadata={"source": source, "page": index + 1}))
    else:
        if suffix == ".docx":
            with ZipFile(path) as archive:
                if sum(entry.file_size for entry in archive.infolist()) > 100 * 1024 * 1024:
                    raise ValueError("Word 解压后过大，请拆分文件。")
            text = docx2txt.process(str(path))
        elif suffix in {".html", ".htm"}:
            soup = BeautifulSoup(path.read_bytes(), "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        elif suffix in {".txt", ".md"}:
            for encoding in ("utf-8-sig", "gb18030", "big5"):
                try:
                    text = path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("无法识别文本编码，请另存为 UTF-8 后上传。")
        else:
            raise ValueError("不支持的文件格式。")
        if len(text) > MAX_CHARACTERS:
            raise ValueError("正文超过 200 万字，请拆分文件后上传。")
        if text.strip():
            pages.append(Document(page_content=text, metadata={"source": source}))
    if not pages:
        raise ValueError("没有可读取的正文。扫描 PDF 需要先进行文字识别。")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    ).split_documents(pages)
    if len(chunks) > 10000:
        raise ValueError("资料片段过多，请拆分文件。")
    return chunks
