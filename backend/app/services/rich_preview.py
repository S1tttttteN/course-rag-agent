"""将 HTML 和 DOCX 转成供隔离阅读器使用的静态内容。"""

import base64
from binascii import Error as Base64Error
from pathlib import Path
from urllib.parse import urlsplit
from xml.parsers.expat import ExpatError
from zipfile import BadZipFile

import mammoth
import nh3
from bs4 import BeautifulSoup

from app.core.config import MAX_UPLOAD_BYTES
from app.retrieval.loader import MAX_CHARACTERS, read_html, validate_docx_archive


RASTER_SIGNATURES = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
    "image/webp": (b"RIFF",),
}
BODY_ATTRIBUTES = {"id", "class", "style", "dir", "lang"}


def _link_value(tag: str, attr: str, value: str) -> str | None:
    """只保留片段链接和带主机名的 HTTP(S) 链接。

    Args:
        tag: 属性所属的元素名。
        attr: 由清理器传入的属性名。
        value: 清理前的属性值。
    Returns:
        可保留的属性值，或表示删除该属性的 None。
    """
    if tag != "a" or attr != "href":
        return value
    value = value.strip()
    if value.startswith("#"):
        return value
    try:
        url = urlsplit(value)
        if url.scheme in {"http", "https"} and url.hostname:
            return value
    except ValueError:
        return None
    return None


_CLEANER = nh3.Cleaner(
    tags={
        "a", "abbr", "article", "aside", "b", "blockquote", "br", "caption", "cite",
        "code", "col", "colgroup", "dd", "del", "details", "div", "dl", "dt", "em",
        "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header",
        "hr", "i", "img", "kbd", "li", "main", "mark", "nav", "ol", "p", "pre", "q",
        "s", "samp", "section", "small", "span", "strong", "sub", "summary", "sup",
        "table", "tbody", "td", "tfoot", "th", "thead", "tr", "u", "ul", "var",
    },
    clean_content_tags={
        "script", "style", "noscript", "iframe", "form", "object", "embed", "template",
        "svg", "math", "canvas", "textarea", "button", "select",
    },
    attributes={
        "*": BODY_ATTRIBUTES | {"title", "hidden"},
        "a": {"href", "name"},
        "img": {"data-preview-image", "alt", "width", "height"},
        "col": {"span"},
        "colgroup": {"span"},
        "ol": {"start", "reversed", "type"},
        "li": {"value"},
        "td": {"colspan", "rowspan"},
        "th": {"colspan", "rowspan", "scope"},
        "details": {"open"},
    },
    attribute_filter=_link_value,
    url_schemes={"http", "https"},
)


def _embedded_image(content_type: str, data: bytes) -> str | None:
    """校验栅格格式标记，为阅读器编码内嵌图片。

    Args:
        content_type: 文件或原件中声明的媒体类型。
        data: 已在大小预算内读取的图片字节。
    Returns:
        可在隔离阅读器内恢复成图片的 data URL，或 None。
    """
    signatures = RASTER_SIGNATURES.get(content_type, ())
    if not signatures or not data.startswith(signatures):
        return None
    if content_type == "image/webp" and data[8:12] != b"WEBP":
        return None
    return f"data:{content_type};base64,{base64.b64encode(data).decode('ascii')}"


def _read_html_preview(path: Path) -> tuple[BeautifulSoup, str]:
    """提取 HTML 正文树和文件内样式，并标记可显示的图片。

    Args:
        path: 已保存的 HTML 文件路径。
    Returns:
        移除 head 和 style 后的文档树，以及独立的 CSS 文本。
    """
    soup = read_html(path)
    styles = []
    for style in soup.find_all("style"):
        if style.get("type", "text/css").lower() == "text/css":
            css = style.get_text()
            media = style.get("media", "").strip()
            if not media:
                styles.append(css)
            elif "{" not in media and "}" not in media:
                styles.append(f"@media {media} {{\n{css}\n}}")
        style.decompose()
    if soup.head:
        soup.head.decompose()
    remaining = MAX_UPLOAD_BYTES
    for image in soup.find_all("img"):
        image.attrs.pop("data-preview-image", None)
        header, _, payload = image.attrs.pop("src", "").strip().partition(",")
        if not (header.startswith("data:") and header.endswith(";base64")):
            continue
        content_type = header[5:-7].lower()
        if content_type not in RASTER_SIGNATURES or len(payload) > (remaining + 2) // 3 * 4:
            continue
        try:
            data = base64.b64decode(payload, validate=True)
        except Base64Error:
            continue
        if len(data) <= remaining and (embedded := _embedded_image(content_type, data)):
            image["data-preview-image"] = embedded
            remaining -= len(data)
    return soup, "\n".join(styles)


def _read_docx_preview(path: Path) -> BeautifulSoup:
    """在关闭外部文件访问的情况下转换 Word 正文。

    Args:
        path: 已保存的 DOCX 文件路径。
    Returns:
        Mammoth 生成的文档树，供后续清理。
    Raises:
        FileNotFoundError: 原件已删除。
        ValueError: 压缩包超限或内容无法转换。
    """
    remaining = MAX_UPLOAD_BYTES

    @mammoth.images.img_element
    def convert_image(image):
        """为可用栅格图片分配本次预览的图片预算。

        Args:
            image: Mammoth 提供的图片句柄。
        Returns:
            供阅读器恢复图片的属性字典。
        """
        nonlocal remaining
        if image.content_type not in RASTER_SIGNATURES or not remaining:
            return {}
        with image.open() as file:
            data = file.read(remaining + 1)
        if len(data) > remaining or not (embedded := _embedded_image(image.content_type, data)):
            return {}
        remaining -= len(data)
        return {"data-preview-image": embedded}

    validate_docx_archive(path)
    try:
        with path.open("rb") as file:
            result = mammoth.convert_to_html(
                file, external_file_access=False, include_embedded_style_map=False,
                convert_image=convert_image,
            )
    except FileNotFoundError:
        raise
    except (OSError, KeyError, ValueError, ExpatError, BadZipFile, RuntimeError) as error:
        raise ValueError("无法转换 DOCX，请确认文件可在 Word 中打开。") from error
    return BeautifulSoup(result.value, "html.parser")


def render_rich_preview(path: Path) -> dict[str, str]:
    """生成指定原件的清理后正文和独立样式文本。

    Args:
        path: 已校验格式的 HTML、HTM 或 DOCX 原件路径。
    Returns:
        包含清理后 HTML 和样式文本的字典。
    Raises:
        FileNotFoundError: 原件已删除。
        ValueError: DOCX 无法转换或正文超长。
    """
    if path.suffix.lower() == ".docx":
        try:
            soup = _read_docx_preview(path)
        except BadZipFile as error:
            raise ValueError("无法转换 DOCX，请确认文件可在 Word 中打开。") from error
        css = ""
    else:
        soup, css = _read_html_preview(path)

    body = soup.body if soup.body is not None else soup
    if len(body.get_text()) > MAX_CHARACTERS:
        raise ValueError("正文超过 200 万字，请拆分文件后上传。")
    wrapper = soup.new_tag("div", attrs={key: value for key, value in body.attrs.items() if key in BODY_ATTRIBUTES})
    for child in list(body.contents):
        wrapper.append(child.extract())
    return {"html": _CLEANER.clean(str(wrapper)), "css": css}
