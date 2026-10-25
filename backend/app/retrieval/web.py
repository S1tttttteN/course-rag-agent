"""Optional public-web tools: search snippets and bounded HTML/text extraction."""
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup
import httpx


def public_url(value, resolve=False):
    parsed = urlsplit(value)
    if parsed.scheme not in {'https', 'http'} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Only public HTTP(S) URLs are supported')
    if parsed.port not in {None, 80, 443}:
        raise ValueError('Unsupported URL port')
    hostname = parsed.hostname.lower().rstrip('.')
    if hostname == 'localhost' or hostname.endswith(('.localhost', '.local', '.internal')) or '.' not in hostname:
        raise ValueError('Local network URLs are not allowed')
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError('Private address is not allowed')
    if resolve:
        addresses = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == 'https' else 80), type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
            raise ValueError('Private address is not allowed')
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or '/', parsed.query, ''))


def search_web(query):
    from ddgs import DDGS
    import certifi
    rows = DDGS(timeout=12, verify=certifi.where()).text(query, region='cn-zh', max_results=3, backend='auto')
    result = []
    for row in rows:
        try:
            url = public_url(row['href'])
        except (KeyError, ValueError):
            continue
        text = row.get('body', '').strip()[:900]
        if text:
            result.append({'url': url, 'title': row.get('title', url), 'text': text, 'kind': 'web_snippet'})
    return result


def read_webpage(url):
    current = public_url(url, resolve=True)
    with httpx.Client(timeout=15, follow_redirects=False, trust_env=False,
                      headers={'User-Agent': 'CourseStudyRoom/0.1.0-alpha.1 (public-source-reader)'}) as client:
        for _ in range(5):
            with client.stream('GET', current) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    current = public_url(urljoin(current, response.headers['location']), resolve=True)
                    continue
                response.raise_for_status()
                content_type = response.headers.get('content-type', '').lower()
                if not any(kind in content_type for kind in ('text/html', 'text/plain', 'application/xhtml')):
                    raise ValueError('This tool reads HTML or text; open other formats in a browser')
                body = bytearray()
                for chunk in response.iter_bytes(chunk_size=16384):
                    body.extend(chunk)
                    if len(body) >= 1_000_000:
                        break
                soup = BeautifulSoup(bytes(body), 'html.parser')
                title = soup.title.get_text(' ', strip=True) if soup.title else current
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript']):
                    tag.decompose()
                main = soup.find('main') or soup.find('article') or soup.body or soup
                text = '\n'.join(line.strip() for line in main.get_text('\n').splitlines() if line.strip())[:4000]
                if not text:
                    raise ValueError('No readable text found')
                return {'url': url, 'title': title, 'text': text, 'kind': 'web_page'}
    raise ValueError('Too many redirects')
