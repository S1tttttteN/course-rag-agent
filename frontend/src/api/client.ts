async function request<T>(
  path: string, init?: RequestInit, read: (response: Response) => Promise<T> = response => response.json(),
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...init,
      headers: { ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...init?.headers },
    });
  } catch {
    throw new Error('无法连接服务，请确认后端已在本机 8000 端口启动，再重试。');
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === 'string' ? body.detail : '操作未完成，请稍后重试。');
  }
  return response.status === 204 ? undefined as T : read(response);
}

export { request };
