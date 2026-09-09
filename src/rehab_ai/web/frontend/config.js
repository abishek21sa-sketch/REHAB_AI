// Vercel hosts the browser lab while Render hosts the Starlette API. The
// override keeps preview and local environments on the same static artifact.
window.__REHAB_API_BASE__ = (window.__REHAB_API_BASE__ || ((location.hostname === 'localhost' || location.hostname === '127.0.0.1') ? '' : 'https://rehab-ai-api.onrender.com')).replace(/\/$/, '');
const _rehabFetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  const url = typeof input === 'string' ? input : input.url;
  if (url.startsWith('/api/') || url === '/health') {
    return _rehabFetch(`${window.__REHAB_API_BASE__}${url}`, init);
  }
  return _rehabFetch(input, init);
};
