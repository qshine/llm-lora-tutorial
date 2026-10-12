// 切换语言时停留在同一页：中文站在站点根目录，英文站在 en/，两边页面路径一一对应。
// 标题锚点随语言不同，所以只保留页面、不带 #锚点。
document.addEventListener('click', (event) => {
  const link = event.target.closest('a.md-select__link[hreflang]');
  const config = document.getElementById('__config');
  if (!link || !config) return;
  const root = new URL(JSON.parse(config.textContent).base.replace(/\/?$/, '/'), location.href);
  if (!location.href.startsWith(root.href)) return;
  event.preventDefault();
  const page = location.pathname.slice(root.pathname.length);
  location.href = new URL(page, link.href).href;
});
