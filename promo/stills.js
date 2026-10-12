const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
(async () => {
  const [w, h, ...times] = process.argv.slice(2);
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: +w, height: +h } });
  p.on('console', m => console.log('console:', m.text())); p.on('pageerror', e => console.log('ERR', e.message));
  await p.goto('file://' + __dirname + '/index.html');
  await p.evaluate(() => window.ready);
  for (const t of times) {
    await p.evaluate(t => render(t), +t);
    await p.screenshot({ path: `still_${w}x${h}_${t}.png` });
  }
  await b.close();
})();
