const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const { spawn } = require('child_process');
(async () => {
  const [w, h, fps, f0, f1, out] = process.argv.slice(2).map((v, i) => i < 5 ? +v : v);
  const ff = spawn('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2pipe', '-framerate', fps, '-c:v', 'png', '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '10', '-pix_fmt', 'yuv444p', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: w, height: h } });
  p.on('pageerror', e => console.log('ERR', e.message));
  await p.goto('file://' + __dirname + '/index.html');
  await p.evaluate(() => window.ready);
  for (let f = f0; f < f1; f++) {
    await p.evaluate(t => render(t), f / fps);
    const buf = await p.screenshot({ type: 'png' });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (f % 100 === 0) console.log(out, f);
  }
  ff.stdin.end(); await new Promise(r => ff.on('close', r)); await b.close();
})();
