(function () {
  const canvas = document.getElementById('backdrop');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let dpr = 1;
  let w = 0;
  let h = 0;

  const GLYPHS = ['·', '+', '×', '○'];
  const SPACING = 72;
  const AMPLITUDE = 11;
  const GLYPH_SIZE = 24;
  const PRESS_RADIUS = 160;
  const PRESS_FORCE = 14;

  function isDark() {
    const attr = document.documentElement.getAttribute('data-theme');
    if (attr === 'dark') return true;
    if (attr === 'light') return false;
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }

  function resize() {
    dpr = Math.max(1, window.devicePixelRatio || 1);
    w = window.innerWidth;
    h = window.innerHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  window.addEventListener('resize', resize);
  resize();

  let mouseX = -9999;
  let mouseY = -9999;
  let lastMouseAt = -Infinity;
  let pressActive = 0;

  window.addEventListener('mousemove', function (e) {
    mouseX = e.clientX;
    mouseY = e.clientY;
    lastMouseAt = performance.now();
  });
  window.addEventListener('mouseleave', function () {
    lastMouseAt = -Infinity;
  });

  function hash(a, b) {
    const s = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453;
    return s - Math.floor(s);
  }

  const start = performance.now();

  function frame(now) {
    const t = (now - start) / 1000;
    ctx.clearRect(0, 0, w, h);
    ctx.font = `${GLYPH_SIZE}px 'JetBrains Mono', ui-monospace, monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const dark = isDark();
    const mutedColor = 'rgba(140,140,140,0.35)';

    // smoothed 0..1 press activity — ramps up while mouse moves recently, decays after idle
    const sinceMove = (now - lastMouseAt) / 1000;
    let target = 0;
    if (sinceMove < 0.1) target = 1;
    else if (sinceMove < 0.6) target = 1 - (sinceMove - 0.1) / 0.5;
    else target = 0;
    pressActive += (target - pressActive) * 0.18;

    const cols = Math.ceil(w / SPACING) + 2;
    const rows = Math.ceil(h / SPACING) + 2;
    const rerollIdx = Math.floor(t / 5);

    for (let j = 0; j < rows; j++) {
      const staggerX = (j % 2) * (SPACING / 2);
      let goldCol = -1;
      if (j % 2 === 1) {
        goldCol = Math.floor(hash(j + 0.1, rerollIdx + 0.7) * cols);
      }
      for (let i = 0; i < cols; i++) {
        const h1 = hash(i * 19.7, j * 31.3);
        const glyph = GLYPHS[Math.floor(h1 * GLYPHS.length)];
        const phase = ((i * 12.9898 + j * 78.233) % (Math.PI * 2));
        let x = i * SPACING + staggerX + Math.sin(t * 0.9 + phase) * AMPLITUDE;
        let y = j * SPACING + Math.cos(t * 1.1 + phase * 1.3) * AMPLITUDE;

        if (pressActive > 0.01) {
          const dx = x - mouseX;
          const dy = y - mouseY;
          const d = Math.sqrt(dx * dx + dy * dy);
          if (d < PRESS_RADIUS) {
            const falloff = (1 - d / PRESS_RADIUS);
            const push = PRESS_FORCE * falloff * falloff * pressActive;
            const nd = d || 1;
            x += (dx / nd) * push;
            y += (dy / nd) * push;
          }
        }

        if (i === goldCol) {
          ctx.save();
          ctx.translate(x, y);
          ctx.rotate(t * 0.22 + phase);
          const grad = ctx.createLinearGradient(-12, -12, 12, 12);
          if (dark) {
            grad.addColorStop(0, '#F4D88B');
            grad.addColorStop(0.55, '#D4AF37');
            grad.addColorStop(1, '#9C7A1E');
          } else {
            grad.addColorStop(0, '#E8C36A');
            grad.addColorStop(0.55, '#B8860B');
            grad.addColorStop(1, '#7A5A0F');
          }
          ctx.fillStyle = grad;
          ctx.fillText(glyph, 0, 0);
          ctx.restore();
        } else {
          ctx.fillStyle = mutedColor;
          ctx.fillText(glyph, x, y);
        }
      }
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
