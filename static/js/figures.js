(function () {
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  // Linked highlighting: hovering any [data-ref], or focusing a prose term, lights every element that shares a ref.
  const linked = [...document.querySelectorAll('[data-ref]')], refs = (el) => el.dataset.ref.split(' ');
  let hovered = null, focused = null;
  const paint = () => {
    const keys = (hovered || focused)?.dataset.ref.split(' ') ?? [];
    for (const t of linked) t.classList.toggle('is-lit', refs(t).some((k) => keys.includes(k)));
    for (const svg of document.querySelectorAll('svg')) svg.classList.toggle('has-lit', !!svg.querySelector('.is-lit'));
  };
  for (const el of linked) {
    el.addEventListener('pointerenter', () => { hovered = el; paint(); });
    el.addEventListener('pointerleave', () => { hovered = null; paint(); });
    el.addEventListener('focus', () => { focused = el; paint(); });
    el.addEventListener('blur', () => { focused = null; paint(); });
  }

  // Stepper / scene player. Opens on the full picture, because the first view must show the answer.
  for (const root of document.querySelectorAll('.ve-steps')) {
    const parts = [...root.querySelectorAll('[data-s]')], caps = [...root.querySelectorAll('.ve-cap li')];
    const range = root.querySelector('input[type=range]'), play = root.querySelector('[data-play]');
    const n = caps.length; let i = 0, timer = null;
    if (range) range.max = n - 1;
    const show = (k) => {
      i = Math.max(0, Math.min(n - 1, k));
      for (const el of parts) { const s = el.dataset.s.trim().split(' ').map(Number); el.classList.toggle('on', s[0] <= i + 1); el.classList.toggle('now', s.includes(i + 1)); }
      caps.forEach((c, j) => c.hidden = j !== i);
      root.classList.toggle('is-full', i === n - 1);
      if (range) range.value = i;
    };
    const stop = () => { clearInterval(timer); timer = null; if (play) play.textContent = 'Play'; };
    root.addEventListener('click', (e) => { const b = e.target.closest('[data-go]'); if (b) { stop(); show(i + +b.dataset.go); } });
    range?.addEventListener('input', () => { stop(); show(+range.value); });
    play?.addEventListener('click', () => {
      if (timer) return stop();
      if (i === n - 1) show(0);
      play.textContent = 'Pause';
      timer = setInterval(() => (i < n - 1 ? show(i + 1) : stop()), 3500);
    });
    show(n - 1);
    addEventListener('beforeprint', () => show(n - 1));
  }

})();
