document.addEventListener('DOMContentLoaded', function () {
  renderMathInElement(document.querySelector('.wf-prose'), {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false },
    ],
    throwOnError: false,
  });
});
