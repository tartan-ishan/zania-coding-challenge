// ── Dark mode (copied verbatim from reference) ────────
function applyTheme(dark) {
  document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  sessionStorage.setItem('qa_theme', dark ? 'dark' : 'light');
}

const $darkToggle = document.getElementById('dark-toggle');
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const savedTheme  = sessionStorage.getItem('qa_theme');
applyTheme(savedTheme ? savedTheme === 'dark' : prefersDark);

$darkToggle.addEventListener('click', () => {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

  const rect = $darkToggle.getBoundingClientRect();
  const x = Math.round(rect.left + rect.width / 2);
  const y = Math.round(rect.top  + rect.height / 2);
  const r = Math.ceil(Math.hypot(
    Math.max(x, window.innerWidth  - x),
    Math.max(y, window.innerHeight - y)
  ));

  document.documentElement.style.setProperty('--ripple-clip-start', `circle(0px at ${x}px ${y}px)`);
  document.documentElement.style.setProperty('--ripple-clip-end',   `circle(${r}px at ${x}px ${y}px)`);

  if (!document.startViewTransition) {
    applyTheme(!isDark);
    lucide.createIcons();
    return;
  }

  const transition = document.startViewTransition(() => {
    applyTheme(!isDark);
    lucide.createIcons();
  });
  transition.ready.catch(() => {});
});

// ── File upload ───────────────────────────────────────
const docFile   = document.getElementById('doc-file');
const qFile     = document.getElementById('q-file');
const docZone   = document.getElementById('doc-zone');
const qZone     = document.getElementById('q-zone');
const docName   = document.getElementById('doc-name');
const qName     = document.getElementById('q-name');
const submitBtn  = document.getElementById('submit-btn');
const outputBody = document.getElementById('output-body');
const resultCount = document.getElementById('result-count');

function updateZone(zone, nameEl, file) {
  if (file) {
    zone.classList.add('has-file');
    nameEl.textContent = file.name;
  } else {
    zone.classList.remove('has-file');
    nameEl.textContent = '';
  }
  submitBtn.disabled = !(docFile.files[0] && qFile.files[0]);
}

docFile.addEventListener('change', () => updateZone(docZone, docName, docFile.files[0]));
qFile.addEventListener('change',   () => updateZone(qZone,  qName,  qFile.files[0]));

[docZone, qZone].forEach(zone => {
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const input = zone.querySelector('input[type="file"]');
    const dt = new DataTransfer();
    dt.items.add(e.dataTransfer.files[0]);
    input.files = dt.files;
    input.dispatchEvent(new Event('change'));
  });
});

submitBtn.addEventListener('click', async () => {
  submitBtn.classList.add('loading');
  submitBtn.disabled = true;
  resultCount.style.display = 'none';

  outputBody.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon" style="background:var(--primary-light)">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/>
          <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/>
          <line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/>
          <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/>
        </svg>
      </div>
      <p>Analyzing document…</p>
      <span>This may take a moment.</span>
    </div>`;

  const form = new FormData();
  form.append('document_file', docFile.files[0]);
  form.append('questions_file', qFile.files[0]);

  try {
    const res  = await fetch('/api/v1/qa', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) {
      const msg = friendlyError(data.detail, res.status);
      outputBody.innerHTML = `<div class="banner error">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;margin-top:1px"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        ${escHtml(msg)}
      </div>`;
      return;
    }

    const answers = data.answers || {};
    const entries = Object.entries(answers);

    resultCount.textContent = `${entries.length} answer${entries.length !== 1 ? 's' : ''}`;
    resultCount.style.display = 'inline';

    if (entries.length === 0) {
      outputBody.innerHTML = `<div class="banner success">No answers returned.</div>`;
      return;
    }

    outputBody.innerHTML = `<div class="qa-list">${entries.map(([q, a], i) => {
      const answer = a.answer ?? a;
      const isUnavailable = answer === 'Data Not Available';
      const reasoning = a.stepwise_reasoning ?? [];
      const citations = a.citations ?? [];
      const confidence = a.confidence ?? null;

      const confidencePct = confidence !== null ? Math.round(confidence * 100) : null;
      const confidenceHtml = confidencePct !== null ? `
        <span class="confidence-inline">
          <span class="confidence-label">Confidence</span>
          <span class="confidence-bar-wrap"><span class="confidence-bar" style="width:${confidencePct}%"></span></span>
          <span class="confidence-pct">${confidencePct}%</span>
        </span>` : '';

      const reasoningDetails = reasoning.length ? `
        <details class="qa-details">
          <summary>Reasoning</summary>
          <div class="qa-details-body">
            <ol>${reasoning.map(s => `<li>${escHtml(s)}</li>`).join('')}</ol>
          </div>
        </details>` : '';

      const citationsDetails = citations.length ? `
        <details class="qa-details">
          <summary>Citations</summary>
          <div class="qa-details-body">
            <ul>${citations.map(c => `<li>${escHtml(c)}</li>`).join('')}</ul>
          </div>
        </details>` : '';

      const hasFooter = confidencePct !== null || reasoning.length || citations.length;
      const footerHtml = hasFooter ? `
        <div class="qa-footer">
          ${reasoningDetails}
          ${citationsDetails}
          ${confidenceHtml}
        </div>` : '';

      return `
      <div class="qa-item">
        <div class="qa-question">
          <span class="q-badge">${i + 1}</span>
          ${escHtml(q)}
        </div>
        <div class="qa-answer ${isUnavailable ? 'unavailable' : ''}">
          ${isUnavailable ? escHtml(answer) : renderMarkdown(answer)}
        </div>
        ${footerHtml}
      </div>`;
    }).join('')}
    </div>`;

  } catch (err) {
    outputBody.innerHTML = `<div class="banner error">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;margin-top:1px"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      Could not reach the server. Is the backend running?
    </div>`;
  } finally {
    submitBtn.classList.remove('loading');
    submitBtn.disabled = false;
  }
});

// ── Helpers ───────────────────────────────────────────
const ERROR_MESSAGES = {
  UNSUPPORTED_DOCUMENT_TYPE:  'Please upload a PDF or JSON file as your document.',
  UNSUPPORTED_QUESTIONS_TYPE: 'Please upload a JSON file for your questions.',
  DOCUMENT_TOO_LARGE:         'Your document is too large. Please upload a file under 20 MB.',
  QUESTIONS_FILE_TOO_LARGE:   'Your questions file is too large. Please keep it under 100 KB.',
  INVALID_DOCUMENT:           'The document could not be read. Check that it is a valid, non-empty PDF or JSON file.',
  INVALID_QUESTIONS:          'The questions file could not be read. It must be a JSON array of strings.',
  TOO_MANY_QUESTIONS:         'Too many questions submitted. Please reduce the number of questions and try again.',
  RETRIEVER_BUILD_FAILED:     'Something went wrong while processing your document. Please try again.',
  AI_SERVICE_ERROR:           'The AI service is temporarily unavailable. Please wait a moment and try again.',
};

function friendlyError(detail, httpStatus) {
  if (detail && typeof detail === 'object') {
    return ERROR_MESSAGES[detail.error_code] || detail.message || `Unexpected error (${httpStatus}).`;
  }
  // Plain string detail (e.g. FastAPI validation errors before our handler runs)
  if (typeof detail === 'string') return detail;
  return `Something went wrong (${httpStatus}). Please try again.`;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderMarkdown(str) {
  const lines  = String(str).split('\n');
  const output = [];
  let inUl = false, inOl = false;

  const flush = () => {
    if (inUl) { output.push('</ul>'); inUl = false; }
    if (inOl) { output.push('</ol>'); inOl = false; }
  };

  const inline = s => s
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,     '<em>$1</em>')
    .replace(/_(.+?)_/g,       '<em>$1</em>')
    .replace(/`([^`]+)`/g,     '<code>$1</code>');

  for (const line of lines) {
    const ul = line.match(/^[-*] (.+)/);
    const ol = line.match(/^\d+\. (.+)/);
    if (ul) {
      if (inOl) { output.push('</ol>'); inOl = false; }
      if (!inUl) { output.push('<ul>'); inUl = true; }
      output.push(`<li>${inline(ul[1])}</li>`);
    } else if (ol) {
      if (inUl) { output.push('</ul>'); inUl = false; }
      if (!inOl) { output.push('<ol>'); inOl = true; }
      output.push(`<li>${inline(ol[1])}</li>`);
    } else if (/^---+$/.test(line)) {
      flush(); output.push('<hr>');
    } else if (line.trim() === '') {
      flush();
    } else {
      flush(); output.push(`<p>${inline(line)}</p>`);
    }
  }
  flush();
  return output.join('');
}
