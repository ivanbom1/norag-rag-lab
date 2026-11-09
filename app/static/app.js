const STATUS_TEXT = {
  answered: "Answer",
  unverified: "Warning: some citations could not be found in the retrieved text.",
  not_found: "Not found in the selected documents.",
  invalid_response: "Error: the language model returned a malformed answer.",
  llm_error: "Error: the language model could not be reached.",
};

let lastResult = null;
let openedChunk = null;
let corpus = null;
let graphs = {};

const selectedRetrievers = () => [...document.querySelectorAll("#retrievers input:checked")].map((box) => box.value);
const foundBy = (ranks) => Object.entries(ranks).map(([name, rank]) => `${name} #${rank}`).join(", ");

// "KST-5031 —raised_by→ postroom, then postroom —owned_by→ Messaging": how the graph reached a chunk, from the seed.
const graphPaths = (graph) => (graph ? graph.paths : []).map((path) =>
  el("li", "graph: " + path.map((e) => `${e.source} —${e.relation}→ ${e.target}`).join(", then ")));

const corpusPath = (path) => `/api/corpora/${encodeURIComponent(corpus)}${path}`;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.status === 204 ? null : response;
}

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

// Documents are untrusted, so text is always inserted with textContent, never innerHTML.
function renderHighlighted(container, text, start, end) {
  container.replaceChildren(
    document.createTextNode(text.slice(0, start)),
    el("mark", text.slice(start, end)),
    document.createTextNode(text.slice(end)),
  );
}

function findQuote(text, quote) {
  const words = quote.trim().split(/\s+/).map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const match = new RegExp(words.join("\\s+")).exec(text);
  return match ? [match.index, match.index + match[0].length] : null;
}

async function loadDocuments() {
  const documents = await (await api(corpusPath("/documents"))).json();
  const list = document.getElementById("documents");
  list.replaceChildren(...documents.map((doc) => {
    const item = el("li");
    const checkbox = el("input");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.value = doc.name;
    const remove = el("button", "Delete");
    remove.onclick = () => deleteDocument(doc.name);
    item.append(checkbox, ` ${doc.name} (${Math.round(doc.size / 1024)} KB) `, remove);
    return item;
  }));
}

async function deleteDocument(name) {
  if (!confirm(`Delete ${name}?`)) return;
  const message = document.getElementById("documents-message");
  try {
    await api(corpusPath(`/documents/${encodeURIComponent(name)}`), { method: "DELETE" });
    message.textContent = `${name} deleted.`;
    await loadDocuments();
  } catch (error) {
    message.textContent = error.message;
  }
}

document.getElementById("upload").onchange = async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const message = document.getElementById("documents-message");
  message.textContent = `Uploading and indexing ${file.name}…`;
  const form = new FormData();
  form.append("file", file);
  try {
    const result = await (await api(corpusPath("/documents"), { method: "POST", body: form })).json();
    message.textContent = `${result.name} indexed (${result.chunks} chunks).`;
    await loadDocuments();
  } catch (error) {
    message.textContent = error.message;
  }
  event.target.value = "";
};

function updateGraphRetriever() {
  const label = document.getElementById("graph-retriever");
  const box = label.querySelector("input");
  box.disabled = !graphs[corpus];
  if (box.disabled) box.checked = false;
  label.title = box.disabled ? "No graph for this corpus / chunk settings (build it with python -m app.graph)" : "";
}

async function loadSettings() {
  const settings = await (await api("/api/settings")).json();
  const form = document.getElementById("settings");
  for (const key of ["chunk_size", "overlap_percent", "top_k"]) form.elements[key].value = settings[key];
  for (const box of document.querySelectorAll("#retrievers input")) box.checked = settings.retrievers.includes(box.value);
  graphs = settings.graphs;
  updateGraphRetriever();
}

document.getElementById("settings").onsubmit = async (event) => {
  event.preventDefault();
  const form = event.target;
  const message = document.getElementById("settings-message");
  const button = form.querySelector("button");
  const body = Object.fromEntries(["chunk_size", "overlap_percent", "top_k"].map((k) => [k, Number(form.elements[k].value)]));
  button.disabled = true;
  message.textContent = "Indexing… this can take a few minutes for a new chunk size.";
  try {
    const result = await (await api("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })).json();
    graphs = result.graphs;
    updateGraphRetriever();
    message.textContent = result.reindexed.length
      ? `Re-indexed ${result.reindexed.length} document(s) in ${result.seconds} s.`
      : "Saved. No re-indexing needed.";
  } catch (error) {
    message.textContent = error.message;
  }
  button.disabled = false;
};

document.getElementById("ask").onsubmit = async (event) => {
  event.preventDefault();
  const question = event.target.elements.question.value.trim();
  const documents = [...document.querySelectorAll("#documents input:checked")].map((box) => box.value);
  const button = event.target.querySelector("button");
  button.disabled = true;
  try {
    const response = await api(corpusPath("/ask"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, documents, retrievers: selectedRetrievers() }),
    });
    renderResult(await response.json());
  } catch (error) {
    renderResult({ status: "llm_error", answer: error.message, citations: [], retrieved: [] });
  }
  button.disabled = false;
};

function renderResult(result) {
  lastResult = result;
  document.getElementById("result").hidden = false;
  document.getElementById("reference").hidden = true;
  document.getElementById("document").hidden = true;

  const status = document.getElementById("status");
  status.textContent = STATUS_TEXT[result.status];
  status.className = `status-${result.status}`;
  document.getElementById("answer").textContent = result.answer || "";

  document.getElementById("citations").replaceChildren(...result.citations.map((citation) => {
    const item = el("li");
    const link = el("span", `[${citation.label}]`, "link");
    link.onclick = () => openChunk(citation.label, citation.quote);
    item.append(link, ` “${citation.quote}”`);
    if (citation.found_by) item.append(el("span", ` (found by ${foundBy(citation.found_by)})`, "message"));
    if (!citation.verified) item.append(el("span", " (quote not found in source)", "unverified"));
    return item;
  }));

  document.getElementById("retrieved").replaceChildren(...result.retrieved.map((chunk) => {
    const item = el("li");
    const link = el("span", `[${chunk.label}]`, "link");
    link.onclick = () => openChunk(chunk.label);
    item.append(link, ` ${chunk.source}, chars ${chunk.start}–${chunk.end}, found by ${foundBy(chunk.found_by)}`);
    if (chunk.graph) item.append(el("ul", undefined, "message"));
    if (chunk.graph) item.lastChild.append(...graphPaths(chunk.graph));
    return item;
  }));
}

function openChunk(label, quote) {
  const chunk = lastResult.retrieved.find((c) => c.label === label);
  const panel = document.getElementById("reference");
  panel.hidden = false;
  document.getElementById("document").hidden = true;
  const text = document.getElementById("reference-text");
  if (!chunk) {
    document.getElementById("reference-title").textContent = `[${label}] was not among the retrieved chunks`;
    text.textContent = "";
    document.getElementById("reference-path").replaceChildren();
    document.getElementById("show-in-document").hidden = true;
    return;
  }
  openedChunk = chunk;
  document.getElementById("reference-title").textContent =
    `[${label}] ${chunk.source}, chars ${chunk.start}–${chunk.end}, found by ${foundBy(chunk.found_by)}`;
  document.getElementById("reference-path").replaceChildren(...graphPaths(chunk.graph));
  const span = quote ? findQuote(chunk.text, quote) : null;
  if (span) renderHighlighted(text, chunk.text, span[0], span[1]);
  else text.textContent = chunk.text;
  document.getElementById("show-in-document").hidden = false;
  panel.scrollIntoView({ behavior: "smooth" });
}

document.getElementById("show-in-document").onclick = async () => {
  const chunk = openedChunk;
  const panel = document.getElementById("document");
  const pre = document.getElementById("document-text");
  try {
    const text = await (await api(corpusPath(`/documents/${encodeURIComponent(chunk.source)}/text`))).text();
    document.getElementById("document-title").textContent = chunk.source;
    renderHighlighted(pre, text, chunk.start, chunk.end);
    panel.hidden = false;
    pre.querySelector("mark").scrollIntoView({ block: "center" });
  } catch (error) {
    document.getElementById("document-title").textContent = error.message;
    pre.textContent = "";
    panel.hidden = false;
  }
};

for (const button of document.querySelectorAll("#tabs button")) {
  button.onclick = () => {
    for (const other of document.querySelectorAll("#tabs button")) {
      other.classList.toggle("active", other === button);
      document.getElementById(other.dataset.tab).hidden = other !== button;
    }
  };
}

const KS = [1, 3, 5, 10];

function row(cells, tag = "td") {
  const tr = el("tr");
  tr.append(...cells.map((cell) => el(tag, String(cell))));
  return tr;
}

document.getElementById("evaluate").onsubmit = async (event) => {
  event.preventDefault();
  const button = event.target.querySelector("button");
  const answers = event.target.elements.answers.checked;
  const message = document.getElementById("evaluate-message");
  button.disabled = true;
  message.textContent = answers ? "Running retrieval and answers… this takes a few minutes." : "Running retrieval…";
  try {
    const report = await (await api(corpusPath("/evaluate"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answers, retrievers: selectedRetrievers() }),
    })).json();
    renderEvaluation(report);
    const s = report.settings;
    message.textContent = `${report.corpus}, chunk size ${s.chunk_size}, overlap ${s.overlap_percent}%, ` +
      `retrievers ${s.retrievers.join(" + ")}. Saved ${report.path}`;
  } catch (error) {
    message.textContent = error.message;
  }
  button.disabled = false;
};

function formatCost({ retrieval, answer }) {
  const parts = [];
  if (retrieval) {
    const perRetriever = Object.entries(retrieval.retriever_ms).map(([name, ms]) => `${name} ${ms}`).join(", ");
    parts.push(`retrieval ${retrieval.retrieval_ms} ms (${perRetriever}, fusion ${retrieval.fusion_ms}), ` +
      `${retrieval.llm_calls} LLM calls`);
  }
  if (answer) {
    parts.push(`answer ${answer.llm_calls} LLM calls, ${Math.round(answer.prompt_tokens)} prompt + ` +
      `${Math.round(answer.completion_tokens)} completion tokens, ` +
      `${Math.round(answer.retrieval_ms + answer.llm_ms)} ms (LLM ${Math.round(answer.llm_ms)})`);
  }
  return parts.join("; ");
}

function renderEvaluation(report) {
  document.getElementById("evaluation").hidden = false;
  const header = ["type", "n", ...KS.map((k) => `hit@${k}`), ...KS.map((k) => `recall@${k}`)];
  const groups = [["overall", report.summary.overall], ...Object.entries(report.summary.by_type)];
  document.getElementById("evaluation-summary").replaceChildren(
    row(header, "th"),
    ...groups.map(([kind, stats]) => row([
      kind, stats.questions, ...KS.map((k) => stats[`hit@${k}`].toFixed(2)), ...KS.map((k) => stats[`recall@${k}`].toFixed(2)),
    ])),
  );
  document.getElementById("evaluation-cost").textContent = `Cost per question: ${formatCost(report.summary.cost)}`;

  const withAnswers = report.results.some((result) => result.answer);
  document.getElementById("evaluation-questions").replaceChildren(
    row(["id", "type", "question", "evidence ranks", ...(withAnswers ? ["answer status", "answer"] : [])], "th"),
    ...report.results.map((result) => row([
      result.id,
      result.type,
      result.question,
      result.retrieval ? result.retrieval.evidence_ranks.map((rank) => rank ?? "miss").join(", ") : "unanswerable",
      ...(withAnswers ? [result.answer.status, result.answer.answer ?? ""] : []),
    ])),
  );
}

const TOKEN = /[\p{L}\p{N}]+(?:[-.'’][\p{L}\p{N}]+)*/gu;

function linkCell(text, onclick) {
  const td = el("td");
  const link = el("span", text, "link");
  link.onclick = onclick;
  td.append(link);
  return td;
}

function highlightWord(container, text, word) {
  const nodes = [];
  let last = 0;
  for (const match of text.matchAll(TOKEN)) {
    if (match[0].toLowerCase() !== word) continue;
    nodes.push(document.createTextNode(text.slice(last, match.index)), el("mark", match[0]));
    last = match.index + match[0].length;
  }
  nodes.push(document.createTextNode(text.slice(last)));
  container.replaceChildren(...nodes);
}

async function showWordChunks(word) {
  const data = await (await api(corpusPath(`/words/${encodeURIComponent(word)}`))).json();
  document.getElementById("word-chunks").hidden = false;
  document.getElementById("word-chunk-text").hidden = true;
  const shown = data.chunks.length < data.chunk_count ? ` (top ${data.chunks.length} shown)` : "";
  document.getElementById("word-chunks-title").textContent =
    `“${data.word}”: ${data.count} occurrences in ${data.chunk_count} chunks${shown}`;
  document.getElementById("word-chunks-table").replaceChildren(
    row(["chunk", "source", "chars", "count"], "th"),
    ...data.chunks.map((chunk) => {
      const tr = el("tr");
      tr.append(
        linkCell(chunk.id, () => {
          const pre = document.getElementById("word-chunk-text");
          pre.hidden = false;
          highlightWord(pre, chunk.text, data.word);
          pre.scrollIntoView({ behavior: "smooth" });
        }),
        el("td", chunk.source), el("td", `${chunk.start}–${chunk.end}`), el("td", String(chunk.count)),
      );
      return tr;
    }),
  );
  document.getElementById("word-chunks").scrollIntoView({ behavior: "smooth" });
}

document.getElementById("vocabulary").onsubmit = async (event) => {
  event.preventDefault();
  const form = event.target;
  const params = new URLSearchParams({ contains: form.elements.contains.value, stopwords: form.elements.stopwords.checked });
  const data = await (await api(corpusPath(`/words?${params}`))).json();
  document.getElementById("vocabulary-summary").textContent =
    `${data.distinct_words} distinct words, ${data.tokens} words in total, ${data.chunks} chunks. ` +
    `${data.matching_words} match; showing ${data.words.length}.`;
  document.getElementById("vocabulary-table").replaceChildren(
    row(["word", "count", "chunks"], "th"),
    ...data.words.map((entry) => {
      const tr = el("tr");
      tr.append(linkCell(entry.word, () => showWordChunks(entry.word)), el("td", String(entry.count)), el("td", String(entry.chunks)));
      return tr;
    }),
  );
};

async function loadQuestionSet() {
  const select = document.querySelector("#question-words select");
  const questions = await (await api(corpusPath("/questions"))).json();
  select.replaceChildren(select.options[0], ...questions.map((q) => {
    const option = el("option", `${q.id} [${q.type}] ${q.question}`);
    option.value = q.id;
    option.dataset.question = q.question;
    return option;
  }));
  select.onchange = () => {
    const option = select.selectedOptions[0];
    if (option.value) document.querySelector("#question-words textarea").value = option.dataset.question;
  };
}

document.getElementById("question-words").onsubmit = async (event) => {
  event.preventDefault();
  const form = event.target;
  const question = form.elements.question.value.trim();
  const option = form.elements.question_id.selectedOptions[0];
  const questionId = option.value && option.dataset.question === question ? option.value : null;
  const data = await (await api(corpusPath("/words/question"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, question_id: questionId }),
  })).json();

  document.getElementById("question-words-result").hidden = false;
  const evidence = data.evidence_chunks.map((chunk) => chunk.id);
  document.getElementById("question-words-summary").textContent =
    `${data.words.length} distinct words; absent from the corpus: ${data.absent.join(", ") || "none"}. ` +
    (questionId ? `Chunks holding the evidence: ${evidence.join(", ") || "none at this chunk setting"}.` : "");
  document.getElementById("question-words-table").replaceChildren(
    row(["word", "count", `chunks (of ${data.total_chunks})`, ...(questionId ? ["in evidence chunks"] : [])], "th"),
    ...data.words.map((entry) => {
      const tr = el("tr", undefined, entry.count ? (entry.stopword ? "stopword" : "") : "absent");
      tr.append(
        entry.count ? linkCell(entry.word, () => showWordChunks(entry.word)) : el("td", `${entry.word} (absent)`),
        el("td", String(entry.count)),
        el("td", entry.stopword ? `${entry.chunks} (stopword)` : String(entry.chunks)),
      );
      if (questionId) tr.append(el("td", `${entry.in_evidence_chunks.length} / ${evidence.length}`));
      return tr;
    }),
  );
};

async function loadCorpora() {
  const corpora = await (await api("/api/corpora")).json();
  const select = document.getElementById("corpus");
  select.replaceChildren(...corpora.map((name) => {
    const option = el("option", name);
    option.value = name;
    return option;
  }));
  corpus = corpora[0];
  updateGraphRetriever();
  select.onchange = () => {
    corpus = select.value;
    for (const id of ["result", "reference", "document", "evaluation", "question-words-result", "word-chunks"]) {
      document.getElementById(id).hidden = true;
    }
    document.getElementById("vocabulary-table").replaceChildren();
    loadDocuments();
    loadQuestionSet();
    updateGraphRetriever();
  };
  await loadDocuments();
  await loadQuestionSet();
}

loadCorpora();
loadSettings();
