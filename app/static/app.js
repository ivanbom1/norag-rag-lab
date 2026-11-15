const STATUS_TEXT = {
  answered: "",
  unverified: "Some sources below could not be matched exactly in the documents. Check them before relying on this answer.",
  not_found: "I could not find this in the selected documents. Try rephrasing your question, or check more documents on the left.",
  invalid_response: "Something went wrong while writing the answer. Please try again.",
  llm_error: "The answer service could not be reached. Check your internet connection and your API key, then try again.",
};
const SHOWS_ANSWER = new Set(["answered", "unverified"]);
const STEP_NAMES = {
  dense: "Search by meaning",
  bm25: "Search by keywords",
  graph: "Search the concept graph",
  fusion: "Combine the search results",
  rerank: "Rerank the passages",
  redact: "Remove personal data",
  llm: "Write the answer",
};

let lastResult = null;
let openedChunk = null;
let corpus = null;
let graphs = {};

const rerankOn = () => document.getElementById("rerank").checked;
const selectedRetrievers = () => [...document.querySelectorAll("#retrievers input:checked")].map((box) => box.value);
const foundBy = (ranks) => Object.entries(ranks).map(([name, rank]) => `${name} #${rank}`).join(", ");

// "KST-5031 —raised_by→ postroom, then postroom —owned_by→ Messaging": how the graph reached a chunk, from the seed.
const graphPaths = (graph) => (graph ? graph.paths : []).map((path) =>
  el("li", "graph: " + path.map((e) => `${e.source} —${e.relation}→ ${e.target}`).join(", then ")));

const SVG_NS = "http://www.w3.org/2000/svg";
// Hubs like "dashboards" reach dozens of entities in 2 hops; the text under each chunk still lists every path.
const DIAGRAM_PATHS_PER_CHUNK = 2;
const TYPE_COLORS = {
  ErrorCode: "#f8c4c4", Service: "#c4dcf8", Team: "#c8ecc4", Channel: "#f8e4b0", Person: "#e4d0f4",
  Plan: "#c4ecec", Role: "#f4d4e8", Feature: "#eeeeee", Policy: "#e8e0cc",
};

function svg(tag, attributes, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

// One diagram for every graph path behind the retrieved chunks: seeds in the first column, then hop 1 and hop 2.
function renderGraphDiagram(chunks) {
  const nodes = new Map();
  const edges = new Map();
  const place = (key, name, type, column) => {
    if (!nodes.has(key) || column < nodes.get(key).column) nodes.set(key, { name, type, column });
  };
  for (const chunk of chunks) {
    const paths = chunk.graph ? [...chunk.graph.paths].sort((a, b) => a.length - b.length) : [];
    for (const path of paths.slice(0, DIAGRAM_PATHS_PER_CHUNK)) {
      path.forEach((step, i) => {
        const source = `${step.source_type}:${step.source}`;
        const target = `${step.target_type}:${step.target}`;
        place(source, step.source, step.source_type, step.forward ? i : i + 1);
        place(target, step.target, step.target_type, step.forward ? i + 1 : i);
        edges.set(`${source}|${step.relation}|${target}`, { source, target, relation: step.relation });
      });
    }
  }
  const view = document.getElementById("graph-view");
  view.hidden = nodes.size === 0;
  if (view.hidden) return;

  const [rowHeight, gap, charWidth, maxChars] = [46, 200, 7, 34];
  const label = (name) => (name.length > maxChars ? name.slice(0, maxChars - 1) + "…" : name);
  const columns = [];
  for (const [key, node] of [...nodes].sort((a, b) => a[1].name.localeCompare(b[1].name))) {
    (columns[node.column] ||= []).push(key);
    node.width = label(node.name).length * charWidth + 16;
    node.row = columns[node.column].length - 1;
  }
  let x = 10;
  for (const column of columns) {
    const width = Math.max(...column.map((key) => nodes.get(key).width));
    for (const key of column) Object.assign(nodes.get(key), { x, y: 10 + nodes.get(key).row * rowHeight });
    x += width + gap;
  }

  const height = 20 + Math.max(...columns.map((column) => column.length)) * rowHeight;
  const drawing = svg("svg", { width: x - gap + 20, height, "font-family": "system-ui, sans-serif", "font-size": 12 });
  const marker = svg("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 10, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" });
  marker.append(svg("path", { d: "M0,0 L10,5 L0,10 z", fill: "#888" }));
  const defs = svg("defs", {});
  defs.append(marker);
  drawing.append(defs);
  for (const edge of edges.values()) {
    const [a, b] = [nodes.get(edge.source), nodes.get(edge.target)];
    const [y1, y2] = [a.y + 12, b.y + 12];
    const sameColumn = a.column === b.column;
    const x1 = a.column <= b.column ? a.x + a.width : a.x;
    const x2 = sameColumn ? b.x + b.width : (a.column < b.column ? b.x : b.x + b.width);
    const bend = sameColumn ? Math.max(x1, x2) + 50 : (x1 + x2) / 2;
    drawing.append(svg("path", { d: `M${x1},${y1} C${bend},${y1} ${bend},${y2} ${x2},${y2}`, fill: "none", stroke: "#888", "stroke-width": 1.5, "marker-end": "url(#arrow)" }));
    // the label sits near the right-hand end, where edges leaving the same entity have already fanned out
    const t = sameColumn ? 0.5 : (x2 > x1 ? 0.8 : 0.2);
    const at = (p0, p1, p2, p3) => (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3;
    drawing.append(svg("text", { x: at(x1, bend, bend, x2), y: at(y1, y1, y2, y2) - 5, "text-anchor": "middle", "font-style": "italic", fill: "#444",
      "paint-order": "stroke", stroke: "#fafafa", "stroke-width": 4 }, edge.relation));
  }
  for (const node of nodes.values()) {
    const group = svg("g", {});
    group.append(svg("title", {}, `${node.name} (${node.type})`),
      svg("rect", { x: node.x, y: node.y, width: node.width, height: 24, rx: 5, fill: TYPE_COLORS[node.type], stroke: "#666" }),
      svg("text", { x: node.x + 8, y: node.y + 16, fill: "#222" }, label(node.name)));
    drawing.append(group);
  }
  document.getElementById("graph-diagram").replaceChildren(drawing);
  const types = [...new Set([...nodes.values()].map((node) => node.type))];
  document.getElementById("graph-legend").replaceChildren(...types.map((type) => {
    const swatch = el("span", type, "legend");
    swatch.style.background = TYPE_COLORS[type];
    return swatch;
  }));
}

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
    const name = el("label", ` ${doc.name} `, "inline");
    name.prepend(checkbox);
    item.append(name, el("span", `(${Math.round(doc.size / 1024)} KB) `, "message"), remove);
    return item;
  }));
}

async function deleteDocument(name) {
  if (!confirm(`Delete ${name}? It will be removed from the collection and no longer used in answers.`)) return;
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
  message.textContent = `Adding ${file.name}, this can take a minute…`;
  const form = new FormData();
  form.append("file", file);
  try {
    const result = await (await api(corpusPath("/documents"), { method: "POST", body: form })).json();
    message.textContent = `${result.name} was added and can now be used in answers.`;
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
  document.getElementById("rerank").checked = Boolean(settings.rerank);
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
  const message = document.getElementById("ask-message");
  message.classList.remove("loading");
  if (!question) {
    message.textContent = "Type a question first.";
    return;
  }
  if (!documents.length) {
    message.textContent = "Check at least one document on the left.";
    return;
  }
  button.disabled = true;
  message.textContent = "Searching your documents and writing an answer";
  message.classList.add("loading");
  document.getElementById("result").hidden = true;
  try {
    const response = await api(corpusPath("/ask"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, documents, retrievers: selectedRetrievers(), rerank: rerankOn() }),
    });
    renderResult(await response.json());
  } catch (error) {
    renderResult({ status: "llm_error", answer: error.message, citations: [], retrieved: [] });
  }
  message.textContent = "";
  message.classList.remove("loading");
  button.disabled = false;
};

document.querySelector("#ask textarea").onkeydown = (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    document.getElementById("ask").requestSubmit();
  }
};

function renderResult(result) {
  lastResult = result;
  document.getElementById("result").hidden = false;
  document.getElementById("reference").hidden = true;
  document.getElementById("document").hidden = true;

  const status = document.getElementById("status");
  const showsAnswer = SHOWS_ANSWER.has(result.status);
  status.textContent = STATUS_TEXT[result.status] ?? "";
  status.className = `status-${result.status}`;
  status.hidden = !status.textContent;
  document.getElementById("answer").textContent = showsAnswer ? result.answer || "" : "";
  document.getElementById("error-detail").textContent = showsAnswer ? "" : result.answer || "";
  document.getElementById("sources").hidden = !result.citations.length;
  document.getElementById("trace").hidden = true;
  document.getElementById("details-button").textContent = "Details";
  document.getElementById("how-found").hidden = !result.retrieved.length;

  document.getElementById("citations").replaceChildren(...result.citations.map((citation) => {
    const item = el("li");
    const link = el("span", `[${citation.label}]`, "link");
    link.onclick = () => openChunk(citation.label, citation.quote);
    item.append(link, ` “${citation.quote}”`);
    if (citation.found_by) item.append(el("span", ` (found by ${foundBy(citation.found_by)})`, "message technical"));
    if (!citation.verified) item.append(el("span", " (could not be matched in the document)", "unverified"));
    return item;
  }));

  renderGraphDiagram(result.retrieved);
  document.getElementById("retrieved").replaceChildren(...result.retrieved.map((chunk) => {
    const item = el("li");
    const link = el("span", `[${chunk.label}]`, "link");
    link.onclick = () => openChunk(chunk.label);
    item.append(link, ` ${chunk.source}`, el("span", `, chars ${chunk.start}–${chunk.end}, found by ${foundBy(chunk.found_by)}`, "technical"));
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
  document.getElementById("reference-title").replaceChildren(
    `[${label}] ${chunk.source}`,
    el("span", `, chars ${chunk.start}–${chunk.end}, found by ${foundBy(chunk.found_by)}`, "message technical"),
  );
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
      body: JSON.stringify({ answers, retrievers: selectedRetrievers(), rerank: rerankOn() }),
    })).json();
    renderEvaluation(report);
    const s = report.settings;
    message.textContent = `${report.corpus}, chunk size ${s.chunk_size}, overlap ${s.overlap_percent}%, ` +
      `retrievers ${s.retrievers.join(" + ")}${s.rerank ? ", reranked" : ""}. Saved ${report.path}`;
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

function traceFromCost(result) {
  const cost = result.cost || {};
  const spans = Object.entries(cost.retriever_ms || {}).map(([name, ms]) => ({ name, duration_ms: ms }));
  if (cost.retriever_ms) {
    spans.push({ name: "fusion", duration_ms: cost.fusion_ms, chunk_ids: cost.rerank_candidates ? undefined : result.retrieved.map((chunk) => chunk.id) });
  }
  if (cost.rerank_candidates) {
    spans.push({ name: "rerank", duration_ms: cost.rerank_ms, error: cost.rerank_error, candidates: cost.rerank_candidates,
      chunk_ids: result.retrieved.map((chunk) => chunk.id) });
  }
  if (cost.llm_calls) {
    spans.push({
      name: "llm", duration_ms: cost.llm_ms, model: result.model,
      prompt_tokens: cost.prompt_tokens, completion_tokens: cost.completion_tokens,
    });
  }
  return { trace_id: null, kind: "question", status: result.status, spans };
}

async function loadTrace(result) {
  if (result.trace) return result.trace;
  if (result.trace_id) return (await api(`/api/traces/${encodeURIComponent(result.trace_id)}`)).json();
  return traceFromCost(result);
}

const formatMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);

function spanTokens(span) {
  if (span.prompt_tokens === undefined && span.completion_tokens === undefined) return "";
  return `${span.prompt_tokens ?? 0} in, ${span.completion_tokens ?? 0} out`;
}

function renderTrace(trace) {
  const spans = trace.spans || [];
  const total = trace.duration_ms ?? spans.reduce((sum, span) => sum + (span.duration_ms || 0), 0);
  const longest = Math.max(1, ...spans.map((span) => span.duration_ms || 0));
  const tokens = spans.reduce((sum, span) => sum + (span.prompt_tokens || 0) + (span.completion_tokens || 0), 0);
  const money = spans.reduce((sum, span) => sum + (span.cost || 0), 0);
  const parts = [`${spans.length} steps in ${formatMs(total)}`];
  if (tokens) parts.push(`${tokens} tokens`);
  if (money) parts.push(`$${money.toFixed(5)}`);
  if (trace.trace_id) parts.push(`trace ${trace.trace_id}`);
  document.getElementById("trace-summary").textContent = parts.join(" · ");

  document.getElementById("trace-spans").replaceChildren(
    row(["Step", "Time", "", "Tokens", "Cost", "Passages"], "th"),
    ...spans.map((span) => {
      const tr = el("tr", undefined, span.name === "llm" ? "llm" : "");
      const step = el("td", STEP_NAMES[span.name] || span.name);
      step.append(el("span", ` (${span.name}${span.model ? `, ${span.model}` : ""})`, "message"));
      if (span.error) step.append(el("div", span.error, "unverified"));
      const bar = el("td", undefined, "bar");
      const fill = el("div", undefined, "bar-fill");
      fill.style.width = `${((span.duration_ms || 0) / longest) * 100}%`;
      bar.append(fill);
      const ids = span.chunk_ids || [];
      const passages = el("td", ids.length ? String(ids.length) : "", "number");
      passages.title = ids.join("\n");
      tr.append(
        step,
        el("td", span.duration_ms === undefined ? "" : formatMs(span.duration_ms), "number"),
        bar,
        el("td", spanTokens(span), "number"),
        el("td", span.cost ? `$${span.cost.toFixed(5)}` : "", "number"),
        passages,
      );
      return tr;
    }),
  );
  document.getElementById("trace-raw").textContent = JSON.stringify(trace, null, 2);
}

document.getElementById("details-button").onclick = async () => {
  const panel = document.getElementById("trace");
  const button = document.getElementById("details-button");
  if (!panel.hidden) {
    panel.hidden = true;
    button.textContent = "Details";
    return;
  }
  button.disabled = true;
  try {
    renderTrace(await loadTrace(lastResult));
  } catch (error) {
    document.getElementById("trace-summary").textContent = `The details for this answer could not be loaded: ${error.message}`;
    document.getElementById("trace-spans").replaceChildren();
    document.getElementById("trace-raw").textContent = "";
  }
  panel.hidden = false;
  button.textContent = "Hide details";
  button.disabled = false;
};

function showTab(id) {
  for (const button of document.querySelectorAll("#tabs button")) {
    button.classList.toggle("active", button.dataset.tab === id);
    document.getElementById(button.dataset.tab).hidden = button.dataset.tab !== id;
  }
}

function applyAdvanced(open) {
  document.body.classList.toggle("advanced", open);
  document.getElementById("tabs").hidden = !open;
  if (!open) showTab("tab-ask");
  try { localStorage.setItem("advanced", open ? "1" : ""); } catch {}
}

const advanced = document.getElementById("advanced");
advanced.ontoggle = () => applyAdvanced(advanced.open);
try { advanced.open = localStorage.getItem("advanced") === "1"; } catch {}
applyAdvanced(advanced.open);

loadCorpora();
loadSettings();
