const graphSvg = document.querySelector("#graphSvg");
const workspaceShell = document.querySelector("#workspaceShell");
const controlKicker = document.querySelector("#controlKicker");
const controlTitle = document.querySelector("#controlTitle");
const controlHint = document.querySelector("#controlHint");
const questionInput = document.querySelector("#questionInput");
const hopsInput = document.querySelector("#hopsInput");
const topKInput = document.querySelector("#topKInput");
const modeInput = document.querySelector("#modeInput");
const modelInput = document.querySelector("#modelInput");
const centerInput = document.querySelector("#centerInput");
const llmInput = document.querySelector("#llmInput");
const graphButton = document.querySelector("#graphButton");
const askButton = document.querySelector("#askButton");
const compareButton = document.querySelector("#compareButton");
const compareViewButton = document.querySelector("#compareViewButton");
const ragSearchButton = document.querySelector("#ragSearchButton");
const hybridButton = document.querySelector("#hybridButton");
const graphMeta = document.querySelector("#graphMeta");
const answerMeta = document.querySelector("#answerMeta");
const answerBox = document.querySelector("#answerBox");
const contextBox = document.querySelector("#contextBox");
const compareResults = document.querySelector("#compareResults");
const entitiesList = document.querySelector("#entitiesList");
const topList = document.querySelector("#topList");
const statsStrip = document.querySelector("#statsStrip");
const llmStatus = document.querySelector("#llmStatus");
const vectorStatus = document.querySelector("#vectorStatus");
const promptGrid = document.querySelector("#promptGrid");
const ragEvidence = document.querySelector("#ragEvidence");
const cypherExamples = document.querySelector("#cypherExamples");
const cypherInput = document.querySelector("#cypherInput");
const cypherResults = document.querySelector("#cypherResults");
const runCypherButton = document.querySelector("#runCypherButton");
const copyCypherButton = document.querySelector("#copyCypherButton");
const copyStarterCypher = document.querySelector("#copyStarterCypher");
const graphLessonTitle = document.querySelector("#graphLessonTitle");
const graphLessonText = document.querySelector("#graphLessonText");
const graphLessonGnn = document.querySelector("#graphLessonGnn");
const graphLessonVisual = document.querySelector("#graphLessonVisual");
const graphLessonCaption = document.querySelector("#graphLessonCaption");
const lectureSteps = document.querySelector("#lectureSteps");
const lectureDuration = document.querySelector("#lectureDuration");
const lectureTitle = document.querySelector("#lectureTitle");
const lectureProgress = document.querySelector("#lectureProgress");
const lectureProgressBar = document.querySelector("#lectureProgressBar");
const lecturePoints = document.querySelector("#lecturePoints");
const lectureDemoAction = document.querySelector("#lectureDemoAction");
const lectureSpeakerNotes = document.querySelector("#lectureSpeakerNotes");
const quizQuestion = document.querySelector("#quizQuestion");
const quizAnswer = document.querySelector("#quizAnswer");
const revealQuizButton = document.querySelector("#revealQuizButton");
const applyLectureButton = document.querySelector("#applyLectureButton");
const prevLectureButton = document.querySelector("#prevLectureButton");
const nextLectureButton = document.querySelector("#nextLectureButton");
const zoomOutButton = document.querySelector("#zoomOutButton");
const zoomResetButton = document.querySelector("#zoomResetButton");
const zoomInButton = document.querySelector("#zoomInButton");

const COLORS = {
  character: "#1f6b4b",
  weapon: "#8d3d3a",
  place: "#bd8a2b",
  language: "#3c5f94",
  race: "#6f5aa8",
  entity: "#64736b",
};

const STAT_TOOLTIPS = {
  entities:
    "Nos do tipo Entity no Neo4j. Inclui personagens, racas, armas, lugares, linguas e conceitos do grafo.",
  relationships:
    "Arestas dirigidas no Neo4j: ontologia, coocorrencia, mencoes de texto, falas, capitulos e links preditos.",
  characters:
    "Entidades com label Character. Elas alimentam PageRank, caminhos, vizinhanca k-hop e deteccao de entidades.",
  sources:
    "Obras de origem carregadas no corpus: livros completos e scripts dos filmes.",
  chapters:
    "Capitulos identificados nos livros. Eles estruturam chunks e similaridade entre capitulos.",
  retrievalUnits:
    "Unidades recuperaveis pelo RAG: TextChunk dos livros + DialogueLine dos scripts. Cada uma pode ter embedding.",
  textChunks:
    "Chunks dos livros completos, com cerca de 360 palavras e overlap de 60 palavras.",
  dialogueLines:
    "Falas dos scripts. Cada linha vira uma unidade RAG ligada ao personagem que fala e entidades mencionadas.",
};

const PROMPTS = [
  "Qual a relacao de Frodo com Sauron?",
  "Por que o Anel importa para Frodo?",
  "Como Gandalf se conecta a Aragorn?",
  "Quem sao os conectores entre Frodo e Mordor?",
  "Qual personagem e estruturalmente central na rede?",
  "O que muda quando aumento hops de 1 para 3?",
];

const CONTROL_CONTEXT = {
  overview: {
    kicker: "Demo",
    title: "Escolha a trilha",
    hint: "Abertura da aula: use os atalhos para entrar no modo certo sem carregar controles tecnicos demais.",
  },
  rag: {
    kicker: "RAG",
    title: "Busca textual",
    hint: "Ajuste pergunta, top-k e modelo. Aqui a estrutura do grafo sai de cena para o texto aparecer.",
  },
  graph: {
    kicker: "Graph",
    title: "Laboratorio Cypher",
    hint: "Controle centro e saltos do subgrafo enquanto explora queries estruturais.",
  },
  graphrag: {
    kicker: "GraphRAG",
    title: "Texto guiado por grafo",
    hint: "Combine pergunta, k-hop e top-k para mostrar como o grafo reforca a recuperacao textual.",
  },
  compare: {
    kicker: "Compare",
    title: "Mesma pergunta, tres metodos",
    hint: "Rode os tres modos lado a lado. O painel mostra so os parametros que afetam a comparacao.",
  },
  lecture: {
    kicker: "Lecture",
    title: "Roteiro da aula",
    hint: "Avance pelos passos, revele quizzes e aplique cada etapa no playground.",
  },
};

const state = {
  cypherExamples: [],
  activeCypherExample: null,
  lectureSteps: [],
  lectureIndex: 0,
  hasCompare: false,
  lastAnswerMode: null,
  lastQuestion: "",
  graphViewBox: null,
  dragging: false,
  dragStart: null,
};

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status}: ${text}`);
  }
  return response.json();
}

function activeView() {
  return document.querySelector(".view.active")?.id.replace("view-", "") || "overview";
}

function updateControlContext(name) {
  const copy = CONTROL_CONTEXT[name] || CONTROL_CONTEXT.overview;
  controlKicker.textContent = copy.kicker;
  controlTitle.textContent = copy.title;
  controlHint.textContent = copy.hint;
}

function setAnswerPlaceholder(meta, text) {
  answerMeta.textContent = meta;
  answerBox.innerHTML = `<p class="muted">${escapeHtml(text)}</p>`;
  contextBox.textContent = "";
}

function resetPipelinePlaceholder() {
  document.querySelector("#pipeEntities").textContent = "-";
  document.querySelector("#pipeGraph").textContent = "-";
  document.querySelector("#pipeVectors").textContent = "-";
  document.querySelector("#pipeAnswer").textContent = "-";
}

function renderComparePlaceholder() {
  compareResults.innerHTML = `
    <article class="compare-placeholder">RAG vai mostrar evidencias textuais recuperadas por embedding.</article>
    <article class="compare-placeholder">Graph vai mostrar caminhos, vizinhos e relacoes estruturais.</article>
    <article class="compare-placeholder">GraphRAG vai combinar a estrutura com os chunks recuperados.</article>
  `;
}

function updateViewPlaceholders(name) {
  if (name === "rag" && state.lastAnswerMode !== "rag") {
    setAnswerPlaceholder("RAG aguardando execucao", "Rode a busca para ver resposta, chunks recuperados e contexto textual.");
    ragEvidence.innerHTML = `<article class="evidence-card"><h3>Aguardando busca</h3><p>Os chunks e falas recuperados aparecem aqui depois da execucao vetorial.</p></article>`;
  }
  if (name === "graphrag" && state.lastAnswerMode !== "hybrid") {
    setAnswerPlaceholder("GraphRAG aguardando execucao", "Execute o GraphRAG para preencher entidades, subgrafo, evidencias e resposta.");
    resetPipelinePlaceholder();
  }
  if (name === "compare" && !state.hasCompare) {
    renderComparePlaceholder();
  }
}

function setView(name) {
  workspaceShell.dataset.view = name;
  updateControlContext(name);
  window.scrollTo({ top: 0 });
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === name);
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === `view-${name}`);
  });
  if (name === "rag") modeInput.value = "rag";
  if (name === "graph") modeInput.value = "graph";
  if (name === "graphrag") modeInput.value = "hybrid";
  if (name === "compare") modeInput.value = "hybrid";
  updateViewPlaceholders(name);
}

function setTags(names) {
  entitiesList.innerHTML = "";
  if (!names || names.length === 0) {
    const empty = document.createElement("span");
    empty.textContent = "nenhuma";
    entitiesList.appendChild(empty);
    return;
  }
  for (const name of names) {
    const tag = document.createElement("span");
    tag.textContent = name;
    entitiesList.appendChild(tag);
  }
}

function createStat(label, value, tooltip) {
  const item = document.createElement("span");
  item.className = "stat-item";
  item.tabIndex = 0;
  item.dataset.tooltip = tooltip;
  item.setAttribute("aria-label", `${label}: ${value}. ${tooltip}`);
  const labelNode = document.createElement("strong");
  labelNode.textContent = `${label}: `;
  item.append(labelNode, document.createTextNode(String(value)));
  return item;
}

function renderStats(stats) {
  statsStrip.innerHTML = "";
  statsStrip.append(
    createStat("Entidades", stats.entities || 0, STAT_TOOLTIPS.entities),
    createStat("Relacoes", stats.relationships || 0, STAT_TOOLTIPS.relationships),
    createStat("Personagens", stats.characters || 0, STAT_TOOLTIPS.characters),
    createStat("Fontes", `${stats.books || 0} livros + ${stats.movies || 0} filmes`, STAT_TOOLTIPS.sources),
    createStat("Capitulos", stats.chapters || 0, STAT_TOOLTIPS.chapters),
    createStat("Unidades RAG", stats.retrievalDocuments || 0, STAT_TOOLTIPS.retrievalUnits),
    createStat("Chunks livro", stats.textChunks || 0, STAT_TOOLTIPS.textChunks),
    createStat("Falas script", stats.dialogueLines || 0, STAT_TOOLTIPS.dialogueLines),
  );
  topList.innerHTML = "";
  for (const item of stats.topCharacters || []) {
    const row = document.createElement("li");
    const pagerank = Number(item.pagerank || 0).toFixed(4);
    const degree = Number(item.weightedDegree || 0).toFixed(0);
    row.innerHTML = `<strong>${escapeHtml(item.name)}</strong><span>PR ${pagerank} · grau ${degree}</span>`;
    topList.appendChild(row);
  }
}

function formatModelOption(model) {
  const details = [model.parameterSize, model.quantization].filter(Boolean).join(" · ");
  return details ? `${model.name} · ${details}` : model.name;
}

async function loadModels() {
  modelInput.disabled = true;
  modelInput.innerHTML = `<option value="">Carregando Ollama...</option>`;
  try {
    const payload = await getJson("/api/models");
    const models = payload.models || [];
    modelInput.innerHTML = "";
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = formatModelOption(model);
      if (model.name === payload.defaultModel) option.selected = true;
      modelInput.appendChild(option);
    }
    if (!models.length) {
      const option = document.createElement("option");
      option.value = payload.defaultModel || "";
      option.textContent = payload.defaultModel || "Ollama sem modelos";
      modelInput.appendChild(option);
    }
    modelInput.disabled = !modelInput.value;
    modelInput.title = payload.ok
      ? "Modelos lidos do Ollama local do host."
      : `Ollama indisponivel; usando default. ${payload.error || ""}`;
  } catch (error) {
    modelInput.innerHTML = `<option value="">Ollama indisponivel</option>`;
    modelInput.disabled = true;
    modelInput.title = error.message;
  }
}

async function loadVectorStatus() {
  try {
    const payload = await getJson("/api/vector/status");
    vectorStatus.textContent = payload.ready
      ? `${payload.documents} docs · ${payload.dimensions}d · ${payload.embeddingModel || payload.defaultEmbeddingModel}`
      : `ausente · rode make vectors`;
    vectorStatus.classList.toggle("bad", !payload.ready);
  } catch (error) {
    vectorStatus.textContent = `erro: ${error.message}`;
    vectorStatus.classList.add("bad");
  }
}

function edgeClass(edge) {
  if (edge.type === "ENEMY_OF") return "edge semantic enemy";
  if (edge.type === "CO_OCCURS_WITH") return "edge cooccur";
  if (edge.type === "PREDICTED_LINK") return "edge predicted";
  if (edge.type !== "INTERACTS_WITH") return "edge semantic";
  return "edge";
}

function scaleFactory(nodes, width, height) {
  const xs = nodes.map((node) => node.x || 0);
  const ys = nodes.map((node) => node.y || 0);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const pad = 64;
  const spanX = Math.max(maxX - minX, 0.1);
  const spanY = Math.max(maxY - minY, 0.1);
  return {
    x(value) {
      return pad + ((value - minX) / spanX) * (width - pad * 2);
    },
    y(value) {
      return pad + ((value - minY) / spanY) * (height - pad * 2);
    },
  };
}

function setGraphViewBox(width, height) {
  state.graphViewBox = { x: 0, y: 0, width, height };
  applyGraphViewBox();
}

function applyGraphViewBox() {
  const box = state.graphViewBox;
  if (!box) return;
  graphSvg.setAttribute("viewBox", `${box.x} ${box.y} ${box.width} ${box.height}`);
}

function zoomGraph(factor) {
  const box = state.graphViewBox;
  if (!box) return;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const nextWidth = Math.max(180, Math.min(1800, box.width * factor));
  const nextHeight = Math.max(160, Math.min(1400, box.height * factor));
  state.graphViewBox = {
    x: cx - nextWidth / 2,
    y: cy - nextHeight / 2,
    width: nextWidth,
    height: nextHeight,
  };
  applyGraphViewBox();
}

function renderGraph(graph) {
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  graphSvg.innerHTML = "";
  const width = graphSvg.clientWidth || 760;
  const height = graphSvg.clientHeight || 520;
  setGraphViewBox(width, height);

  if (nodes.length === 0) {
    graphMeta.textContent = "Grafo vazio. Rode make seed.";
    return;
  }

  const scale = scaleFactory(nodes, width, height);
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const labelLimit = nodes.length > 65 ? 32 : nodes.length > 45 ? 42 : nodes.length;
  const labeledNodeIds = new Set(
    [...nodes]
      .sort((left, right) => Number(right.size || 0) - Number(left.size || 0))
      .slice(0, labelLimit)
      .map((node) => node.id),
  );
  const edgeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const nodeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  graphSvg.append(edgeLayer, nodeLayer);

  for (const edge of edges) {
    const source = byId.get(edge.source);
    const target = byId.get(edge.target);
    if (!source || !target) continue;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", scale.x(source.x || 0));
    line.setAttribute("y1", scale.y(source.y || 0));
    line.setAttribute("x2", scale.x(target.x || 0));
    line.setAttribute("y2", scale.y(target.y || 0));
    line.setAttribute("class", edgeClass(edge));
    line.setAttribute("stroke-width", Math.min(5, 0.7 + Math.log2(Number(edge.weight || 1) + 1)));
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    const weight = edge.weight || edge.confidence || 1;
    title.textContent = `${edge.sourceName} -[${edge.type}, peso=${weight}]-> ${edge.targetName}`;
    line.appendChild(title);
    edgeLayer.appendChild(line);
  }

  for (const node of nodes) {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const x = scale.x(node.x || 0);
    const y = scale.y(node.y || 0);
    const color = COLORS[node.group] || COLORS.entity;
    group.setAttribute("transform", `translate(${x}, ${y})`);
    group.setAttribute("class", "node-group");

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("r", Math.max(7, Number(node.size || 10)));
    circle.setAttribute("fill", color);
    circle.setAttribute("class", "node");
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${node.name} · ${node.kind || "Entity"} · PR ${Number(node.pagerank || 0).toFixed(4)}`;
    circle.appendChild(title);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "node-label");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("y", -Math.max(11, Number(node.size || 10)) - 6);
    label.textContent = node.name;

    group.append(circle);
    if (labeledNodeIds.has(node.id)) {
      group.appendChild(label);
    }
    group.addEventListener("click", () => {
      centerInput.value = node.name;
      loadGraph();
    });
    nodeLayer.appendChild(group);
  }

  graphMeta.textContent = `${nodes.length} nos · ${edges.length} arestas`;
}

async function loadGraph() {
  const center = encodeURIComponent(centerInput.value.trim());
  const hops = encodeURIComponent(hopsInput.value);
  graphMeta.textContent = "Carregando subgrafo...";
  try {
    const graph = await getJson(`/api/graph?center=${center}&hops=${hops}&limit=180`);
    renderGraph(graph);
  } catch (error) {
    graphMeta.textContent = `Erro: ${error.message}`;
  }
}

function renderAnswer(text) {
  const escaped = escapeHtml(String(text || ""));
  return escaped.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function payload(modeOverride) {
  return {
    question: questionInput.value,
    hops: Number(hopsInput.value),
    top_k: Number(topKInput.value),
    mode: modeOverride || modeInput.value,
    model: modelInput.value || null,
    use_llm: llmInput.checked,
  };
}

function renderEvidenceCards(documents, target = ragEvidence) {
  target.innerHTML = "";
  if (!documents || !documents.length) {
    target.innerHTML = `<article class="evidence-card"><h3>Nenhuma evidencia textual</h3><p>Ajuste a pergunta ou gere o indice com make vectors.</p></article>`;
    return;
  }
  for (const doc of documents) {
    const card = document.createElement("article");
    card.className = "evidence-card";
    const source = doc.sourceTitle || doc.sourceType || "texto";
    const chapter = doc.chapterTitle ? ` / ${doc.chapterTitle}` : "";
    const speaker = doc.speaker ? ` / ${doc.speaker}` : "";
    const vector = doc.vectorScore == null ? "" : ` · cosine ${Number(doc.vectorScore).toFixed(3)}`;
    const boost = doc.graphBoost ? ` · boost ${Number(doc.graphBoost).toFixed(3)}` : "";
    const mentions = (doc.mentions || []).slice(0, 8).join(", ");
    card.innerHTML = `
      <div class="card-kicker">${escapeHtml(doc.retrievalMethod || "retrieval")} · score ${Number(doc.score || 0).toFixed(3)}${vector}${boost}</div>
      <h3>${escapeHtml(source + chapter + speaker)}</h3>
      <p>${escapeHtml(doc.snippet || doc.text || "").slice(0, 720)}</p>
      <small>${mentions ? `mencoes: ${escapeHtml(mentions)}` : "sem mencoes detectadas"}</small>
    `;
    target.appendChild(card);
  }
}

async function askQuestion(modeOverride = null) {
  const requestedMode = modeOverride || modeInput.value;
  answerBox.textContent = "";
  contextBox.textContent = "";
  llmStatus.textContent = llmInput.checked ? "Ollama local gerando resposta..." : "Recuperando contexto...";
  answerMeta.textContent = "Executando retrieval...";
  askButton.disabled = true;
  try {
    const result = await getJson("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload(requestedMode)),
    });
    state.lastAnswerMode = result.mode || requestedMode;
    state.lastQuestion = questionInput.value;
    setTags(result.entities || []);
    answerBox.innerHTML = renderAnswer(result.answer || "");
    contextBox.textContent = result.context || "";
    answerMeta.textContent = `${result.mode} · ${result.retrieval?.method || "sem texto"} · top-k ${result.topK || topKInput.value}`;
    llmStatus.textContent = `${result.model} · ${result.llmStatus}`;
    if ((result.mode || requestedMode) === "rag" && result.documents) renderEvidenceCards(result.documents, ragEvidence);
    if (result.graph && result.graph.nodes) renderGraph(result.graph);
    updatePipeline(result);
    return result;
  } catch (error) {
    llmStatus.textContent = `Erro: ${error.message}`;
    answerMeta.textContent = "Erro na execucao";
    return null;
  } finally {
    askButton.disabled = false;
  }
}

async function compareQuestion() {
  compareResults.innerHTML = "";
  answerMeta.textContent = "Comparando RAG, Graph e GraphRAG...";
  llmStatus.textContent = llmInput.checked ? "Comparando com Ollama local..." : "Comparando sem LLM local...";
  compareButton.disabled = true;
  compareViewButton.disabled = true;
  try {
    const result = await getJson("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload("hybrid")),
    });
    state.hasCompare = true;
    state.lastQuestion = questionInput.value;
    renderCompare(result.results || {});
    const hybrid = result.results?.hybrid;
    if (hybrid) {
      setTags(hybrid.entities || []);
      llmStatus.textContent = `${hybrid.model} · comparacao ${hybrid.llmStatus}`;
      if (hybrid.graph && hybrid.graph.nodes) renderGraph(hybrid.graph);
      updatePipeline(hybrid);
    }
  } catch (error) {
    llmStatus.textContent = `Erro: ${error.message}`;
  } finally {
    compareButton.disabled = false;
    compareViewButton.disabled = false;
  }
}

function renderCompare(results) {
  const labels = {
    rag: "RAG Vetorial",
    graph: "Graph",
    hybrid: "GraphRAG",
  };
  const limits = {
    rag: "Forte para narrativa textual; fraco para explicar caminhos estruturais.",
    graph: "Forte para conexoes, centralidade e k-hop; fraco para detalhes narrativos.",
    hybrid: "Combina estrutura e texto: melhor para perguntas com entidades e relacoes.",
  };
  const badges = {
    rag: "texto",
    graph: "estrutura",
    hybrid: "hibrido",
  };
  compareResults.innerHTML = "";
  for (const mode of ["rag", "graph", "hybrid"]) {
    const result = results[mode];
    if (!result) continue;
    const card = document.createElement("article");
    card.className = `compare-card mode-${mode}`;
    const docs = result.documents || [];
    const graph = result.graph || {};
    const entities = (result.entities || []).slice(0, 5).join(", ") || "sem entidades";
    const method = result.retrieval?.method || (mode === "graph" ? "subgrafo" : "retrieval");
    const graphMetaText = graph.nodes ? ` · ${graph.nodes.length} nos / ${graph.edges?.length || 0} arestas` : "";
    const evidence = docs
      .slice(0, 3)
      .map((doc) => {
        const source = doc.sourceTitle || doc.sourceType || "texto";
        const chapter = doc.chapterTitle ? ` / ${doc.chapterTitle}` : "";
        return `<li>${escapeHtml(source + chapter)} · ${Number(doc.score || 0).toFixed(2)}</li>`;
      })
      .join("");
    card.innerHTML = `
      <div class="compare-card-head">
        <h3>${labels[mode]}</h3>
        <span class="mode-badge">${badges[mode]}</span>
      </div>
      <div class="compare-meta">${escapeHtml(method)} · entidades: ${escapeHtml(entities)}${escapeHtml(graphMetaText)}</div>
      <div class="compare-answer">${renderAnswer(result.answer || "")}</div>
      <div class="method-note">${limits[mode]}</div>
      ${evidence ? `<ol class="mini-evidence">${evidence}</ol>` : `<p class="muted">Sem evidencias textuais diretas neste modo.</p>`}
    `;
    compareResults.appendChild(card);
  }
}

function updatePipeline(result) {
  document.querySelector("#pipeEntities").textContent = (result.entities || []).join(", ") || "nenhuma";
  const graph = result.graph || {};
  document.querySelector("#pipeGraph").textContent = graph.nodes
    ? `${graph.nodes.length} nos, ${graph.edges?.length || 0} arestas`
    : "modo sem subgrafo";
  document.querySelector("#pipeVectors").textContent = result.documents?.length
    ? `${result.documents.length} evidencias · ${result.retrieval?.method || "retrieval"}`
    : "sem evidencia textual";
  document.querySelector("#pipeAnswer").textContent = result.llmStatus || "pronto";
}

async function loadCypherExamples() {
  const payload = await getJson("/api/cypher/examples");
  state.cypherExamples = payload.examples || [];
  cypherExamples.innerHTML = "";
  for (const example of state.cypherExamples) {
    const button = document.createElement("button");
    button.className = "query-item";
    button.type = "button";
    button.dataset.exampleId = example.id;
    const visual = example.visual || {};
    const visualHint = visual.center ? `${visual.center} · ${visual.hops || 1}-hop` : "visual livre";
    button.innerHTML = `
      <strong>${escapeHtml(example.title)}</strong>
      <span>${escapeHtml(example.explain)}</span>
      <small>${escapeHtml(visualHint)}</small>
    `;
    button.addEventListener("click", () => {
      selectCypherExample(example.id, { loadGraph: true });
    });
    cypherExamples.appendChild(button);
  }
  if (state.cypherExamples[0]) selectCypherExample(state.cypherExamples[0].id, { loadGraph: false });
}

function renderCypherLesson(example) {
  if (!example) {
    graphLessonTitle.textContent = "Query customizada";
    graphLessonText.textContent = "Edite ou rode uma consulta read-only. Se quiser que o grafo acompanhe a tabela, selecione um exemplo didatico.";
    graphLessonGnn.textContent = "Queries livres nao tem mapeamento didatico automatico.";
    graphLessonVisual.textContent = "visualizacao livre";
    graphLessonCaption.textContent = "O grafo permanece no centro e hops atuais.";
    return;
  }
  const visual = example.visual || {};
  graphLessonTitle.textContent = example.title || "Exemplo Cypher";
  graphLessonText.textContent = example.lesson || example.explain || "";
  graphLessonGnn.textContent = example.gnn || "Use a consulta para explicar estrutura e propagacao.";
  graphLessonVisual.textContent = visual.center ? `${visual.center} · ${visual.hops || 1}-hop` : "visualizacao livre";
  graphLessonCaption.textContent = visual.caption || "A tabela e o grafo devem ser lidos juntos.";
}

function selectCypherExample(exampleId, options = {}) {
  const example = state.cypherExamples.find((item) => item.id === exampleId);
  if (!example) return null;
  state.activeCypherExample = example;
  cypherInput.value = example.query;
  document.querySelectorAll(".query-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.exampleId === example.id);
  });
  renderCypherLesson(example);
  const visual = example.visual || {};
  if (visual.center) centerInput.value = visual.center;
  if (visual.hops) hopsInput.value = String(visual.hops);
  if (options.loadGraph && visual.center) loadGraph();
  return example;
}

async function runCypher() {
  cypherResults.innerHTML = `<p class="muted">Rodando consulta read-only...</p>`;
  runCypherButton.disabled = true;
  try {
    const result = await getJson("/api/cypher/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: cypherInput.value, limit: 100 }),
    });
    renderTable(result.columns || [], result.rows || [], cypherResults);
    if (state.activeCypherExample?.visual?.center) {
      await loadGraph();
    }
  } catch (error) {
    cypherResults.innerHTML = `<p class="error-text">${escapeHtml(error.message)}</p>`;
  } finally {
    runCypherButton.disabled = false;
  }
}

function renderTable(columns, rows, target) {
  if (!rows.length) {
    target.innerHTML = `<p class="muted">Consulta sem linhas.</p>`;
    return;
  }
  const head = columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("");
  const body = rows
    .map((row) => {
      const cells = columns
        .map((column) => `<td>${escapeHtml(formatCell(row[column]))}</td>`)
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  target.innerHTML = `<p class="table-note">${rows.length} linhas retornadas. Leia a tabela junto com o grafo ao lado.</p><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function formatCell(value) {
  if (value == null) return "";
  if (Array.isArray(value)) return value.join(" -> ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

async function loadLecture() {
  const payload = await getJson("/api/lecture");
  state.lectureSteps = payload.steps || [];
  renderLecture();
}

function renderLecture() {
  const steps = state.lectureSteps;
  lectureSteps.innerHTML = "";
  steps.forEach((step, index) => {
    const button = document.createElement("button");
    button.className = "lecture-step";
    button.classList.toggle("active", index === state.lectureIndex);
    button.dataset.stepId = step.id;
    button.type = "button";
    button.innerHTML = `<span>${index + 1}</span><strong>${escapeHtml(step.title)}</strong><small>${escapeHtml(step.duration)}</small>`;
    button.addEventListener("click", () => {
      state.lectureIndex = index;
      renderLecture();
    });
    lectureSteps.appendChild(button);
  });
  const step = steps[state.lectureIndex];
  if (!step) return;
  const progress = steps.length ? ((state.lectureIndex + 1) / steps.length) * 100 : 0;
  lectureDuration.textContent = step.duration;
  lectureTitle.textContent = step.title;
  lectureProgress.textContent = `Etapa ${state.lectureIndex + 1} de ${steps.length}`;
  lectureProgressBar.style.width = `${progress}%`;
  lecturePoints.innerHTML = (step.talkingPoints || []).map((point) => `<li>${escapeHtml(point)}</li>`).join("");
  lectureDemoAction.textContent = step.demoAction || "Aplique o passo no playground e discuta o resultado.";
  lectureSpeakerNotes.innerHTML = (step.speakerNotes || [])
    .map((note) => `<li>${escapeHtml(note)}</li>`)
    .join("");
  quizQuestion.textContent = step.quiz?.question || "-";
  quizAnswer.textContent = "";
  quizAnswer.dataset.answer = step.quiz?.answer || "";
}

function applyLectureStep() {
  const step = state.lectureSteps[state.lectureIndex];
  if (!step) return;
  questionInput.value = step.question || questionInput.value;
  if (step.center) centerInput.value = step.center;
  if (step.hops) hopsInput.value = String(step.hops);
  if (step.topK) topKInput.value = String(step.topK);
  if (["rag", "graph", "hybrid"].includes(step.mode)) modeInput.value = step.mode;
  if (step.mode === "compare") setView("compare");
  else if (step.mode === "overview") setView("overview");
  else if (step.mode === "hybrid") setView("graphrag");
  else setView(step.mode);
  if (step.cypherExample) {
    selectCypherExample(step.cypherExample, { loadGraph: true });
  } else if (step.mode === "graph") {
    loadGraph();
  }
}

function renderPrompts() {
  promptGrid.innerHTML = "";
  for (const prompt of PROMPTS) {
    const button = document.createElement("button");
    button.className = "prompt-card";
    button.type = "button";
    button.textContent = prompt;
    button.addEventListener("click", () => {
      questionInput.value = prompt;
      if (prompt.includes("hops")) setView("graph");
      else setView("graphrag");
    });
    promptGrid.appendChild(button);
  }
}

async function copyText(text) {
  await navigator.clipboard.writeText(text);
}

function setupGraphPanZoom() {
  graphSvg.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoomGraph(event.deltaY > 0 ? 1.12 : 0.88);
  });
  graphSvg.addEventListener("pointerdown", (event) => {
    if (!state.graphViewBox) return;
    state.dragging = true;
    state.dragStart = { x: event.clientX, y: event.clientY, box: { ...state.graphViewBox } };
    graphSvg.setPointerCapture(event.pointerId);
  });
  graphSvg.addEventListener("pointermove", (event) => {
    if (!state.dragging || !state.dragStart) return;
    const box = state.dragStart.box;
    const rect = graphSvg.getBoundingClientRect();
    const dx = ((event.clientX - state.dragStart.x) / rect.width) * box.width;
    const dy = ((event.clientY - state.dragStart.y) / rect.height) * box.height;
    state.graphViewBox = { ...box, x: box.x - dx, y: box.y - dy };
    applyGraphViewBox();
  });
  graphSvg.addEventListener("pointerup", () => {
    state.dragging = false;
    state.dragStart = null;
  });
}

async function boot() {
  setTags([]);
  renderPrompts();
  setupGraphPanZoom();
  setView(activeView());
  const modelsPromise = loadModels();
  const vectorPromise = loadVectorStatus();
  try {
    const stats = await getJson("/api/stats");
    renderStats(stats);
  } catch (error) {
    statsStrip.innerHTML = `<span>Neo4j indisponivel</span>`;
  }
  try {
    await loadCypherExamples();
  } catch (error) {
    cypherExamples.innerHTML = `<p class="error-text">${escapeHtml(error.message)}</p>`;
  }
  try {
    await loadLecture();
  } catch (error) {
    lectureTitle.textContent = `Erro: ${error.message}`;
  }
  await modelsPromise;
  await vectorPromise;
  await loadGraph();
}

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});
document.querySelectorAll(".demo-jump").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.demoQuestion || questionInput.value;
    modeInput.value = button.dataset.demoMode || modeInput.value;
    setView(button.dataset.demoView || "overview");
  });
});
questionInput.addEventListener("input", () => {
  state.hasCompare = false;
  state.lastAnswerMode = null;
  updateViewPlaceholders(activeView());
});
graphButton.addEventListener("click", loadGraph);
askButton.addEventListener("click", () => askQuestion());
ragSearchButton.addEventListener("click", () => askQuestion("rag"));
hybridButton.addEventListener("click", () => askQuestion("hybrid"));
compareButton.addEventListener("click", compareQuestion);
compareViewButton.addEventListener("click", compareQuestion);
runCypherButton.addEventListener("click", runCypher);
copyCypherButton.addEventListener("click", () => copyText(cypherInput.value));
copyStarterCypher.addEventListener("click", () => copyText(state.cypherExamples[0]?.query || ""));
cypherInput.addEventListener("input", () => {
  state.activeCypherExample = null;
  document.querySelectorAll(".query-item").forEach((item) => item.classList.remove("active"));
  renderCypherLesson(null);
});
hopsInput.addEventListener("change", loadGraph);
centerInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") loadGraph();
});
zoomOutButton.addEventListener("click", () => zoomGraph(1.18));
zoomInButton.addEventListener("click", () => zoomGraph(0.85));
zoomResetButton.addEventListener("click", () => {
  const width = graphSvg.clientWidth || 760;
  const height = graphSvg.clientHeight || 520;
  setGraphViewBox(width, height);
});
prevLectureButton.addEventListener("click", () => {
  state.lectureIndex = Math.max(0, state.lectureIndex - 1);
  renderLecture();
});
nextLectureButton.addEventListener("click", () => {
  state.lectureIndex = Math.min(state.lectureSteps.length - 1, state.lectureIndex + 1);
  renderLecture();
});
document.querySelectorAll(".lecture-nav-button").forEach((button) => {
  button.addEventListener("click", () => {
    const direction = button.dataset.lectureNav;
    state.lectureIndex += direction === "next" ? 1 : -1;
    state.lectureIndex = Math.max(0, Math.min(state.lectureSteps.length - 1, state.lectureIndex));
    renderLecture();
  });
});
revealQuizButton.addEventListener("click", () => {
  quizAnswer.textContent = quizAnswer.dataset.answer || "";
});
applyLectureButton.addEventListener("click", applyLectureStep);

boot();
