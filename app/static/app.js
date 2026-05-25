const graphSvg = document.querySelector("#graphSvg");
const workspaceShell = document.querySelector("#workspaceShell");
const controlKicker = document.querySelector("#controlKicker");
const controlTitle = document.querySelector("#controlTitle");
const controlHint = document.querySelector("#controlHint");
const questionInput = document.querySelector("#questionInput");
const hopsInput = document.querySelector("#hopsInput");
const topKInput = document.querySelector("#topKInput");
const modeInput = document.querySelector("#modeInput");
const graphRagStrategyInput = document.querySelector("#graphRagStrategyInput");
const modelInput = document.querySelector("#modelInput");
const centerInput = document.querySelector("#centerInput");
const llmInput = document.querySelector("#llmInput");
const graphButton = document.querySelector("#graphButton");
const askButton = document.querySelector("#askButton");
const compareButton = document.querySelector("#compareButton");
const compareViewButton = document.querySelector("#compareViewButton");
const ragSearchButton = document.querySelector("#ragSearchButton");
const hybridButton = document.querySelector("#hybridButton");
const strategyCompareButton = document.querySelector("#strategyCompareButton");
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
const traceGroundingMeta = document.querySelector("#traceGroundingMeta");
const traceEntities = document.querySelector("#traceEntities");
const traceGraphMeta = document.querySelector("#traceGraphMeta");
const traceGraphEvidence = document.querySelector("#traceGraphEvidence");
const traceRetrievalMeta = document.querySelector("#traceRetrievalMeta");
const traceDocuments = document.querySelector("#traceDocuments");
const tracePromptMeta = document.querySelector("#tracePromptMeta");
const tracePromptSections = document.querySelector("#tracePromptSections");
const tracePromptPreview = document.querySelector("#tracePromptPreview");
const graphRagStrategyCards = document.querySelector("#graphRagStrategyCards");
const activeStrategyKicker = document.querySelector("#activeStrategyKicker");
const activeStrategyName = document.querySelector("#activeStrategyName");
const activeStrategyDescription = document.querySelector("#activeStrategyDescription");
const activeStrategyStageMap = document.querySelector("#activeStrategyStageMap");
const activeStrategyBestFor = document.querySelector("#activeStrategyBestFor");
const activeStrategyGraphRole = document.querySelector("#activeStrategyGraphRole");
const activeStrategyTextRole = document.querySelector("#activeStrategyTextRole");
const activeStrategyLectureCue = document.querySelector("#activeStrategyLectureCue");
const activeStrategyVisualHint = document.querySelector("#activeStrategyVisualHint");
const activeStrategyRisk = document.querySelector("#activeStrategyRisk");
const activeStrategyReferences = document.querySelector("#activeStrategyReferences");
const strategyCompareResults = document.querySelector("#strategyCompareResults");
const traceGraphTitle = document.querySelector("#traceGraphTitle");
const traceRetrievalTitle = document.querySelector("#traceRetrievalTitle");
const tracePromptTitle = document.querySelector("#tracePromptTitle");
const cypherExamples = document.querySelector("#cypherExamples");
const cypherPromptInput = document.querySelector("#cypherPromptInput");
const cypherInput = document.querySelector("#cypherInput");
const cypherResults = document.querySelector("#cypherResults");
const runCypherButton = document.querySelector("#runCypherButton");
const generateCypherButton = document.querySelector("#generateCypherButton");
const synthesizeGraphButton = document.querySelector("#synthesizeGraphButton");
const cypherGenerationStatus = document.querySelector("#cypherGenerationStatus");
const graphSynthesis = document.querySelector("#graphSynthesis");
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
const lectureQuestion = document.querySelector("#lectureQuestion");
const lectureModePill = document.querySelector("#lectureModePill");
const lectureContrasts = document.querySelector("#lectureContrasts");
const lecturePoints = document.querySelector("#lecturePoints");
const lectureDemoAction = document.querySelector("#lectureDemoAction");
const lectureArchitecturePanel = document.querySelector("#lectureArchitecturePanel");
const lectureArchitectureName = document.querySelector("#lectureArchitectureName");
const lectureArchitectureFlow = document.querySelector("#lectureArchitectureFlow");
const lectureArchitectureTakeaway = document.querySelector("#lectureArchitectureTakeaway");
const lectureArchitectureRisk = document.querySelector("#lectureArchitectureRisk");
const lectureSpeakerNotes = document.querySelector("#lectureSpeakerNotes");
const quizQuestion = document.querySelector("#quizQuestion");
const quizOptions = document.querySelector("#quizOptions");
const quizAnswer = document.querySelector("#quizAnswer");
const revealQuizButton = document.querySelector("#revealQuizButton");
const applyLectureButton = document.querySelector("#applyLectureButton");
const prevLectureButton = document.querySelector("#prevLectureButton");
const nextLectureButton = document.querySelector("#nextLectureButton");
const zoomOutButton = document.querySelector("#zoomOutButton");
const zoomResetButton = document.querySelector("#zoomResetButton");
const zoomInButton = document.querySelector("#zoomInButton");
const zoomFitButton = document.querySelector("#zoomFitButton");
const graphDetail = document.querySelector("#graphDetail");
const graphLegend = document.querySelector("#graphLegend");

const COLORS = {
  character: "#1f6b4b",
  weapon: "#8d3d3a",
  place: "#bd8a2b",
  language: "#3c5f94",
  race: "#6f5aa8",
  document: "#b35f2d",
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

const GRAPH_RAG_STRATEGIES = [
  {
    id: "kg_index",
    shortName: "KG-as-Index",
    name: "KG-as-Index / Graph Boost",
    description:
      "Entidades da pergunta abrem um subgrafo k-hop; chunks vetoriais que mencionam sementes ou vizinhos recebem boost.",
    flow: ["pergunta", "entidades", "subgrafo", "boost", "sintese"],
    stageDetails: [
      ["Grounding", "Aliases resolvem entidades sementes."],
      ["Subgrafo", "Neo4j expande a vizinhanca k-hop."],
      ["Boost", "Chunks conectados sobem no ranking."],
      ["Sintese", "LLM recebe texto + estrutura."],
    ],
    bestFor: "Perguntas relacionais com texto narrativo e trilha estrutural auditavel.",
    graphRole: "Indice e reranker: o grafo altera a ordem das evidencias.",
    textRole: "Embedding traz candidatos; o boost favorece chunks conectados ao subgrafo.",
    lectureCue: "Use como baseline GraphRAG para a pergunta Frodo-Sauron.",
    visualHint: "Procure cosine alto e boost positivo nos docs recuperados.",
    risk: "Bom default para perguntas relacionais; amplifica erro quando o entity grounding erra.",
    references: [
      ["Microsoft Local Search", "https://microsoft.github.io/graphrag/query/local_search/"],
      ["KG2RAG", "https://arxiv.org/abs/2502.06864"],
    ],
  },
  {
    id: "vector_first",
    shortName: "Vector-first",
    name: "Vector-first / Graph Expansion",
    description:
      "Comeca com RAG vetorial puro; as entidades dos chunks recuperados viram sementes para expandir o grafo depois.",
    flow: ["embedding", "top-k", "mencoes", "grafo", "sintese"],
    stageDetails: [
      ["Busca inicial", "A pergunta vai primeiro ao indice vetorial."],
      ["Mencoes", "Entidades dos chunks viram sementes."],
      ["Expansao", "O grafo entra depois dos hits textuais."],
      ["Sintese", "Texto lidera; grafo explica."],
    ],
    bestFor: "Quando o texto deve liderar e o grafo deve explicar os hits.",
    graphRole: "Explicador posterior derivado dos chunks recuperados.",
    textRole: "Cosine puro decide o top-k inicial.",
    lectureCue: "Mostra como um RAG comum pode ganhar explicabilidade estrutural.",
    visualHint: "Score permanece cosine puro; sementes vêm de mencoes dos chunks.",
    risk: "Didatico para mostrar RAG que ganha explicabilidade depois, mas depende muito dos primeiros chunks.",
    references: [
      ["RAG", "https://arxiv.org/abs/2005.11401"],
      ["LightRAG", "https://arxiv.org/abs/2410.05779"],
    ],
  },
  {
    id: "graph_filter",
    shortName: "Graph filter",
    name: "Graph-Constrained Retrieval",
    description:
      "O subgrafo vira filtro duro dentro da busca vetorial: so entram chunks/falas que mencionam entidades ativadas pela vizinhanca estrutural.",
    flow: ["entidades", "subgrafo", "filtro", "rerank", "sintese"],
    stageDetails: [
      ["Grounding", "Entidades ativam um subgrafo."],
      ["Filtro", "Docs precisam mencionar sementes/vizinhos."],
      ["Rerank", "Vetores reordenam candidatos filtrados."],
      ["Sintese", "Mais precisao, menos cobertura."],
    ],
    bestFor: "Quando precisao e auditabilidade importam mais que recall amplo.",
    graphRole: "Filtro duro do conjunto candidato.",
    textRole: "Texto explica, mas precisa estar ligado por MENTIONS.",
    lectureCue: "Use para discutir precision vs recall.",
    visualHint: "Docs devem trazer seed hit, graph hit ou foco do subgrafo.",
    risk: "Aumenta precisao e auditabilidade, mas pode remover trechos bons sem ligacao MENTIONS.",
    references: [
      ["GRAG", "https://arxiv.org/abs/2405.16506"],
      ["KG2RAG", "https://arxiv.org/abs/2502.06864"],
    ],
  },
  {
    id: "path",
    shortName: "Paths",
    name: "Path / Connector Retrieval",
    description:
      "Perguntas entre entidades usam shortest paths e conectores 2-hop como foco do reranking textual.",
    flow: ["pares", "paths", "conectores", "chunks", "sintese"],
    stageDetails: [
      ["Pares", "Entidades formam pares relacionais."],
      ["Caminhos", "Shortest paths e conectores viram foco."],
      ["Rerank", "Chunks com entidades do caminho sobem."],
      ["Explicacao", "Resposta narra a ponte estrutural."],
    ],
    bestFor: "Perguntas 'como A se conecta a B?' ou 'quem faz a ponte?'.",
    graphRole: "Raciocinio por caminho e conectores 2-hop.",
    textRole: "Texto sustenta por que a ponte importa.",
    lectureCue: "Boa ponte com message passing e receptive field.",
    visualHint: "Veja Caminhos, Conectores e foco nos docs.",
    risk: "Caminhos curtos sao bons para explicar estrutura, mas nem sempre sao a melhor explicacao narrativa.",
    references: [
      ["HippoRAG", "https://arxiv.org/abs/2405.14831"],
      ["GNN-RAG", "https://arxiv.org/abs/2405.20139"],
    ],
  },
  {
    id: "community",
    shortName: "Community",
    name: "Community / Local-to-Global",
    description:
      "Usa comunidades estruturais dos personagens para trazer um contexto mais agregado antes da sintese.",
    flow: ["entidades", "comunidade", "centrais", "evidencias", "sintese"],
    stageDetails: [
      ["Comunidade", "Sementes localizam grupo estrutural."],
      ["Centrais", "Nos centrais viram contexto agregado."],
      ["Evidencias", "Chunks da comunidade recebem reforco."],
      ["Resumo", "Combina local e global."],
    ],
    bestFor: "Perguntas amplas sobre grupos, aliancas e nucleos narrativos.",
    graphRole: "Agregador local-to-global por comunidade.",
    textRole: "Texto representa a comunidade, nao so a entidade exata.",
    lectureCue: "Use para explicar GraphRAG global/local sem summaries precomputados.",
    visualHint: "Subgrafo deve destacar personagens da mesma comunidade.",
    risk: "Bom para perguntas amplas, mas pode diluir relacoes muito especificas.",
    references: [
      ["From Local to Global", "https://arxiv.org/abs/2404.16130"],
      ["Global Search", "https://microsoft.github.io/graphrag/examples_notebooks/global_search/"],
    ],
  },
  {
    id: "cypher",
    shortName: "Cypher",
    name: "Symbolic Cypher / MENTIONS Query",
    description:
      "Entidades da pergunta alimentam uma consulta simbolica auditavel; aqui a demo usa template segura e a aba Graph mostra geracao por LLM.",
    flow: ["entidades", "Cypher", "linhas/docs", "grafo MENTIONS", "sintese"],
    stageDetails: [
      ["Entidades", "Nomes viram parametros da query."],
      ["Cypher", "Consulta simbolica busca via MENTIONS."],
      ["Linhas", "Score vem de entityHits."],
      ["Sintese", "Query vira evidencia auditavel."],
    ],
    bestFor: "Perguntas que podem virar consulta clara por entidades, relacoes ou documentos ligados.",
    graphRole: "Plano simbolico e auditavel.",
    textRole: "Docs entram por MENTIONS; embedding nao decide o ranking principal.",
    lectureCue: "Conecta a aba Graph com query-driven GraphRAG.",
    visualHint: "Trace deve mostrar symbolic entity hits e a Cypher equivalente.",
    risk: "Excelente para auditoria, mas depende do entity grounding e do template escolhido.",
    references: [
      ["Neo4j Cypher", "https://neo4j.com/docs/cypher-manual/current/"],
      ["GraphRAG Local Search", "https://microsoft.github.io/graphrag/query/local_search/"],
    ],
  },
];

const CONTROL_CONTEXT = {
  overview: {
    kicker: "Explorer",
    title: "Modos de Recuperacao",
    hint: "Selecione a estrategia para analisar entidades, texto e estrutura do corpus.",
  },
  rag: {
    kicker: "RAG",
    title: "Busca textual",
    hint: "A pergunta vira embedding e recupera evidencias textuais por similaridade.",
  },
  graph: {
    kicker: "Graph",
    title: "Exploracao estrutural",
    hint: "Use centro/hops para navegar livremente; use Cypher para gerar um subgrafo auditavel.",
  },
  graphrag: {
    kicker: "GraphRAG",
    title: "Texto guiado por grafo",
    hint: "Entidades ativam subgrafos; subgrafos reforcam a recuperacao textual.",
  },
  compare: {
    kicker: "Compare",
    title: "Mesma pergunta, tres metodos",
    hint: "RAG, Graph e GraphRAG retornam evidencias complementares para a mesma pergunta.",
  },
  lecture: {
    kicker: "Flow",
    title: "Exploracao Guiada",
    hint: "Uma sequencia compacta para percorrer corpus, RAG, grafo, GraphRAG e comparacao.",
  },
};

const state = {
  cypherExamples: [],
  activeCypherExample: null,
  graphRagStrategies: GRAPH_RAG_STRATEGIES.map(normalizeStrategy),
  activeGraphRagStrategy: "kg_index",
  lectureSteps: [],
  lectureIndex: 0,
  hasCompare: false,
  lastAnswerMode: null,
  lastQuestion: "",
  lastGraph: null,
  lastCypherResult: null,
  selectedGraphNodeId: null,
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

function normalizeStrategy(strategy) {
  return {
    ...strategy,
    references: (strategy.references || []).map((reference) =>
      Array.isArray(reference) ? reference : [reference.label, reference.url],
    ),
    stageDetails: (strategy.stageDetails || strategy.flow || []).map((stage) => {
      if (Array.isArray(stage)) return { label: stage[0], detail: stage[1] || "" };
      if (typeof stage === "string") return { label: stage, detail: "" };
      return stage;
    }),
  };
}

async function loadGraphRagStrategies() {
  try {
    const payload = await getJson("/api/graphrag/strategies");
    if (payload.strategies?.length) {
      state.graphRagStrategies = payload.strategies.map(normalizeStrategy);
    }
  } catch (_error) {
    state.graphRagStrategies = GRAPH_RAG_STRATEGIES.map(normalizeStrategy);
  }
  renderGraphRagStrategies();
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

function renderGraphRagTracePlaceholder() {
  if (!traceEntities) return;
  const strategy = strategyById(state.activeGraphRagStrategy);
  renderActiveGraphRagStrategy();
  traceGraphTitle.textContent = `${strategy.shortName || "GraphRAG"}: Graph step`;
  traceRetrievalTitle.textContent = `${strategy.shortName || "GraphRAG"}: Evidence step`;
  tracePromptTitle.textContent = `${strategy.shortName || "GraphRAG"} Prompt`;
  traceGroundingMeta.textContent = "aguardando execucao";
  traceEntities.innerHTML = `<span class="trace-empty">A pergunta vai resolver entidades por aliases.</span>`;
  traceGraphMeta.textContent = strategy.shortName || "GraphRAG";
  traceGraphEvidence.innerHTML = `
    <div class="trace-row">
      <strong>Fluxo</strong>
      <span>${escapeHtml((strategy.flow || []).join(" -> "))}</span>
    </div>
    <div class="trace-row">
      <strong>Grafo</strong>
      <span>${escapeHtml(strategy.graphRole || strategy.description || "")}</span>
    </div>
    <div class="trace-row">
      <strong>Texto</strong>
      <span>${escapeHtml(strategy.textRole || "Evidencias textuais entram conforme a estrategia selecionada.")}</span>
    </div>
    <div class="trace-row">
      <strong>Aula</strong>
      <span>${escapeHtml(strategy.description || "Selecione uma estrategia GraphRAG.")}</span>
    </div>
  `;
  traceRetrievalMeta.textContent = strategy.visualHint || "aguardando retrieval";
  traceDocuments.innerHTML = `<article class="trace-doc-empty">Execute o GraphRAG para ver como esta estrategia escolhe chunks, falas ou evidencias simbolicas.</article>`;
  tracePromptMeta.textContent = "contexto montado pela estrategia";
  tracePromptSections.innerHTML = "";
  tracePromptPreview.textContent = "O prompt final enviado ao LLM aparece aqui depois da execução.";
}

function renderComparePlaceholder() {
  compareResults.innerHTML = `
    <article class="compare-placeholder">RAG textual: cosine puro sobre livros e scripts.</article>
    <article class="compare-placeholder">Graph estrutural: caminhos, vizinhos e relacoes, sem chunks.</article>
    <article class="compare-placeholder">GraphRAG: subgrafo + evidencias textuais com boost.</article>
  `;
}

function updateViewPlaceholders(name) {
  if (name === "rag" && state.lastAnswerMode !== "rag") {
    setAnswerPlaceholder("RAG aguardando execucao", "A busca preenche resposta, chunks recuperados e contexto textual.");
    ragEvidence.className = "rag-evidence-groups";
    ragEvidence.innerHTML = `<article class="evidence-card"><h3>Aguardando busca</h3><p>Os chunks e falas recuperados aparecem aqui depois da execucao vetorial.</p></article>`;
  }
  if (name === "graphrag" && state.lastAnswerMode !== "hybrid") {
    setAnswerPlaceholder("GraphRAG aguardando execucao", "Execute o GraphRAG para preencher entidades, subgrafo, evidencias e resposta.");
    resetPipelinePlaceholder();
    renderGraphRagTracePlaceholder();
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

function fitGraph() {
  const width = graphSvg.clientWidth || 760;
  const height = graphSvg.clientHeight || 520;
  setGraphViewBox(width, height);
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

function graphNodeLabel(node) {
  return node.name || node.sourceTitle || node.chapterTitle || node.id || "no";
}

function graphNodeMeta(node) {
  const labels = (node.labels || []).join(", ") || "Entity";
  const parts = [labels];
  if (node.race) parts.push(`raca ${node.race}`);
  if (node.community != null) parts.push(`comunidade ${node.community}`);
  if (node.pagerank != null) parts.push(`PR ${Number(node.pagerank || 0).toFixed(4)}`);
  return parts.join(" · ");
}

function edgeWeightLabel(edge) {
  const weight = edge.weight ?? edge.confidence ?? 1;
  const confidence = edge.confidence == null ? "" : ` · conf ${Number(edge.confidence).toFixed(3)}`;
  const source = edge.sourceDataset ? ` · ${edge.sourceDataset}` : "";
  return `peso ${Number(weight || 0).toFixed(3)}${confidence}${source}`;
}

function renderGraphLegend(nodes, edges) {
  if (!graphLegend) return;
  const nodeCounts = nodes.reduce((counts, node) => {
    const group = node.group || "entity";
    counts[group] = (counts[group] || 0) + 1;
    return counts;
  }, {});
  const edgeCounts = edges.reduce((counts, edge) => {
    const type = edge.type || "REL";
    counts[type] = (counts[type] || 0) + 1;
    return counts;
  }, {});
  const nodeItems = Object.entries(nodeCounts)
    .sort((left, right) => right[1] - left[1])
    .map(([group, count]) => `<span><i style="background:${COLORS[group] || COLORS.entity}"></i>${escapeHtml(group)} ${count}</span>`)
    .join("");
  const edgeItems = Object.entries(edgeCounts)
    .sort((left, right) => right[1] - left[1])
    .slice(0, 5)
    .map(([type, count]) => `<span>${escapeHtml(type)} ${count}</span>`)
    .join("");
  graphLegend.innerHTML = nodeItems + edgeItems;
}

function setGraphDetail(html) {
  if (graphDetail) graphDetail.innerHTML = html;
}

function clearGraphHighlight() {
  state.selectedGraphNodeId = null;
  graphSvg.querySelectorAll(".node-group, .edge").forEach((item) => {
    item.classList.remove("selected", "neighbor", "dimmed");
  });
}

function selectGraphNode(node, graph) {
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  const neighborIds = new Set([node.id]);
  for (const edge of edges) {
    if (edge.source === node.id) neighborIds.add(edge.target);
    if (edge.target === node.id) neighborIds.add(edge.source);
  }
  state.selectedGraphNodeId = node.id;
  graphSvg.querySelectorAll(".node-group").forEach((item) => {
    const id = item.dataset.nodeId;
    item.classList.toggle("selected", id === node.id);
    item.classList.toggle("neighbor", id !== node.id && neighborIds.has(id));
    item.classList.toggle("dimmed", !neighborIds.has(id));
  });
  graphSvg.querySelectorAll(".edge").forEach((item) => {
    const connected = item.dataset.source === node.id || item.dataset.target === node.id;
    item.classList.toggle("selected", connected);
    item.classList.toggle("dimmed", !connected);
  });
  const degree = edges.filter((edge) => edge.source === node.id || edge.target === node.id).length;
  setGraphDetail(`
    <strong>${escapeHtml(graphNodeLabel(node))}</strong>
    <span>${escapeHtml(graphNodeMeta(node))}</span>
    <span>${degree} arestas incidentes · ${nodes.length} nos no subgrafo</span>
  `);
}

function selectGraphEdge(edge) {
  clearGraphHighlight();
  graphSvg.querySelectorAll(".edge").forEach((item) => {
    item.classList.toggle("selected", item.dataset.edgeId === edge.id);
    item.classList.toggle("dimmed", item.dataset.edgeId !== edge.id);
  });
  setGraphDetail(`
    <strong>${escapeHtml(edge.sourceName || edge.source)} -[${escapeHtml(edge.type || "REL")}]-> ${escapeHtml(edge.targetName || edge.target)}</strong>
    <span>${escapeHtml(edgeWeightLabel(edge))}</span>
    <span>${escapeHtml(edge.method || "relacao observada")}</span>
  `);
}

function renderGraph(graph, options = {}) {
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  state.lastGraph = graph;
  state.selectedGraphNodeId = null;
  synthesizeGraphButton.disabled = nodes.length === 0 && !state.lastCypherResult;
  graphSvg.innerHTML = "";
  const width = graphSvg.clientWidth || 760;
  const height = graphSvg.clientHeight || 520;
  setGraphViewBox(width, height);
  renderGraphLegend(nodes, edges);

  if (nodes.length === 0) {
    graphMeta.textContent = "Grafo vazio. Banco sem dados carregados.";
    setGraphDetail("Sem nos para inspecionar.");
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
    line.dataset.edgeId = edge.id || "";
    line.dataset.source = edge.source || "";
    line.dataset.target = edge.target || "";
    line.setAttribute("stroke-width", Math.min(5, 0.7 + Math.log2(Number(edge.weight || 1) + 1)));
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    const weight = edge.weight || edge.confidence || 1;
    title.textContent = `${edge.sourceName} -[${edge.type}, peso=${weight}]-> ${edge.targetName}`;
    line.appendChild(title);
    line.addEventListener("click", (event) => {
      event.stopPropagation();
      selectGraphEdge(edge);
    });
    edgeLayer.appendChild(line);
  }

  for (const node of nodes) {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const x = scale.x(node.x || 0);
    const y = scale.y(node.y || 0);
    const color = COLORS[node.group] || COLORS.entity;
    group.setAttribute("transform", `translate(${x}, ${y})`);
    group.setAttribute("class", "node-group");
    group.dataset.nodeId = node.id || "";

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("r", Math.max(7, Number(node.size || 10)));
    circle.setAttribute("fill", color);
    circle.setAttribute("class", "node");
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${graphNodeLabel(node)} · ${graphNodeMeta(node)}`;
    circle.appendChild(title);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "node-label");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("y", -Math.max(11, Number(node.size || 10)) - 6);
    label.textContent = graphNodeLabel(node);

    group.append(circle);
    if (labeledNodeIds.has(node.id)) {
      group.appendChild(label);
    }
    group.addEventListener("dblclick", () => {
      centerInput.value = graphNodeLabel(node);
      loadGraph();
    });
    group.addEventListener("click", (event) => {
      event.stopPropagation();
      selectGraphNode(node, graph);
    });
    nodeLayer.appendChild(group);
  }

  graphMeta.textContent = options.meta || `${nodes.length} nos · ${edges.length} arestas`;
  setGraphDetail(`${nodes.length} nos · ${edges.length} arestas. Clique em um no ou aresta para detalhes.`);
}

async function loadGraph() {
  const center = encodeURIComponent(centerInput.value.trim());
  const hops = encodeURIComponent(hopsInput.value);
  graphMeta.textContent = "Carregando subgrafo...";
  try {
    const graph = await getJson(`/api/graph?center=${center}&hops=${hops}&limit=180`);
    state.lastCypherResult = {
      query: `center=${decodeURIComponent(center || "")}; hops=${decodeURIComponent(hops)}`,
      rows: [],
      graph,
    };
    renderGraph(graph, { meta: `Centro/Hops · ${graph.nodes?.length || 0} nos · ${graph.edges?.length || 0} arestas` });
    graphSynthesis.innerHTML = `<strong>Leitura Graph-only</strong><p>Subgrafo por centro/hops carregado. Use Executar + Explicar para gerar uma leitura estrutural da query.</p>`;
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
    graph_rag_strategy: graphRagStrategyInput?.value || state.activeGraphRagStrategy || "kg_index",
    model: modelInput.value || null,
    use_llm: llmInput.checked,
  };
}

function strategyById(strategyId) {
  return state.graphRagStrategies.find((item) => item.id === strategyId) || state.graphRagStrategies[0];
}

function renderGraphRagStrategies() {
  if (!graphRagStrategyCards) return;
  graphRagStrategyCards.innerHTML = state.graphRagStrategies
    .map(
      (strategy) => `
        <button class="strategy-card ${strategy.id === state.activeGraphRagStrategy ? "active" : ""}" data-strategy-id="${escapeHtml(strategy.id)}" type="button">
          <span>${escapeHtml(strategy.shortName || strategy.id)}</span>
          <strong>${escapeHtml(strategy.name)}</strong>
          <p>${escapeHtml(strategy.description)}</p>
          <div class="strategy-card-meta">
            <small>${escapeHtml(strategy.graphRole || "grafo como contexto")}</small>
            <small>${escapeHtml(strategy.visualHint || (strategy.flow || []).join(" -> "))}</small>
          </div>
        </button>
      `,
    )
    .join("");
  graphRagStrategyCards.querySelectorAll("[data-strategy-id]").forEach((button) => {
    button.addEventListener("click", () => setGraphRagStrategy(button.dataset.strategyId));
  });
  renderActiveGraphRagStrategy();
}

function renderActiveGraphRagStrategy() {
  const strategy = strategyById(state.activeGraphRagStrategy);
  if (graphRagStrategyInput && graphRagStrategyInput.value !== strategy.id) graphRagStrategyInput.value = strategy.id;
  if (!activeStrategyName) return;
  activeStrategyKicker.textContent = "Variante ativa";
  activeStrategyName.textContent = strategy.name;
  activeStrategyDescription.textContent = strategy.description;
  activeStrategyStageMap.innerHTML = (strategy.stageDetails || [])
    .map(
      (stage, index) => `
        <article>
          <strong>${index + 1}</strong>
          <span>${escapeHtml(stage.label || "")}</span>
          <p>${escapeHtml(stage.detail || "")}</p>
        </article>
      `,
    )
    .join("");
  activeStrategyBestFor.textContent = strategy.bestFor || "Perguntas que precisam combinar grafo e texto.";
  activeStrategyGraphRole.textContent = strategy.graphRole || "O grafo define ou explica o contexto estrutural.";
  activeStrategyTextRole.textContent = strategy.textRole || "O texto sustenta a resposta com evidencias recuperadas.";
  activeStrategyLectureCue.textContent = strategy.lectureCue || "Use a variante para contrastar onde o grafo entra no pipeline.";
  activeStrategyVisualHint.textContent = strategy.visualHint ? `No trace: ${strategy.visualHint}` : "";
  activeStrategyRisk.textContent = strategy.risk ? `Risco: ${strategy.risk}` : "";
  activeStrategyReferences.innerHTML = (strategy.references || [])
    .map((reference) => {
      const label = Array.isArray(reference) ? reference[0] : reference.label;
      const url = Array.isArray(reference) ? reference[1] : reference.url;
      return `<a href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`;
    })
    .join("");
  graphRagStrategyCards?.querySelectorAll("[data-strategy-id]").forEach((button) => {
    button.classList.toggle("active", button.dataset.strategyId === strategy.id);
  });
}

function setGraphRagStrategy(strategyId) {
  const strategy = strategyById(strategyId);
  state.activeGraphRagStrategy = strategy.id;
  renderActiveGraphRagStrategy();
  if (activeView() === "graphrag") {
    state.lastAnswerMode = null;
    setAnswerPlaceholder(strategy.shortName || "GraphRAG", `Variante selecionada: ${strategy.name}. Execute para preencher evidencias e trace.`);
    resetPipelinePlaceholder();
    renderGraphRagTracePlaceholder();
  }
}

function sourceLabel(sourceType) {
  if (sourceType === "book") return "Livro";
  if (sourceType === "dialogue") return "Script";
  return "Texto";
}

function scoreLabel(doc) {
  const vector = doc.vectorScore;
  const score = Number(doc.score || 0);
  const boost = Number(doc.graphBoost || 0);
  const base = vector == null ? `score ${score.toFixed(3)}` : `cosine ${Number(vector).toFixed(3)}`;
  return boost > 0 ? `${base} · boost ${boost.toFixed(3)} · final ${score.toFixed(3)}` : base;
}

function groupDocumentsBySource(documents) {
  return (documents || []).reduce((groups, doc) => {
    const key = doc.sourceType || "other";
    if (!groups[key]) groups[key] = [];
    groups[key].push(doc);
    return groups;
  }, {});
}

function renderEvidenceCards(documents, target = ragEvidence) {
  target.innerHTML = "";
  if (!documents || !documents.length) {
    target.innerHTML = `<article class="evidence-card"><h3>Nenhuma evidencia textual</h3><p>Ajuste a pergunta ou gere o indice com make vectors.</p></article>`;
    return;
  }
  for (const doc of documents) {
    const card = document.createElement("article");
    card.className = `evidence-card source-${doc.sourceType || "other"}`;
    const source = doc.sourceTitle || doc.sourceType || "texto";
    const chapter = doc.chapterTitle ? ` / ${doc.chapterTitle}` : "";
    const speaker = doc.speaker ? ` / ${doc.speaker}` : "";
    const mentions = (doc.mentions || []).slice(0, 8).join(", ");
    const rank = doc.sourceRank ? `#${doc.sourceRank}` : "";
    card.innerHTML = `
      <div class="evidence-top">
        <span class="source-badge">${escapeHtml(sourceLabel(doc.sourceType))}${rank ? ` ${rank}` : ""}</span>
        <span class="card-kicker">${escapeHtml(doc.retrievalMethod || "retrieval")} · ${escapeHtml(scoreLabel(doc))}</span>
      </div>
      <h3>${escapeHtml(source + chapter + speaker)}</h3>
      <p>${escapeHtml(doc.snippet || doc.text || "").slice(0, 720)}</p>
      <small>${mentions ? `mencoes: ${escapeHtml(mentions)}` : "sem mencoes detectadas"}</small>
    `;
    target.appendChild(card);
  }
}

function renderRagEvidence(result) {
  const bySource = result.documentsBySource || groupDocumentsBySource(result.documents || []);
  const groups = [
    ["book", "Livros"],
    ["dialogue", "Scripts"],
  ];
  ragEvidence.className = "rag-evidence-groups";
  ragEvidence.innerHTML = "";
  let rendered = 0;
  for (const [sourceType, title] of groups) {
    const docs = bySource[sourceType] || [];
    if (!docs.length) continue;
    rendered += docs.length;
    const section = document.createElement("section");
    section.className = "rag-source-group";
    const grid = document.createElement("div");
    grid.className = "evidence-cards";
    section.innerHTML = `
      <div class="source-group-head">
        <h3>${escapeHtml(title)}</h3>
        <span>${docs.length} de top-k ${result.topKPerSource || result.topK || topKInput.value}</span>
      </div>
    `;
    section.appendChild(grid);
    ragEvidence.appendChild(section);
    renderEvidenceCards(docs, grid);
  }
  if (!rendered) {
    ragEvidence.innerHTML = `<article class="evidence-card"><h3>Nenhuma evidencia textual</h3><p>Ajuste a pergunta ou gere o indice com make vectors.</p></article>`;
  }
}

function renderGraphRagTrace(result) {
  if (!traceEntities) return;
  const trace = result.trace || {};
  const grounding = trace.grounding || {};
  const graph = trace.graph || {};
  const retrieval = trace.retrieval || {};
  const prompt = trace.prompt || {};
  const variant = trace.variant || {};
  const runtime = trace.strategy?.runtime || {};
  const steps = trace.steps || [];
  const stepById = steps.reduce((lookup, step) => {
    lookup[step.id] = step;
    return lookup;
  }, {});

  document.querySelector("#pipeEntities").textContent = stepById.grounding?.value || `${grounding.entityCount || 0} entidades`;
  document.querySelector("#pipeGraph").textContent = stepById.graph?.value || `${graph.nodeCount || 0} nos / ${graph.edgeCount || 0} arestas`;
  document.querySelector("#pipeVectors").textContent = stepById.retrieval?.value || `${retrieval.documents || 0} docs`;
  document.querySelector("#pipeAnswer").textContent = stepById.prompt?.value || result.llmStatus || "-";

  if (variant.id) {
    state.activeGraphRagStrategy = variant.id;
    renderActiveGraphRagStrategy();
  }
  const activeStrategy = strategyById(variant.id || state.activeGraphRagStrategy);
  traceGraphTitle.textContent = `${activeStrategy.shortName || "GraphRAG"}: Graph step`;
  traceRetrievalTitle.textContent = `${activeStrategy.shortName || "GraphRAG"}: Evidence step`;
  tracePromptTitle.textContent = `${activeStrategy.shortName || "GraphRAG"} Prompt`;

  traceGroundingMeta.textContent = `${grounding.entityCount || 0} entidades resolvidas`;
  const entities = grounding.entities || [];
  traceEntities.innerHTML = entities.length
    ? entities
        .map((entity) => {
          const aliases = (entity.aliases || []).slice(0, 4).join(", ");
          const labels = (entity.labels || []).filter((label) => label !== "Entity").join(", ") || entity.kind || "Entity";
          return `
            <span class="trace-chip">
              <strong>${escapeHtml(entity.name)}</strong>
              <small>match: ${escapeHtml(entity.matchedAlias || entity.name)} · ${escapeHtml(labels)}</small>
              ${aliases ? `<em>aliases: ${escapeHtml(aliases)}</em>` : ""}
            </span>
          `;
        })
        .join("")
    : `<span class="trace-empty">Nenhuma entidade foi resolvida a partir da pergunta.</span>`;

  const relTypes = Object.entries(graph.relationshipTypes || {})
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, 5)
    .map(([name, count]) => `${name} ${count}`)
    .join(" · ");
  const graphMetaSuffix = (variant.id || state.activeGraphRagStrategy) === "cypher"
    ? "query MENTIONS"
    : `hops ${graph.hops || hopsInput.value}`;
  traceGraphMeta.textContent = `${graph.nodeCount || 0} nos · ${graph.edgeCount || 0} arestas · ${graphMetaSuffix}`;
  const paths = graph.paths || [];
  const directEdges = graph.directEdges || [];
  const connectors = graph.connectors || [];
  const runtimeNotes = runtime.notes || trace.strategy?.notes || [];
  const runtimeStatus = runtime.degraded
    ? `fallback para ${runtime.fallbackStrategy || "estrategia alternativa"}`
    : "execucao principal";
  const isSymbolicQuery = (variant.id || state.activeGraphRagStrategy) === "cypher";
  const graphSeedText =
    [
      ...new Set(
        [
          ...(grounding.graphSeeds || []),
          ...(grounding.derivedEntities || []),
          ...(grounding.pathEntities || []),
          ...(grounding.communityEntities || []).slice(0, 6),
          ...(grounding.queryEntities || []),
        ].filter(Boolean),
      ),
    ]
      .slice(0, 14)
      .join(", ") || "sem sementes adicionais";
  const relationalRows = isSymbolicQuery
    ? `
      <div class="trace-row">
        <strong>Padrao</strong>
        <span>RetrievalDocument -[:MENTIONS]-&gt; Entity</span>
      </div>
      <div class="trace-row">
        <strong>Resultado</strong>
        <span>${retrieval.documents || 0} documentos ranqueados por entityHits; embedding nao decide este ranking.</span>
      </div>
    `
    : `
      <div class="trace-row">
        <strong>Caminhos</strong>
        <span>${
          paths.length
            ? paths.map((item) => escapeHtml(`${item.source} -> ${item.target}: ${item.path.join(" -> ")}`)).join("<br>")
            : "sem caminho curto"
        }</span>
      </div>
      <div class="trace-row">
        <strong>Diretas</strong>
        <span>${
          directEdges.length
            ? directEdges
                .slice(0, 4)
                .map((edge) => escapeHtml(`${edge.sourceName} -[${edge.type}]-> ${edge.targetName}`))
                .join("<br>")
            : "sem aresta direta entre seeds"
        }</span>
      </div>
      <div class="trace-row">
        <strong>Conectores</strong>
        <span>${
          connectors.length
            ? connectors.map((item) => escapeHtml(`${item.name} (${Number(item.combinedWeight || 0).toFixed(0)})`)).join(", ")
            : "sem conectores 2-hop"
        }</span>
      </div>
    `;
  traceGraphEvidence.innerHTML = `
    <div class="trace-row">
      <strong>Variante</strong>
      <span>${escapeHtml(variant.name || "GraphRAG")}<br>${escapeHtml(variant.subtitle || "")}</span>
    </div>
    <div class="trace-row">
      <strong>Status</strong>
      <span>${escapeHtml(runtimeStatus)}${runtimeNotes.length ? `<br>${escapeHtml(runtimeNotes.join(" "))}` : ""}</span>
    </div>
    <div class="trace-row">
      <strong>Uso</strong>
      <span>${escapeHtml(activeStrategy.bestFor || variant.description || "")}</span>
    </div>
    <div class="trace-row">
      <strong>Grafo</strong>
      <span>${escapeHtml(activeStrategy.graphRole || trace.strategy?.graph || "contexto estrutural")}</span>
    </div>
    <div class="trace-row">
      <strong>Query</strong>
      <span>${escapeHtml(graph.query?.label || "deterministic k-hop expansion")}</span>
    </div>
    <div class="trace-row">
      <strong>Tipos</strong>
      <span>${escapeHtml(relTypes || "sem arestas recuperadas")}</span>
    </div>
    ${relationalRows}
    <div class="trace-row">
      <strong>Sementes</strong>
      <span>${escapeHtml(graphSeedText)}</span>
    </div>
    <div class="trace-row">
      <strong>Sinal visual</strong>
      <span>${escapeHtml(activeStrategy.visualHint || "Compare grafo, score e documentos recuperados.")}</span>
    </div>
    <details class="trace-cypher">
      <summary>Cypher deterministico equivalente</summary>
      <pre>${escapeHtml(graph.query?.cypher || "")}</pre>
    </details>
  `;

  traceRetrievalMeta.textContent = `${retrieval.documents || 0} docs · ${retrieval.boostedDocuments || 0} boosted · ${retrieval.scoreMode || "score"}`;
  const docs = retrieval.topDocuments || [];
  traceDocuments.innerHTML = docs.length
    ? docs.map((doc) => renderTraceDocument(doc)).join("")
    : `<article class="trace-doc-empty">Nenhum chunk textual recuperado para esta execucao.</article>`;

  tracePromptMeta.textContent = `${prompt.selectedChars || 0} chars · ~${prompt.estimatedTokens || 0} tokens`;
  tracePromptSections.innerHTML = (prompt.sections || [])
    .map(
      (section) => `
        <span class="${section.enabled ? "enabled" : "disabled"}">
          <strong>${escapeHtml(section.name)}</strong>
          <small>${section.enabled ? `${section.chars} chars` : "inativo"}</small>
        </span>
      `,
    )
    .join("");
  tracePromptPreview.textContent = prompt.preview || "Sem contexto serializado para este modo.";
}

function renderTraceDocument(doc) {
  const title = [doc.sourceTitle, doc.chapterTitle, doc.speaker ? `fala de ${doc.speaker}` : ""]
    .filter(Boolean)
    .join(" / ");
  const vector = Number(doc.vectorScore || 0);
  const boost = Number(doc.graphBoost || 0);
  const finalScore = Number(doc.score || 0);
  const boostWidth = Math.min(100, Math.max(0, boost * 260));
  const seedHits = (doc.seedHits || []).join(", ");
  const graphHits = (doc.graphHits || []).slice(0, 5).join(", ");
  const focusHits = (doc.strategyFocusHits || []).slice(0, 5).join(", ");
  return `
    <article class="trace-doc-card">
      <div class="trace-doc-head">
        <span class="source-badge">${escapeHtml(sourceLabel(doc.sourceType))}${doc.sourceRank ? ` #${doc.sourceRank}` : ""}</span>
        <span>${escapeHtml(doc.boostReason || "ranked by vector similarity")}</span>
      </div>
      <h3>${escapeHtml(title || "Documento recuperado")}</h3>
      <div class="trace-score-grid">
        <span><strong>${vector.toFixed(3)}</strong><small>cosine</small></span>
        <span><strong>${boost.toFixed(3)}</strong><small>boost</small></span>
        <span><strong>${finalScore.toFixed(3)}</strong><small>final</small></span>
      </div>
      <div class="boost-meter" aria-label="Graph boost ${boost.toFixed(3)}">
        <span style="width: ${boostWidth}%"></span>
      </div>
      <p>${escapeHtml(doc.snippet || "").slice(0, 380)}</p>
      <small>${seedHits ? `seed hits: ${escapeHtml(seedHits)}` : "sem seed hit"}${graphHits ? ` · subgrafo: ${escapeHtml(graphHits)}` : ""}${focusHits ? ` · foco: ${escapeHtml(focusHits)}` : ""}</small>
    </article>
  `;
}

function answerMetaText(result, fallbackTopK) {
  const mode = result.mode || "retrieval";
  if (mode === "graph") {
    const graph = result.graph || {};
    return `graph · subgrafo · ${graph.nodes?.length || 0} nos / ${graph.edges?.length || 0} arestas`;
  }
  const scoreMode = result.retrieval?.scoreMode || "score";
  const strategy = result.graphRagStrategy ? ` · ${strategyById(result.graphRagStrategy).shortName || result.graphRagStrategy}` : "";
  const topK = result.topKPerSource
    ? `top-k ${result.topKPerSource} por fonte`
    : `top-k ${result.topK || fallbackTopK}`;
  return `${mode}${strategy} · ${scoreMode} · ${topK}`;
}

async function askQuestion(modeOverride = null) {
  const requestedMode = modeOverride || modeInput.value;
  answerBox.textContent = "";
  contextBox.textContent = "";
  llmStatus.textContent = llmInput.checked ? "Ollama sintetizando evidencias..." : "Retrieval-only: sem sintese com LLM";
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
    answerMeta.textContent = answerMetaText(result, topKInput.value);
    llmStatus.textContent = `${result.model} · ${result.llmStatus}`;
    if ((result.mode || requestedMode) === "rag") renderRagEvidence(result);
    if (result.graph && result.graph.nodes) renderGraph(result.graph);
    updatePipeline(result);
    if ((result.mode || requestedMode) === "hybrid") renderGraphRagTrace(result);
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
  llmStatus.textContent = llmInput.checked ? "Comparando com sintese Ollama..." : "Comparando em retrieval-only...";
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
      renderGraphRagTrace(hybrid);
    }
  } catch (error) {
    llmStatus.textContent = `Erro: ${error.message}`;
  } finally {
    compareButton.disabled = false;
    compareViewButton.disabled = false;
  }
}

async function compareGraphRagStrategies() {
  if (!strategyCompareResults) return;
  strategyCompareResults.innerHTML = `<article class="trace-doc-empty">Comparando estrategias em retrieval-only...</article>`;
  strategyCompareButton.disabled = true;
  try {
    const result = await getJson("/api/graphrag/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload("hybrid"), use_llm: false }),
    });
    renderStrategyCompare(result.results || {});
  } catch (error) {
    strategyCompareResults.innerHTML = `<article class="trace-doc-empty error-text">${escapeHtml(error.message)}</article>`;
  } finally {
    strategyCompareButton.disabled = false;
  }
}

function renderStrategyCompare(results) {
  const order = state.graphRagStrategies.map((item) => item.id);
  strategyCompareResults.innerHTML = order
    .map((strategyId) => {
      const result = results[strategyId];
      const strategy = strategyById(strategyId);
      if (!result) return "";
      const trace = result.trace || {};
      const graph = trace.graph || {};
      const retrieval = trace.retrieval || {};
      const grounding = trace.grounding || {};
      const runtime = trace.strategy?.runtime || {};
      const topDoc = (retrieval.topDocuments || [])[0];
      const topDocText = topDoc
        ? `${sourceLabel(topDoc.sourceType)} · ${topDoc.sourceTitle || "texto"} · ${scoreLabel(topDoc)}`
        : "sem chunk textual";
      const seeds = [
        ...(grounding.graphSeeds || []),
        ...(grounding.derivedEntities || []),
        ...(grounding.pathEntities || []),
        ...(grounding.communityEntities || []),
        ...(grounding.queryEntities || []),
      ];
      const seedText = [...new Set(seeds)].slice(0, 5).join(", ") || "sem sementes";
      return `
        <article class="strategy-result-card" data-strategy-result="${escapeHtml(strategyId)}">
          <h3>${escapeHtml(strategy.shortName || strategy.name)}</h3>
          <div class="strategy-result-metrics">
            <span><strong>${graph.nodeCount || 0}</strong><small>nos</small></span>
            <span><strong>${graph.edgeCount || 0}</strong><small>arestas</small></span>
            <span><strong>${retrieval.documents || 0}</strong><small>docs</small></span>
          </div>
          <p><strong>Papel:</strong> ${escapeHtml(strategy.graphRole || "")}</p>
          <p><strong>Sementes:</strong> ${escapeHtml(seedText)}</p>
          <p><strong>Score:</strong> ${escapeHtml(retrieval.scoreMode || "n/a")}</p>
          <p><strong>Status:</strong> ${escapeHtml(runtime.degraded ? `fallback ${runtime.fallbackStrategy || ""}` : "principal")}</p>
          <p><strong>Top evidencia:</strong> ${escapeHtml(topDocText)}</p>
          <button class="secondary-button compact-strategy-button" data-open-strategy="${escapeHtml(strategyId)}" type="button">Ver detalhes</button>
        </article>
      `;
    })
    .join("");
  strategyCompareResults.querySelectorAll("[data-open-strategy]").forEach((button) => {
    button.addEventListener("click", () => {
      setGraphRagStrategy(button.dataset.openStrategy);
      strategyCompareResults.scrollIntoView({ block: "nearest" });
    });
  });
}

function renderCompare(results) {
  const labels = {
    rag: "RAG Vetorial",
    graph: "Graph",
    hybrid: "GraphRAG",
  };
  const limits = {
    rag: "Texto puro: ranking por similaridade sobre livros e scripts.",
    graph: "Graph-only: subgrafo, caminhos e centralidade; zero chunks recuperados.",
    hybrid: `GraphRAG: estrategia selecionada (${strategyById(graphRagStrategyInput?.value || "kg_index").shortName}).`,
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
    const counts = result.retrieval?.bySource || {};
    const sourceText = mode === "rag"
      ? ` · livros ${counts.book || 0} / scripts ${counts.dialogue || 0}`
      : "";
    const scoreText = result.retrieval?.scoreMode ? ` · ${result.retrieval.scoreMode}` : "";
    const graphMetaText = graph.nodes ? ` · ${graph.nodes.length} nos / ${graph.edges?.length || 0} arestas` : "";
    const evidence = docs
      .slice(0, 3)
      .map((doc) => {
        const source = doc.sourceTitle || doc.sourceType || "texto";
        const chapter = doc.chapterTitle ? ` / ${doc.chapterTitle}` : "";
        return `<li>${escapeHtml(sourceLabel(doc.sourceType))} · ${escapeHtml(source + chapter)} · ${escapeHtml(scoreLabel(doc))}</li>`;
      })
      .join("");
    card.innerHTML = `
      <div class="compare-card-head">
        <h3>${labels[mode]}</h3>
        <span class="mode-badge">${badges[mode]}</span>
      </div>
      <div class="compare-meta">${escapeHtml(method)}${escapeHtml(scoreText)}${escapeHtml(sourceText)} · entidades: ${escapeHtml(entities)}${escapeHtml(graphMetaText)}</div>
      <div class="compare-answer">${renderAnswer(result.answer || "")}</div>
      <div class="method-note">${limits[mode]}</div>
      ${evidence ? `<ol class="mini-evidence">${evidence}</ol>` : `<p class="muted">Este modo nao recupera chunks textuais.</p>`}
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
    button.addEventListener("click", async () => {
      selectCypherExample(example.id);
      await runCypher();
    });
    cypherExamples.appendChild(button);
  }
  if (state.cypherExamples[0]) selectCypherExample(state.cypherExamples[0].id, { loadGraph: false });
}

function cypherExamplePrompt(example) {
  if (!example) return "";
  return [example.title, example.explain || example.lesson].filter(Boolean).join(". ");
}

function renderCypherLesson(example) {
  if (!example) {
    graphLessonTitle.textContent = "Cypher -> Subgrafo -> Leitura";
    graphLessonText.textContent = "Descreva uma pergunta, gere uma query revisavel, recupere estrutura e sintetize apenas o grafo retornado.";
    graphLessonGnn.textContent = "Use caminhos, vizinhos e pesos para discutir propagacao de informacao.";
    graphLessonVisual.textContent = "query auditavel";
    graphLessonCaption.textContent = "Queries com RETURN p alimentam tabela e grafo; queries escalares continuam tabulares.";
    return;
  }
  const visual = example.visual || {};
  graphLessonTitle.textContent = `Fluxo Graph-only: ${example.title || "Cypher"}`;
  graphLessonText.textContent = example.lesson || example.explain || "";
  graphLessonGnn.textContent = example.gnn || "A consulta revela estrutura local e propagacao entre entidades.";
  graphLessonVisual.textContent = visual.center ? `${visual.center} · ${visual.hops || 1}-hop` : "visualizacao livre";
  graphLessonCaption.textContent = visual.caption || "A descricao do exemplo alimenta a sintese Graph-only.";
}

function selectCypherExample(exampleId, options = {}) {
  const example = state.cypherExamples.find((item) => item.id === exampleId);
  if (!example) return null;
  state.activeCypherExample = example;
  cypherInput.value = example.query;
  cypherPromptInput.value = cypherExamplePrompt(example);
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

function renderCypherResult(result) {
  const graph = result.graph || { nodes: [], edges: [] };
  state.lastCypherResult = result;
  if (graph.nodes?.length) {
    renderGraph(graph, { meta: `Grafo da query · ${graph.nodes.length} nos · ${graph.edges?.length || 0} arestas` });
    graphSynthesis.innerHTML = `<strong>Leitura Graph-only</strong><p>Subgrafo da query pronto para sintese estrutural.</p>`;
  } else {
    synthesizeGraphButton.disabled = false;
    const statusText = result.graphStatus === "empty"
      ? "Consulta sem linhas e sem subgrafo renderizavel."
      : "Esta query retornou apenas valores escalares; use RETURN p ou RETURN a, r, b para visualizar subgrafo.";
    graphSynthesis.innerHTML = `<strong>Leitura Graph-only</strong><p>${escapeHtml(statusText)}</p>`;
  }
  renderTable(result.columns || [], result.rows || [], cypherResults, result.graphStatus);
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
    renderCypherResult(result);
    return result;
  } catch (error) {
    state.lastCypherResult = null;
    cypherResults.innerHTML = `<p class="error-text">${escapeHtml(error.message)}</p>`;
    return null;
  } finally {
    runCypherButton.disabled = false;
  }
}

async function generateCypher() {
  const question = cypherPromptInput.value.trim();
  if (!question) {
    cypherGenerationStatus.textContent = "Descreva a consulta antes de gerar.";
    return;
  }
  generateCypherButton.disabled = true;
  cypherGenerationStatus.textContent = "Ollama gerando Cypher read-only...";
  try {
    const result = await getJson("/api/cypher/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, model: modelInput.value || null }),
    });
    cypherInput.value = result.query || "";
    state.activeCypherExample = null;
    document.querySelectorAll(".query-item").forEach((item) => item.classList.remove("active"));
    renderCypherLesson(null);
    const warnings = (result.warnings || []).length ? ` Alertas: ${(result.warnings || []).join(" ")}` : "";
    cypherGenerationStatus.textContent = `${result.explanation || "Query gerada para revisao."}${warnings}`;
  } catch (error) {
    cypherGenerationStatus.textContent = `Erro ao gerar Cypher: ${error.message}`;
  } finally {
    generateCypherButton.disabled = false;
  }
}

async function synthesizeGraph() {
  const result = state.lastCypherResult || { query: cypherInput.value, rows: [], graph: state.lastGraph || {} };
  synthesizeGraphButton.disabled = true;
  graphSynthesis.innerHTML = `<strong>Leitura Graph-only</strong><p>Ollama sintetizando estrutura...</p>`;
  try {
    const payload = await getJson("/api/cypher/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: cypherPromptInput.value.trim() || questionInput.value.trim() || "Explique o subgrafo recuperado.",
        query: result.query || cypherInput.value,
        rows: result.rows || [],
        graph: result.graph || state.lastGraph || {},
        model: modelInput.value || null,
      }),
    });
    graphSynthesis.innerHTML = `<strong>Leitura Graph-only · ${escapeHtml(payload.model || "Ollama")}</strong><p>${renderAnswer(payload.answer || "")}</p>`;
    llmStatus.textContent = `${payload.model || modelInput.value || "Ollama"} · ${payload.llmStatus}`;
  } catch (error) {
    graphSynthesis.innerHTML = `<strong>Leitura Graph-only</strong><p class="error-text">${escapeHtml(error.message)}</p>`;
  } finally {
    synthesizeGraphButton.disabled = false;
  }
}

async function runCypherAndSynthesize() {
  const result = await runCypher();
  if (result) {
    await synthesizeGraph();
  }
}

function renderTable(columns, rows, target, graphStatus = "scalar-only") {
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
  const note = graphStatus === "rendered"
    ? `${rows.length} linhas retornadas. O grafo da direita veio da propria query.`
    : `${rows.length} linhas retornadas. Resultado tabular sem subgrafo renderizavel.`;
  target.innerHTML = `<p class="table-note">${escapeHtml(note)}</p><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
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

function lectureModeName(mode) {
  if (mode === "rag") return "RAG textual";
  if (mode === "graph") return "Graph-only";
  if (mode === "hybrid") return "GraphRAG";
  if (mode === "compare") return "Comparacao";
  return "Overview";
}

function renderLectureArchitecture(architecture) {
  if (!architecture) {
    lectureArchitecturePanel.hidden = true;
    return;
  }
  lectureArchitecturePanel.hidden = false;
  lectureArchitectureName.textContent = architecture.name || "Arquitetura";
  lectureArchitectureFlow.innerHTML = (architecture.flow || [])
    .map((item) => `<span>${escapeHtml(item)}</span>`)
    .join("");
  lectureArchitectureTakeaway.textContent = architecture.takeaway || "";
  lectureArchitectureRisk.textContent = architecture.risk ? `Risco: ${architecture.risk}` : "";
}

function renderQuiz(quiz) {
  quizQuestion.textContent = quiz?.question || "-";
  quizAnswer.textContent = "";
  quizOptions.innerHTML = "";
  const options = quiz?.options || [];
  for (const [index, option] of options.entries()) {
    const button = document.createElement("button");
    button.className = "quiz-option";
    button.type = "button";
    button.dataset.correct = option.correct ? "true" : "false";
    button.innerHTML = `<span>${String.fromCharCode(65 + index)}</span><strong>${escapeHtml(option.label || "")}</strong>`;
    button.addEventListener("click", () => {
      quizOptions.querySelectorAll(".quiz-option").forEach((item) => {
        item.classList.remove("selected", "correct", "incorrect");
      });
      button.classList.add("selected", option.correct ? "correct" : "incorrect");
      quizAnswer.textContent = option.explanation || quiz.answer || "";
    });
    quizOptions.appendChild(button);
  }
  revealQuizButton.disabled = !quiz?.answer;
  quizAnswer.dataset.answer = quiz?.answer || "";
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
  lectureQuestion.textContent = step.question || "Pergunta livre";
  lectureModePill.textContent = lectureModeName(step.mode);
  lectureContrasts.innerHTML = (step.methodContrasts || [])
    .map(
      (item) => `
        <article class="method-contrast">
          <strong>${escapeHtml(item.label || "")}</strong>
          <p>${escapeHtml(item.text || "")}</p>
        </article>
      `,
    )
    .join("");
  lecturePoints.innerHTML = (step.talkingPoints || []).map((point) => `<li>${escapeHtml(point)}</li>`).join("");
  lectureDemoAction.textContent = step.demoAction || "A etapa atual define pergunta, modo e parametros do workspace.";
  renderLectureArchitecture(step.architecture);
  lectureSpeakerNotes.innerHTML = (step.speakerNotes || [])
    .map((note) => `<li>${escapeHtml(note)}</li>`)
    .join("");
  renderQuiz(step.quiz);
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
  graphSvg.addEventListener("click", (event) => {
    if (event.target === graphSvg) {
      clearGraphHighlight();
      setGraphDetail("Clique em um no ou aresta para ver detalhes.");
    }
  });
  graphSvg.addEventListener("pointerdown", (event) => {
    if (!state.graphViewBox) return;
    if (event.target.closest?.(".node-group") || event.target.closest?.(".edge")) return;
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
  renderGraphRagStrategies();
  setupGraphPanZoom();
  setView(activeView());
  await loadGraphRagStrategies();
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
strategyCompareButton.addEventListener("click", compareGraphRagStrategies);
compareButton.addEventListener("click", compareQuestion);
compareViewButton.addEventListener("click", compareQuestion);
runCypherButton.addEventListener("click", runCypher);
generateCypherButton.addEventListener("click", generateCypher);
synthesizeGraphButton.addEventListener("click", runCypherAndSynthesize);
copyCypherButton.addEventListener("click", () => copyText(cypherInput.value));
copyStarterCypher.addEventListener("click", () => copyText(state.cypherExamples[0]?.query || ""));
cypherInput.addEventListener("input", () => {
  state.activeCypherExample = null;
  document.querySelectorAll(".query-item").forEach((item) => item.classList.remove("active"));
  renderCypherLesson(null);
});
hopsInput.addEventListener("change", loadGraph);
graphRagStrategyInput.addEventListener("change", () => {
  setGraphRagStrategy(graphRagStrategyInput.value);
});
centerInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") loadGraph();
});
zoomOutButton.addEventListener("click", () => zoomGraph(1.18));
zoomInButton.addEventListener("click", () => zoomGraph(0.85));
zoomResetButton.addEventListener("click", fitGraph);
zoomFitButton.addEventListener("click", fitGraph);
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
  quizOptions.querySelectorAll(".quiz-option").forEach((item) => {
    item.classList.toggle("correct", item.dataset.correct === "true");
    item.classList.remove("incorrect", "selected");
  });
});
applyLectureButton.addEventListener("click", applyLectureStep);

boot();
