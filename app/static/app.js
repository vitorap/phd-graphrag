const graphSvg = document.querySelector("#graphSvg");
const questionInput = document.querySelector("#questionInput");
const hopsInput = document.querySelector("#hopsInput");
const modeInput = document.querySelector("#modeInput");
const centerInput = document.querySelector("#centerInput");
const llmInput = document.querySelector("#llmInput");
const graphButton = document.querySelector("#graphButton");
const askButton = document.querySelector("#askButton");
const graphMeta = document.querySelector("#graphMeta");
const answerBox = document.querySelector("#answerBox");
const contextBox = document.querySelector("#contextBox");
const entitiesList = document.querySelector("#entitiesList");
const topList = document.querySelector("#topList");
const statsStrip = document.querySelector("#statsStrip");
const llmStatus = document.querySelector("#llmStatus");

const COLORS = {
  character: "#1f6b4b",
  weapon: "#8d3d3a",
  place: "#bd8a2b",
  language: "#3c5f94",
  race: "#6f5aa8",
  entity: "#64736b",
};

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status}: ${text}`);
  }
  return response.json();
}

function setLoading(message) {
  graphMeta.textContent = message;
  graphSvg.innerHTML = "";
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

function renderStats(stats) {
  statsStrip.innerHTML = `
    <span>Entidades: ${stats.entities}</span>
    <span>Relacoes: ${stats.relationships}</span>
    <span>Personagens: ${stats.characters}</span>
  `;
  topList.innerHTML = "";
  for (const item of stats.topCharacters || []) {
    const row = document.createElement("li");
    const pagerank = Number(item.pagerank || 0).toFixed(3);
    row.textContent = `${item.name} · PR ${pagerank}`;
    topList.appendChild(row);
  }
}

function edgeClass(edge) {
  if (edge.type === "ENEMY_OF") return "edge semantic enemy";
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
  const pad = 54;
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

function renderGraph(graph) {
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  graphSvg.innerHTML = "";
  const width = graphSvg.clientWidth || 800;
  const height = graphSvg.clientHeight || 620;
  graphSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  if (nodes.length === 0) {
    graphMeta.textContent = "Grafo vazio. Rode make seed.";
    return;
  }

  const scale = scaleFactory(nodes, width, height);
  const byId = new Map(nodes.map((node) => [node.id, node]));
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
    title.textContent = `${edge.sourceName} -[${edge.type}, peso=${edge.weight}]-> ${edge.targetName}`;
    line.appendChild(title);
    edgeLayer.appendChild(line);
  }

  for (const node of nodes) {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const x = scale.x(node.x || 0);
    const y = scale.y(node.y || 0);
    const color = COLORS[node.group] || COLORS.entity;
    group.setAttribute("transform", `translate(${x}, ${y})`);

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("r", Math.max(7, Number(node.size || 10)));
    circle.setAttribute("fill", color);
    circle.setAttribute("class", "node");
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${node.name} · ${node.kind || "Entity"}`;
    circle.appendChild(title);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "node-label");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("y", -Math.max(11, Number(node.size || 10)) - 5);
    label.textContent = node.name;

    group.append(circle, label);
    group.addEventListener("click", () => {
      centerInput.value = node.name;
      loadGraph();
    });
    nodeLayer.appendChild(group);
  }

  graphMeta.textContent = `${nodes.length} nos · ${edges.length} arestas`;
}

function renderAnswer(text) {
  const escaped = String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
  return escaped.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

async function loadGraph() {
  const center = encodeURIComponent(centerInput.value.trim());
  const hops = encodeURIComponent(hopsInput.value);
  setLoading("Carregando subgrafo...");
  try {
    const graph = await getJson(`/api/graph?center=${center}&hops=${hops}&limit=180`);
    renderGraph(graph);
  } catch (error) {
    graphMeta.textContent = `Erro: ${error.message}`;
  }
}

async function askQuestion() {
  answerBox.textContent = "";
  contextBox.textContent = "";
  llmStatus.textContent = "Consultando grafo e Ollama...";
  askButton.disabled = true;
  try {
    const payload = {
      question: questionInput.value,
      hops: Number(hopsInput.value),
      mode: modeInput.value,
      use_llm: llmInput.checked,
    };
    const result = await getJson("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setTags(result.entities || []);
    answerBox.innerHTML = renderAnswer(result.answer || "");
    contextBox.textContent = result.context || "";
    llmStatus.textContent = `${result.model} · ${result.llmStatus}`;
    if (result.graph && result.graph.nodes) {
      renderGraph(result.graph);
    }
  } catch (error) {
    llmStatus.textContent = `Erro: ${error.message}`;
  } finally {
    askButton.disabled = false;
  }
}

async function boot() {
  setTags([]);
  try {
    const stats = await getJson("/api/stats");
    renderStats(stats);
  } catch (error) {
    statsStrip.innerHTML = `<span>Neo4j indisponivel</span>`;
  }
  await loadGraph();
}

graphButton.addEventListener("click", loadGraph);
askButton.addEventListener("click", askQuestion);
hopsInput.addEventListener("change", loadGraph);
centerInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") loadGraph();
});

boot();
