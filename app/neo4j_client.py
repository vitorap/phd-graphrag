from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import networkx as nx
from neo4j import GraphDatabase
from neo4j.graph import Node, Path, Relationship

from app.config import settings


SEMANTIC_REL_TYPES = [
    "FRIEND_OF",
    "ENEMY_OF",
    "HAS_WEAPON",
    "INHABITANT",
    "SPEAKS",
    "SPOKEN_BY",
    "SPOKEN_IN",
    "ADOPTED",
    "ADOPTED_BY",
    "NEPHEW_OF",
    "UNCLE_OF",
    "HAS_RACE",
    "HAS_NAME",
    "HAS_FAMILY_NAME",
    "LOCATED_IN",
    "COUSIN_OF",
    "CO_OCCURS_WITH",
    "PREDICTED_LINK",
]

FORBIDDEN_CYPHER = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s+dbms|CALL\s+apoc\.periodic)\b",
    re.IGNORECASE,
)


class Neo4jClient:
    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.driver = GraphDatabase.driver(
            uri or settings.neo4j_uri,
            auth=(username or settings.neo4j_username, password or settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def ping(self) -> bool:
        with self.driver.session() as session:
            session.run("RETURN 1").consume()
        return True

    def stats(self) -> dict[str, Any]:
        with self.driver.session() as session:
            counts = session.run(
                """
                MATCH (n:Entity)
                WITH count(n) AS entities
                MATCH ()-[r]->()
                WITH entities, count(r) AS relationships
                MATCH (c:Character)
                WITH entities, relationships, count(c) AS characters
                MATCH (d:RetrievalDocument)
                WITH entities, relationships, characters, count(d) AS retrievalDocuments
                MATCH (t:TextChunk)
                WITH entities, relationships, characters, retrievalDocuments, count(t) AS textChunks
                MATCH (l:DialogueLine)
                WITH entities, relationships, characters, retrievalDocuments, textChunks, count(l) AS dialogueLines
                MATCH (ch:Chapter)
                WITH entities, relationships, characters, retrievalDocuments, textChunks, dialogueLines, count(ch) AS chapters
                MATCH (b:Book)
                WITH entities, relationships, characters, retrievalDocuments, textChunks, dialogueLines, chapters, count(b) AS books
                MATCH (m:Movie)
                RETURN entities, relationships, characters, retrievalDocuments, textChunks,
                       dialogueLines, chapters, books, count(m) AS movies
                """
            ).single()
            rels = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(r) AS count
                ORDER BY count DESC, type
                """
            )
            top = session.run(
                """
                MATCH (c:Character)
                RETURN c.name AS name, c.race AS race, c.pagerank AS pagerank,
                       c.weightedDegree AS weightedDegree, c.community AS community
                ORDER BY coalesce(c.pagerank, 0) DESC
                LIMIT 12
                """
            )
            return {
                "entities": counts["entities"] if counts else 0,
                "relationships": counts["relationships"] if counts else 0,
                "characters": counts["characters"] if counts else 0,
                "retrievalDocuments": counts["retrievalDocuments"] if counts else 0,
                "textChunks": counts["textChunks"] if counts else 0,
                "dialogueLines": counts["dialogueLines"] if counts else 0,
                "chapters": counts["chapters"] if counts else 0,
                "books": counts["books"] if counts else 0,
                "movies": counts["movies"] if counts else 0,
                "relationshipTypes": [dict(row) for row in rels],
                "topCharacters": [dict(row) for row in top],
            }

    def list_entities(self) -> list[dict[str, Any]]:
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (e:Entity)
                RETURN e.name AS name, e.aliases AS aliases, labels(e) AS labels,
                       e.kind AS kind, e.pagerank AS pagerank
                ORDER BY coalesce(e.pagerank, 0) DESC, e.name
                """
            )
            return [dict(row) for row in rows]

    def top_characters(self, limit: int = 12) -> list[dict[str, Any]]:
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (c:Character)
                RETURN c.name AS name, c.race AS race, c.gender AS gender,
                       c.pagerank AS pagerank, c.weightedDegree AS weightedDegree,
                       c.community AS community
                ORDER BY coalesce(c.pagerank, 0) DESC
                LIMIT $limit
                """,
                limit=limit,
            )
            return [dict(row) for row in rows]

    def graph(self, center: str | None = None, hops: int = 1, limit: int = 160) -> dict[str, Any]:
        if center:
            graph = self.subgraph_for_seeds([center], hops=hops, limit=limit)
            if graph["nodes"]:
                return graph
        return self.global_graph(limit=limit)

    def global_graph(self, limit: int = 160) -> dict[str, Any]:
        with self.driver.session() as session:
            row = session.run(
                """
                MATCH (a:Entity)-[r]->(b:Entity)
                WITH a, r, b
                ORDER BY CASE type(r)
                    WHEN 'INTERACTS_WITH' THEN coalesce(r.weight, 1)
                    WHEN 'CO_OCCURS_WITH' THEN coalesce(r.weight, 1) * 0.9
                    WHEN 'PREDICTED_LINK' THEN coalesce(r.confidence, 0)
                    ELSE 1
                END DESC
                LIMIT $limit
                WITH collect(DISTINCT r) AS rels, collect(DISTINCT a) + collect(DISTINCT b) AS rawNodes
                UNWIND rawNodes AS n
                WITH collect(DISTINCT n) AS nodes, rels
                RETURN
                  [n IN nodes | {
                    id: elementId(n),
                    name: n.name,
                    labels: labels(n),
                    kind: n.kind,
                    race: n.race,
                    gender: n.gender,
                    pagerank: n.pagerank,
                    community: n.community,
                    weightedDegree: n.weightedDegree
                  }] AS nodes,
                  [r IN rels | {
                    id: elementId(r),
                    source: elementId(startNode(r)),
                    target: elementId(endNode(r)),
                    sourceName: startNode(r).name,
                    targetName: endNode(r).name,
                    type: type(r),
                    weight: coalesce(r.weight, 1),
                    confidence: r.confidence,
                    method: r.method,
                    sourceDataset: r.sourceDataset
                  }] AS edges
                """,
                limit=limit,
            ).single()
        return self._with_layout(row["nodes"] if row else [], row["edges"] if row else [])

    def subgraph_for_seeds(self, seeds: list[str], hops: int = 2, limit: int = 180) -> dict[str, Any]:
        hops = max(1, min(int(hops), 4))
        seeds = [seed for seed in seeds if seed]
        if not seeds:
            return self.global_graph(limit=limit)
        with self.driver.session() as session:
            row = session.run(
                f"""
                MATCH (seed:Entity)
                WHERE seed.name IN $seeds
                MATCH p = (seed)-[*1..{hops}]-(n:Entity)
                WHERE all(rel IN relationships(p)
                          WHERE type(rel) <> 'PREDICTED_LINK' OR coalesce(rel.confidence, 0) >= 0.25)
                WITH p, n
                ORDER BY length(p), coalesce(n.pagerank, 0) DESC
                LIMIT $limit
                WITH collect(p) AS paths
                UNWIND paths AS p
                UNWIND nodes(p) AS n
                WITH paths, collect(DISTINCT n) AS nodes
                UNWIND paths AS p
                UNWIND relationships(p) AS r
                WITH nodes, collect(DISTINCT r) AS rels
                RETURN
                  [n IN nodes | {{
                    id: elementId(n),
                    name: n.name,
                    labels: labels(n),
                    kind: n.kind,
                    race: n.race,
                    gender: n.gender,
                    pagerank: n.pagerank,
                    community: n.community,
                    weightedDegree: n.weightedDegree
                  }}] AS nodes,
                  [r IN rels | {{
                    id: elementId(r),
                    source: elementId(startNode(r)),
                    target: elementId(endNode(r)),
                    sourceName: startNode(r).name,
                    targetName: endNode(r).name,
                    type: type(r),
                    weight: coalesce(r.weight, 1),
                    confidence: r.confidence,
                    method: r.method,
                    sourceDataset: r.sourceDataset
                  }}] AS edges
                """,
                seeds=seeds,
                limit=limit,
            ).single()
        return self._with_layout(row["nodes"] if row else [], row["edges"] if row else [])

    def community_subgraph_for_seeds(self, seeds: list[str], limit: int = 180) -> dict[str, Any]:
        seeds = [seed for seed in seeds if seed]
        if not seeds:
            return self.global_graph(limit=limit)
        with self.driver.session() as session:
            row = session.run(
                """
                MATCH (seed:Character)
                WHERE seed.name IN $seeds AND seed.community IS NOT NULL
                WITH collect(DISTINCT seed.community) AS communities, collect(seed) AS seedNodes
                MATCH (a:Character)-[r]-(b:Character)
                WHERE a.community IN communities
                  AND b.community = a.community
                  AND type(r) IN ['INTERACTS_WITH', 'CO_OCCURS_WITH', 'FRIEND_OF', 'ENEMY_OF', 'PREDICTED_LINK']
                  AND (type(r) <> 'PREDICTED_LINK' OR coalesce(r.confidence, 0) >= 0.25)
                WITH seedNodes, a, r, b
                ORDER BY coalesce(r.weight, r.confidence, 1) DESC, coalesce(a.pagerank, 0) DESC
                LIMIT $limit
                WITH seedNodes, collect(DISTINCT r) AS rels, collect(DISTINCT a) + collect(DISTINCT b) AS communityNodes
                WITH rels, seedNodes + communityNodes AS rawNodes
                UNWIND rawNodes AS n
                WITH collect(DISTINCT n) AS nodes, rels
                RETURN
                  [n IN nodes | {
                    id: elementId(n),
                    name: n.name,
                    labels: labels(n),
                    kind: n.kind,
                    race: n.race,
                    gender: n.gender,
                    pagerank: n.pagerank,
                    community: n.community,
                    weightedDegree: n.weightedDegree
                  }] AS nodes,
                  [r IN rels | {
                    id: elementId(r),
                    source: elementId(startNode(r)),
                    target: elementId(endNode(r)),
                    sourceName: startNode(r).name,
                    targetName: endNode(r).name,
                    type: type(r),
                    weight: coalesce(r.weight, 1),
                    confidence: r.confidence,
                    method: r.method,
                    sourceDataset: r.sourceDataset
                  }] AS edges
                """,
                seeds=seeds,
                limit=limit,
            ).single()
        return self._with_layout(row["nodes"] if row else [], row["edges"] if row else [])

    def shortest_path(self, source: str, target: str, max_depth: int = 5) -> list[str]:
        max_depth = max(1, min(int(max_depth), 6))
        with self.driver.session() as session:
            row = session.run(
                f"""
                MATCH (a:Entity {{name: $source}})
                MATCH (b:Entity {{name: $target}})
                MATCH p = shortestPath((a)-[*..{max_depth}]-(b))
                WHERE all(rel IN relationships(p)
                          WHERE type(rel) <> 'PREDICTED_LINK' OR coalesce(rel.confidence, 0) >= 0.25)
                RETURN [n IN nodes(p) | n.name] AS names
                """,
                source=source,
                target=target,
            ).single()
            return [name for name in (row["names"] if row else []) if name]

    def top_neighbors(self, name: str, limit: int = 10) -> list[dict[str, Any]]:
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (:Entity {name: $name})-[r]-(n:Entity)
                RETURN n.name AS name, type(r) AS relation, coalesce(r.weight, r.confidence, 1) AS weight,
                       n.kind AS kind, n.race AS race, n.pagerank AS pagerank
                ORDER BY weight DESC, coalesce(n.pagerank, 0) DESC
                LIMIT $limit
                """,
                name=name,
                limit=limit,
            )
            return [dict(row) for row in rows]

    def retrieval_documents(self, limit: int = 6000) -> list[dict[str, Any]]:
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (d:RetrievalDocument)
                OPTIONAL MATCH (d)-[:MENTIONS]->(e:Entity)
                RETURN d.id AS id,
                       labels(d) AS labels,
                       d.text AS text,
                       d.sourceType AS sourceType,
                       d.sourceTitle AS sourceTitle,
                       d.chapterTitle AS chapterTitle,
                       d.sequence AS sequence,
                       d.speaker AS speaker,
                       d.lineNumber AS lineNumber,
                       collect(DISTINCT e.name) AS mentions
                ORDER BY d.sequence
                LIMIT $limit
                """,
                limit=limit,
            )
            return [dict(row) for row in rows]

    def documents_for_entities(self, names: list[str], limit: int = 12) -> list[dict[str, Any]]:
        if not names:
            return []
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (d:RetrievalDocument)-[:MENTIONS]->(e:Entity)
                WHERE e.name IN $names
                WITH d, count(DISTINCT e) AS entityHits
                OPTIONAL MATCH (d)-[:MENTIONS]->(m:Entity)
                RETURN d.id AS id,
                       labels(d) AS labels,
                       d.text AS text,
                       d.sourceType AS sourceType,
                       d.sourceTitle AS sourceTitle,
                       d.chapterTitle AS chapterTitle,
                       d.sequence AS sequence,
                       d.speaker AS speaker,
                       d.lineNumber AS lineNumber,
                       entityHits,
                       collect(DISTINCT m.name) AS mentions
                ORDER BY entityHits DESC, d.sequence
                LIMIT $limit
                """,
                names=names,
                limit=limit,
            )
            return [dict(row) for row in rows]

    def run_readonly_cypher(self, query: str, limit: int = 100) -> dict[str, Any]:
        wrapped = self.prepare_readonly_cypher(query, limit=limit)
        with self.driver.session(default_access_mode="READ") as session:
            result = session.run(wrapped)
            keys = list(result.keys())
            raw_rows = [{key: row.get(key) for key in keys} for row in result]

        graph_nodes: dict[str, dict[str, Any]] = {}
        graph_edges: dict[str, dict[str, Any]] = {}
        for row in raw_rows:
            collect_graph_values(row, graph_nodes, graph_edges)

        graph = self._with_layout(list(graph_nodes.values()), list(graph_edges.values()))
        if graph["nodes"]:
            graph_status = "rendered"
        elif raw_rows:
            graph_status = "scalar-only"
        else:
            graph_status = "empty"

        rows = [{key: json_safe(value) for key, value in row.items()} for row in raw_rows]
        return {"columns": keys, "rows": rows, "query": wrapped, "graph": graph, "graphStatus": graph_status}

    @staticmethod
    def prepare_readonly_cypher(query: str, limit: int = 100) -> str:
        query = query.strip().removesuffix(";").strip()
        if not query:
            raise ValueError("Cypher vazio")
        if ";" in query:
            raise ValueError("Use uma unica consulta Cypher por vez")
        if FORBIDDEN_CYPHER.search(query):
            raise ValueError("A demo aceita apenas consultas read-only")
        if not re.match(r"^(MATCH|OPTIONAL\s+MATCH|WITH|RETURN|UNWIND|CALL\s+db\.)\b", query, re.IGNORECASE):
            raise ValueError("Consulta precisa comecar com MATCH, WITH, RETURN, UNWIND ou CALL db.*")

        wrapped = f"{query}\nLIMIT {int(max(1, min(limit, 300)))}"
        if re.search(r"\bLIMIT\b", query, re.IGNORECASE):
            wrapped = query
        return wrapped

    def _with_layout(self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
        by_id = {node["id"]: node for node in nodes}
        graph = nx.Graph()
        for node in nodes:
            graph.add_node(node["id"])
        for edge in edges:
            if edge["source"] in by_id and edge["target"] in by_id:
                graph.add_edge(edge["source"], edge["target"], weight=float(edge.get("weight") or 1))

        if graph.number_of_nodes() == 0:
            return {"nodes": [], "edges": []}
        if graph.number_of_nodes() == 1:
            positions = {next(iter(graph.nodes)): (0.0, 0.0)}
        else:
            positions = nx.spring_layout(graph, seed=42, weight="weight", iterations=80)

        for node in nodes:
            x, y = positions.get(node["id"], (0.0, 0.0))
            node["x"] = float(x)
            node["y"] = float(y)
            node["size"] = self._node_size(node)
            node["group"] = self._node_group(node)
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _node_size(node: dict[str, Any]) -> float:
        weighted_degree = node.get("weightedDegree") or 0
        pagerank = node.get("pagerank") or 0
        return min(28.0, 8.0 + (weighted_degree ** 0.35) + pagerank * 150)

    @staticmethod
    def _node_group(node: dict[str, Any]) -> str:
        labels = set(node.get("labels") or [])
        if "Character" in labels:
            return "character"
        if "Weapon" in labels:
            return "weapon"
        if "Place" in labels:
            return "place"
        if "Language" in labels:
            return "language"
        if "Race" in labels:
            return "race"
        return "entity"


def json_safe(value: Any) -> Any:
    if isinstance(value, Node):
        return {"id": element_id(value), "labels": list(value.labels), **dict(value)}
    if isinstance(value, Relationship):
        return {
            "id": element_id(value),
            "type": value.type,
            "source": element_id(value.start_node),
            "target": element_id(value.end_node),
            **dict(value),
        }
    if isinstance(value, Path):
        return {
            "nodes": [json_safe(node) for node in value.nodes],
            "relationships": [json_safe(rel) for rel in value.relationships],
        }
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def collect_graph_values(value: Any, nodes: dict[str, dict[str, Any]], edges: dict[str, dict[str, Any]]) -> None:
    if isinstance(value, Node):
        nodes[element_id(value)] = graph_node(value)
        return
    if isinstance(value, Relationship):
        nodes[element_id(value.start_node)] = graph_node(value.start_node)
        nodes[element_id(value.end_node)] = graph_node(value.end_node)
        edges[element_id(value)] = graph_edge(value)
        return
    if isinstance(value, Path):
        for node in value.nodes:
            nodes[element_id(node)] = graph_node(node)
        for rel in value.relationships:
            nodes[element_id(rel.start_node)] = graph_node(rel.start_node)
            nodes[element_id(rel.end_node)] = graph_node(rel.end_node)
            edges[element_id(rel)] = graph_edge(rel)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            collect_graph_values(item, nodes, edges)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            collect_graph_values(item, nodes, edges)


def graph_node(value: Node) -> dict[str, Any]:
    props = dict(value)
    labels = list(value.labels)
    return {
        "id": element_id(value),
        "name": node_display_name(value),
        "labels": labels,
        "kind": props.get("kind") or next((label for label in labels if label != "Entity"), None),
        "race": props.get("race"),
        "gender": props.get("gender"),
        "pagerank": props.get("pagerank"),
        "community": props.get("community"),
        "weightedDegree": props.get("weightedDegree"),
        "sourceTitle": props.get("sourceTitle"),
        "chapterTitle": props.get("chapterTitle"),
        "sourceType": props.get("sourceType"),
        "speaker": props.get("speaker"),
    }


def graph_edge(value: Relationship) -> dict[str, Any]:
    props = dict(value)
    weight = props.get("weight")
    confidence = props.get("confidence")
    return {
        "id": element_id(value),
        "source": element_id(value.start_node),
        "target": element_id(value.end_node),
        "sourceName": node_display_name(value.start_node),
        "targetName": node_display_name(value.end_node),
        "type": value.type,
        "weight": weight if weight is not None else (confidence if confidence is not None else 1),
        "confidence": confidence,
        "method": props.get("method"),
        "sourceDataset": props.get("sourceDataset"),
    }


def node_display_name(value: Node) -> str:
    props = dict(value)
    for key in ("name", "title", "sourceTitle", "chapterTitle", "speaker", "id"):
        if props.get(key):
            return str(props[key])
    labels = list(value.labels)
    if labels:
        return f"{labels[0]} {element_id(value)}"
    return element_id(value)


def element_id(value: Node | Relationship) -> str:
    return getattr(value, "element_id", str(value.id))
