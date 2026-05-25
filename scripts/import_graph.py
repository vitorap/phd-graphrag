from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import networkx as nx
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

LOTR_CSV = RAW_DIR / "lotr.csv"
PROPERTIES_CSV = RAW_DIR / "lotr_properties.csv"
OWL_XML = RAW_DIR / "LOTRORDFXML.owl"

NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "graphrag-lotr"

SAFE_LABELS = {
    "Character",
    "Weapon",
    "Place",
    "Language",
    "Race",
    "Name",
    "Class",
    "OntologyEntity",
}

WEAPON_TYPES = {"Weapon", "Sword", "Axe", "Bow", "Ring", "Staff"}
CHARACTER_TYPES = {
    "Character",
    "Hobbit",
    "Human",
    "Elf",
    "Dwarf",
    "Maiar",
    "Fallohide",
    "Harfoot",
    "Stoor",
    "Noldor",
    "Teleri",
    "Vanyar",
}

RELATION_MAP = {
    "adopt": "ADOPTED",
    "adopted": "ADOPTED",
    "adoptedBy": "ADOPTED_BY",
    "cousinOf": "COUSIN_OF",
    "enemyOf": "ENEMY_OF",
    "friendOf": "FRIEND_OF",
    "hasWeapon": "HAS_WEAPON",
    "inhabitant": "INHABITANT",
    "nephewOf": "NEPHEW_OF",
    "speaks": "SPEAKS",
    "spokenBy": "SPOKEN_BY",
    "spokenIn": "SPOKEN_IN",
    "uncleOf": "UNCLE_OF",
    "name": "HAS_NAME",
    "family_name": "HAS_FAMILY_NAME",
    "Location": "LOCATED_IN",
}

DATA_PROPERTY_MAP = {
    "age": "age",
    "member": "member",
    "hasEtimology": "etymology",
    "isWrittenAs": "writtenAs",
    "means": "means",
    "dateOfBirth": "dateOfBirth",
    "dateOfDeath": "dateOfDeath",
    "sameAs": "sameAs",
    "comment": "description",
    "label": "label",
}

ALIASES = {
    "Sam": "Samwise",
    "Merry": "Meriadoc",
    "Pippin": "Peregrin",
    "Elessar": "Aragorn",
    "Strider": "Aragorn",
    "Smeagol": "Gollum",
    "Sméagol": "Gollum",
}


@dataclass
class EntityDraft:
    name: str
    labels: set[str] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    props: dict[str, Any] = field(default_factory=dict)


def getenv(name: str, default: str) -> str:
    import os

    return os.getenv(name, default)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value.strip()


def canonical_name(value: str) -> str:
    cleaned = value.strip()
    ascii_key = normalize_text(cleaned)
    return ALIASES.get(cleaned) or ALIASES.get(ascii_key) or cleaned


def local_name(uri: Any) -> str:
    text = unquote(str(uri))
    if "#" in text:
        text = text.rsplit("#", 1)[1]
    else:
        text = text.rstrip("/").rsplit("/", 1)[-1]
    return text


def display_name(uri_or_name: Any) -> str:
    return canonical_name(local_name(uri_or_name).replace("_", " "))


def relation_type(predicate: URIRef) -> str | None:
    candidate = local_name(predicate)
    return RELATION_MAP.get(candidate)


def data_property_name(predicate: URIRef) -> str | None:
    candidate = local_name(predicate)
    return DATA_PROPERTY_MAP.get(candidate)


def safe_props(props: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in props.items():
        if value is None:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, set):
            value = sorted(value)
        cleaned[key] = value
    return cleaned


def add_entity(
    entities: dict[str, EntityDraft],
    name: str,
    labels: set[str] | None = None,
    source: str | None = None,
    aliases: set[str] | None = None,
    props: dict[str, Any] | None = None,
) -> EntityDraft:
    canonical = canonical_name(name)
    entity = entities.setdefault(canonical, EntityDraft(name=canonical))
    entity.labels.update(labels or set())
    if source:
        entity.sources.add(source)
    if aliases:
        entity.aliases.update(alias for alias in aliases if alias and alias != canonical)
    if name != canonical:
        entity.aliases.add(name)
    if props:
        for key, value in props.items():
            if value in (None, "", []):
                continue
            if key in entity.props and entity.props[key] != value:
                if isinstance(entity.props[key], list):
                    values = set(entity.props[key])
                else:
                    values = {entity.props[key]}
                if isinstance(value, list):
                    values.update(value)
                else:
                    values.add(value)
                entity.props[key] = sorted(values)
            else:
                entity.props[key] = value
    return entity


def ensure_files() -> None:
    missing = [path for path in [LOTR_CSV, PROPERTIES_CSV, OWL_XML] if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise RuntimeError(f"arquivos ausentes: {missing_text}. Rode `make data`.")


def read_interactions(entities: dict[str, EntityDraft]) -> list[dict[str, Any]]:
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    graph = nx.Graph()

    with LOTR_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            raw_a, raw_b, raw_sentence = row[0].strip(), row[1].strip(), row[2].strip()
            source = canonical_name(raw_a)
            target = canonical_name(raw_b)
            if source == target:
                continue

            add_entity(entities, source, {"Character"}, "raphtory", aliases={raw_a})
            add_entity(entities, target, {"Character"}, "raphtory", aliases={raw_b})

            left, right = sorted([source, target])
            key = (left, right)
            sentence = int(raw_sentence) if raw_sentence.isdigit() else None
            edge = edges.setdefault(
                key,
                {
                    "source": left,
                    "target": right,
                    "type": "INTERACTS_WITH",
                    "props": {
                        "weight": 0,
                        "firstSentence": sentence,
                        "lastSentence": sentence,
                        "sourceDataset": "raphtory",
                    },
                },
            )
            edge["props"]["weight"] += 1
            if sentence is not None:
                first = edge["props"].get("firstSentence")
                last = edge["props"].get("lastSentence")
                edge["props"]["firstSentence"] = sentence if first is None else min(first, sentence)
                edge["props"]["lastSentence"] = sentence if last is None else max(last, sentence)

    for edge in edges.values():
        graph.add_edge(edge["source"], edge["target"], weight=edge["props"]["weight"])

    pagerank = weighted_pagerank(graph) if graph.number_of_nodes() else {}
    weighted_degree = dict(graph.degree(weight="weight"))
    degree = dict(graph.degree())
    communities = structural_communities(graph) if graph.number_of_edges() else {}

    for name, entity in entities.items():
        if name in graph:
            entity.props["degree"] = int(degree.get(name, 0))
            entity.props["weightedDegree"] = float(weighted_degree.get(name, 0.0))
            entity.props["pagerank"] = float(pagerank.get(name, 0.0))
            entity.props["community"] = int(communities.get(name, -1))

    return list(edges.values())


def weighted_pagerank(
    graph: nx.Graph,
    damping: float = 0.85,
    iterations: int = 60,
    tolerance: float = 1.0e-9,
) -> dict[str, float]:
    nodes = list(graph.nodes())
    if not nodes:
        return {}
    size = len(nodes)
    ranks = {node: 1.0 / size for node in nodes}
    strengths = {
        node: sum(float(data.get("weight", 1.0)) for _, data in graph[node].items())
        for node in nodes
    }

    for _ in range(iterations):
        updated = {node: (1.0 - damping) / size for node in nodes}
        dangling = sum(ranks[node] for node in nodes if strengths[node] == 0.0)
        dangling_share = damping * dangling / size
        for node in nodes:
            updated[node] += dangling_share
            strength = strengths[node]
            if strength == 0.0:
                continue
            for neighbor, data in graph[node].items():
                weight = float(data.get("weight", 1.0))
                updated[neighbor] += damping * ranks[node] * weight / strength

        delta = sum(abs(updated[node] - ranks[node]) for node in nodes)
        ranks = updated
        if delta < tolerance:
            break
    return ranks


def structural_communities(graph: nx.Graph) -> dict[str, int]:
    try:
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=1.15,
            seed=42,
        )
    except Exception:
        return label_propagation_communities(graph)

    labels: dict[str, int] = {}
    ordered = sorted(communities, key=lambda members: (-len(members), sorted(members)[0]))
    for idx, members in enumerate(ordered):
        for member in members:
            labels[member] = idx
    return labels


def label_propagation_communities(graph: nx.Graph, iterations: int = 20) -> dict[str, int]:
    labels = {node: idx for idx, node in enumerate(sorted(graph.nodes()))}
    ordered_nodes = sorted(graph.nodes(), key=lambda node: graph.degree(node, weight="weight"), reverse=True)

    for _ in range(iterations):
        changed = False
        for node in ordered_nodes:
            scores: dict[int, float] = defaultdict(float)
            for neighbor, data in graph[node].items():
                scores[labels[neighbor]] += float(data.get("weight", 1.0))
            if not scores:
                continue
            best_label = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
            if labels[node] != best_label:
                labels[node] = best_label
                changed = True
        if not changed:
            break

    remap = {label: idx for idx, label in enumerate(sorted(set(labels.values())))}
    return {node: remap[label] for node, label in labels.items()}


def read_properties(entities: dict[str, EntityDraft]) -> list[dict[str, Any]]:
    rels: list[dict[str, Any]] = []
    with PROPERTIES_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            raw_name, race, gender = row[0].strip(), row[1].strip(), row[2].strip()
            name = canonical_name(raw_name)
            race_name = race.strip().title()
            add_entity(
                entities,
                name,
                {"Character"},
                "raphtory_properties",
                aliases={raw_name},
                props={"race": race, "gender": gender},
            )
            add_entity(entities, race_name, {"Race"}, "raphtory_properties", props={"kind": "Race"})
            rels.append(
                {
                    "source": name,
                    "target": race_name,
                    "type": "HAS_RACE",
                    "props": {"sourceDataset": "raphtory_properties"},
                }
            )
    return rels


def infer_labels(types: set[str]) -> set[str]:
    labels: set[str] = {"OntologyEntity"}
    if types & CHARACTER_TYPES:
        labels.add("Character")
    if types & WEAPON_TYPES:
        labels.add("Weapon")
    if "Place" in types:
        labels.add("Place")
    if "Language" in types:
        labels.add("Language")
    if "Name" in types:
        labels.add("Name")
    return labels & SAFE_LABELS


def infer_kind(types: set[str], labels: set[str]) -> str:
    for label in ["Character", "Weapon", "Place", "Language", "Name"]:
        if label in labels:
            return label
    if types:
        return sorted(types)[0]
    return "Entity"


def read_ontology(entities: dict[str, EntityDraft]) -> list[dict[str, Any]]:
    rdf_graph = Graph()
    rdf_graph.parse(OWL_XML, format="xml")

    individuals = set(rdf_graph.subjects(RDF.type, OWL.NamedIndividual))
    rels: list[dict[str, Any]] = []

    for subject in individuals:
        name = display_name(subject)
        rdf_types = {
            local_name(obj)
            for obj in rdf_graph.objects(subject, RDF.type)
            if isinstance(obj, URIRef) and obj != OWL.NamedIndividual
        }
        labels = infer_labels(rdf_types)
        props: dict[str, Any] = {
            "kind": infer_kind(rdf_types, labels),
            "rdfTypes": sorted(rdf_types),
        }

        for predicate, obj in rdf_graph.predicate_objects(subject):
            if predicate in {RDF.type}:
                continue
            prop_name = data_property_name(predicate)
            if prop_name and isinstance(obj, Literal):
                props[prop_name] = str(obj)
            elif predicate in {RDFS.comment, RDFS.label} and isinstance(obj, Literal):
                props[data_property_name(predicate) or local_name(predicate)] = str(obj)

        add_entity(entities, name, labels, "lotro_owl", props=props)

    for subject in individuals:
        source = display_name(subject)
        for predicate, obj in rdf_graph.predicate_objects(subject):
            rel_type = relation_type(predicate)
            if not rel_type or not isinstance(obj, URIRef) or obj not in individuals:
                continue
            target = display_name(obj)
            rels.append(
                {
                    "source": source,
                    "target": target,
                    "type": rel_type,
                    "props": {"sourceDataset": "lotro_owl"},
                }
            )

    return rels


def wait_for_neo4j(driver: Any, attempts: int = 30) -> None:
    for attempt in range(1, attempts + 1):
        try:
            with driver.session() as session:
                session.run("RETURN 1").consume()
            return
        except ServiceUnavailable:
            time.sleep(2)
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2)
    raise RuntimeError("Neo4j nao respondeu a tempo")


def write_graph(
    uri: str,
    username: str,
    password: str,
    entities: dict[str, EntityDraft],
    rels: list[dict[str, Any]],
    reset: bool,
) -> None:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    wait_for_neo4j(driver)

    entity_rows = []
    for entity in entities.values():
        labels = sorted(entity.labels & SAFE_LABELS)
        props = safe_props(entity.props)
        props["aliases"] = sorted(entity.aliases)
        props["sources"] = sorted(entity.sources)
        props.setdefault("kind", labels[0] if labels else "Entity")
        entity_rows.append({"name": entity.name, "labels": labels, "props": props})

    rels_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rel in rels:
        rel_type = rel["type"]
        if not rel_type.replace("_", "").isalpha() or rel_type.upper() != rel_type:
            continue
        rels_by_type[rel_type].append(
            {
                "source": rel["source"],
                "target": rel["target"],
                "props": safe_props(rel.get("props", {})),
            }
        )

    with driver.session() as session:
        if reset:
            session.run("MATCH (n) DETACH DELETE n").consume()

        session.run(
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        ).consume()

        session.run(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {name: row.name})
            SET e += row.props
            FOREACH (_ IN CASE WHEN 'Character' IN row.labels THEN [1] ELSE [] END | SET e:Character)
            FOREACH (_ IN CASE WHEN 'Weapon' IN row.labels THEN [1] ELSE [] END | SET e:Weapon)
            FOREACH (_ IN CASE WHEN 'Place' IN row.labels THEN [1] ELSE [] END | SET e:Place)
            FOREACH (_ IN CASE WHEN 'Language' IN row.labels THEN [1] ELSE [] END | SET e:Language)
            FOREACH (_ IN CASE WHEN 'Race' IN row.labels THEN [1] ELSE [] END | SET e:Race)
            FOREACH (_ IN CASE WHEN 'Name' IN row.labels THEN [1] ELSE [] END | SET e:Name)
            FOREACH (_ IN CASE WHEN 'Class' IN row.labels THEN [1] ELSE [] END | SET e:Class)
            FOREACH (_ IN CASE WHEN 'OntologyEntity' IN row.labels THEN [1] ELSE [] END | SET e:OntologyEntity)
            """,
            rows=entity_rows,
        ).consume()

        for rel_type, rows in rels_by_type.items():
            session.run(
                f"""
                UNWIND $rows AS row
                MATCH (a:Entity {{name: row.source}})
                MATCH (b:Entity {{name: row.target}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += row.props
                """,
                rows=rows,
            ).consume()

        node_count = session.run("MATCH (n:Entity) RETURN count(n) AS count").single()["count"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]

    driver.close()
    print(f"importado: {node_count} entidades, {rel_count} relacoes")


def build_graph() -> tuple[dict[str, EntityDraft], list[dict[str, Any]]]:
    entities: dict[str, EntityDraft] = {}
    rels: list[dict[str, Any]] = []
    rels.extend(read_interactions(entities))
    rels.extend(read_properties(entities))
    rels.extend(read_ontology(entities))
    return entities, rels


def main() -> int:
    parser = argparse.ArgumentParser(description="Importa o grafo hibrido LOTR para Neo4j.")
    parser.add_argument("--no-reset", action="store_true", help="nao apaga o grafo atual antes de importar")
    args = parser.parse_args()

    try:
        ensure_files()
        entities, rels = build_graph()
        write_graph(
            getenv("NEO4J_URI", NEO4J_URI),
            getenv("NEO4J_USERNAME", NEO4J_USERNAME),
            getenv("NEO4J_PASSWORD", NEO4J_PASSWORD),
            entities,
            rels,
            reset=not args.no_reset,
        )
    except Exception as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
