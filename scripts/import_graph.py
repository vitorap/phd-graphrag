from __future__ import annotations

import argparse
import csv
import math
import re
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
SNA_DIR = RAW_DIR / "sna_lotr"
SNA_FILES = {
    "book_1": SNA_DIR / "LOTR1_book_CLEAN.txt",
    "book_2": SNA_DIR / "LOTR2_book_CLEAN.txt",
    "book_3": SNA_DIR / "LOTR3_book_CLEAN.txt",
    "nodes": SNA_DIR / "nodes.csv",
    "weighted_edges": SNA_DIR / "weightededges.csv",
    "chapter_edges": SNA_DIR / "edges_chapters.csv",
    "scripts": SNA_DIR / "lotr_scripts.csv",
    "node_sentiment": SNA_DIR / "nodes_with_sentiment.csv",
    "character_sentiment": SNA_DIR / "sentiment_per_character.csv",
    "prediction_1": SNA_DIR / "prediction_1.csv",
    "prediction_2": SNA_DIR / "prediction_2.csv",
    "prediction_3": SNA_DIR / "prediction_3.csv",
}

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
    "sam": "Samwise",
    "samwise gamgee": "Samwise",
    "merry": "Meriadoc",
    "meriadoc brandybuck": "Meriadoc",
    "pippin": "Peregrin",
    "peregrin took": "Peregrin",
    "elessar": "Aragorn",
    "strider": "Aragorn",
    "smeagol": "Gollum",
    "sméagol": "Gollum",
    "barliman": "Barliman Butterbur",
    "farmer maggot": "Maggot",
    "gaffer": "Hamfast Gamgee",
    "rosie": "Rosie Cotton",
    "sandyman": "Ted Sandyman",
    "witch king": "Witch-king of Angmar",
    "witch-king": "Witch-king of Angmar",
    "mouth of sauron": "Mouth of Sauron",
}

BOOK_SOURCES = [
    ("lotr1", "The Fellowship of the Ring", SNA_FILES["book_1"]),
    ("lotr2", "The Two Towers", SNA_FILES["book_2"]),
    ("lotr3", "The Return of the King", SNA_FILES["book_3"]),
]

MOVIE_ORDER = {
    "The Fellowship of the Ring": 1,
    "The Two Towers": 2,
    "The Return of the King": 3,
}

STOP_MENTION_TERMS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "he",
    "i",
    "in",
    "it",
    "man",
    "men",
    "no",
    "of",
    "on",
    "or",
    "the",
    "to",
    "we",
}


@dataclass
class EntityDraft:
    name: str
    labels: set[str] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    props: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChapterDraft:
    id: str
    title: str
    book_id: str
    book_title: str
    local_index: int
    global_index: int
    sentiment: float | None = None


@dataclass
class TextDocumentDraft:
    id: str
    labels: set[str]
    text: str
    source_type: str
    source_title: str
    sequence: int
    mentions: set[str] = field(default_factory=set)
    book_id: str | None = None
    chapter_id: str | None = None
    chapter_title: str | None = None
    movie_id: str | None = None
    speaker: str | None = None
    line_number: int | None = None


def getenv(name: str, default: str) -> str:
    import os

    return os.getenv(name, default)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value.strip()


def normalize_key(value: str) -> str:
    value = normalize_text(value).casefold()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def title_name(value: str) -> str:
    titled = value.strip().title()
    for source, target in {
        " Of ": " of ",
        " The ": " the ",
        " And ": " and ",
        " Ii": " II",
        " Iii": " III",
    }.items():
        titled = titled.replace(source, target)
    return titled


def missing_name(value: str | None) -> bool:
    if value is None:
        return True
    return normalize_key(value) in {"", "na", "n a", "nan", "none", "null"}


def canonical_name(value: str) -> str:
    cleaned = value.strip()
    alias = ALIASES.get(cleaned) or ALIASES.get(normalize_text(cleaned)) or ALIASES.get(normalize_key(cleaned))
    if alias:
        return alias
    if cleaned.isupper():
        return title_name(cleaned)
    if cleaned.islower():
        return title_name(cleaned)
    return cleaned


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
    required = [LOTR_CSV, PROPERTIES_CSV, OWL_XML, *SNA_FILES.values()]
    missing = [path for path in required if not path.exists()]
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


def read_sna_nodes(entities: dict[str, EntityDraft]) -> list[dict[str, Any]]:
    rels: list[dict[str, Any]] = []
    with SNA_FILES["nodes"].open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_name = (row.get("Character") or "").strip()
            if missing_name(raw_name):
                continue
            name = canonical_name(raw_name)
            race = (row.get("Race") or "").strip()
            gender = (row.get("Gender") or "").strip()
            props = {"race": race, "gender": gender, "snaIndex": row.get("Index")}
            add_entity(
                entities,
                name,
                {"Character"},
                "sna_lotr_nodes",
                aliases={raw_name},
                props=props,
            )
            if race:
                race_name = title_name(race)
                add_entity(entities, race_name, {"Race"}, "sna_lotr_nodes", props={"kind": "Race"})
                rels.append(
                    {
                        "source": name,
                        "target": race_name,
                        "type": "HAS_RACE",
                        "props": {"sourceDataset": "sna_lotr_nodes"},
                    }
                )
    return rels


def read_sna_weighted_edges(entities: dict[str, EntityDraft]) -> list[dict[str, Any]]:
    rels: list[dict[str, Any]] = []
    with SNA_FILES["weighted_edges"].open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_raw = (row.get("From") or "").strip()
            target_raw = (row.get("To") or "").strip()
            if missing_name(source_raw) or missing_name(target_raw):
                continue
            source = canonical_name(source_raw)
            target = canonical_name(target_raw)
            if source == target:
                continue
            weight = safe_float(row.get("weight"), default=1.0)
            add_entity(entities, source, {"Character"}, "sna_lotr_weightededges", aliases={source_raw})
            add_entity(entities, target, {"Character"}, "sna_lotr_weightededges", aliases={target_raw})
            rels.append(
                {
                    "source": source,
                    "target": target,
                    "type": "CO_OCCURS_WITH",
                    "props": {"weight": weight, "sourceDataset": "sna_lotr_weightededges"},
                }
            )
    return rels


def read_sna_sentiment(entities: dict[str, EntityDraft]) -> None:
    if SNA_FILES["node_sentiment"].exists():
        with SNA_FILES["node_sentiment"].open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_name = (row.get("Character") or "").strip()
                if missing_name(raw_name):
                    continue
                name = canonical_name(raw_name)
                entity = add_entity(entities, name, {"Character"}, "sna_lotr_sentiment", aliases={raw_name})
                total_words = safe_float(row.get("Total_words"), default=0.0)
                avg_sent = safe_float(row.get("Avg_sent"), default=0.0)
                entity.props["scriptWordCount"] = total_words
                entity.props["scriptAvgSentiment"] = avg_sent

    if SNA_FILES["character_sentiment"].exists():
        with SNA_FILES["character_sentiment"].open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_name = (row.get("Character") or "").strip()
                if missing_name(raw_name):
                    continue
                name = canonical_name(raw_name)
                entity = add_entity(entities, name, {"Character"}, "sna_lotr_character_sentiment", aliases={raw_name})
                for key, prop_name in {
                    "Words_LOTR1": "wordsLOTR1",
                    "Words_LOTR2": "wordsLOTR2",
                    "Words_LOTR3": "wordsLOTR3",
                    "Sent_LOTR1": "sentimentLOTR1",
                    "Sent_LOTR2": "sentimentLOTR2",
                    "Sent_LOTR3": "sentimentLOTR3",
                    "Avg_sent": "avgSentiment",
                    "Total_words": "totalWords",
                }.items():
                    entity.props[prop_name] = safe_float(row.get(key), default=0.0)


def read_sna_predictions(entities: dict[str, EntityDraft]) -> list[dict[str, Any]]:
    rels: list[dict[str, Any]] = []
    prediction_specs = [
        ("prediction_1", "existing_link"),
        ("prediction_2", "jaccard"),
        ("prediction_3", "adamic_adar"),
    ]
    for file_key, method in prediction_specs:
        with SNA_FILES[file_key].open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source_raw = (row.get("Source") or "").strip()
                target_raw = (row.get("Target") or "").strip()
                if missing_name(source_raw) or missing_name(target_raw):
                    continue
                source = canonical_name(source_raw)
                target = canonical_name(target_raw)
                if source == target:
                    continue
                score = first_float(
                    row,
                    ["Jaccard Coefficient", "Adamic Adar", "Adamic Adar Score", "score", "Score"],
                    default=1.0,
                )
                add_entity(entities, source, {"Character"}, f"sna_lotr_{file_key}", aliases={source_raw})
                add_entity(entities, target, {"Character"}, f"sna_lotr_{file_key}", aliases={target_raw})
                rels.append(
                    {
                        "source": source,
                        "target": target,
                        "type": "PREDICTED_LINK",
                        "props": {
                            "confidence": float(score),
                            "method": method,
                            "sourceDataset": f"sna_lotr_{file_key}",
                        },
                    }
                )
    return rels


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def first_float(row: dict[str, Any], keys: list[str], default: float) -> float:
    for key in keys:
        value = safe_float(row.get(key), default=None)
        if value is not None:
            return value
    return default


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


def load_chapter_sentiment() -> dict[int, float]:
    sentiments: dict[int, float] = {}
    chapter_file = SNA_DIR / "nodes_chapter_sent.csv"
    if not chapter_file.exists():
        return sentiments
    with chapter_file.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            chapter = (row.get("Chapter") or "").strip()
            match = re.search(r"(\d+)", chapter)
            if not match:
                continue
            sentiment = safe_float(row.get("SentimentScore"), default=None)
            if sentiment is not None:
                sentiments[int(match.group(1))] = sentiment
    return sentiments


def chapter_sections(book_id: str, book_title: str, path: Path, start_global: int) -> tuple[list[ChapterDraft], list[dict[str, Any]], int]:
    chapter_pattern = re.compile(r"^\s*Chapter\s+(\d+)\s*\.?\s*(.*)$", re.IGNORECASE)
    sentiments = load_chapter_sentiment()
    chapters: list[ChapterDraft] = []
    sections: list[dict[str, Any]] = []
    current_lines: list[str] = []
    current_chapter: ChapterDraft | None = None
    local_count = 0
    global_count = start_global

    def flush() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        if text:
            sections.append({"chapter": current_chapter, "text": text})
        current_lines = []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = chapter_pattern.match(line)
        if match:
            flush()
            local_count += 1
            global_count += 1
            title = f"Chapter {match.group(1)}"
            suffix = match.group(2).strip()
            if suffix:
                title = f"{title}. {suffix}"
            chapter = ChapterDraft(
                id=f"{book_id}_chapter_{local_count:03d}",
                title=title,
                book_id=book_id,
                book_title=book_title,
                local_index=local_count,
                global_index=global_count,
                sentiment=sentiments.get(global_count),
            )
            chapters.append(chapter)
            current_chapter = chapter
        current_lines.append(line)

    flush()
    return chapters, sections, global_count


def chunk_text(text: str, chunk_size: int = 360, overlap: int = 60) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        if len(piece) < 80 and chunks:
            break
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_mention_aliases(entities: dict[str, EntityDraft]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for entity in entities.values():
        if not entity.labels & {"Character", "Weapon", "Place", "Language", "OntologyEntity"}:
            continue
        names = {entity.name, *entity.aliases}
        for name in names:
            key = normalize_key(name)
            if len(key) < 4 or key in STOP_MENTION_TERMS:
                continue
            pairs.append((entity.name, key))
    pairs.sort(key=lambda item: (-len(item[1]), item[1]))
    return pairs


def detect_mentions(text: str, aliases: list[tuple[str, str]], limit: int = 32) -> set[str]:
    normalized = f" {normalize_key(text)} "
    mentions: set[str] = set()
    for entity_name, alias_key in aliases:
        if f" {alias_key} " in normalized:
            mentions.add(entity_name)
            if len(mentions) >= limit:
                break
    return mentions


def slug(value: str) -> str:
    key = normalize_key(value)
    return key.replace(" ", "_") or "unknown"


def build_text_corpus(
    entities: dict[str, EntityDraft],
) -> tuple[list[dict[str, Any]], list[ChapterDraft], list[TextDocumentDraft], list[dict[str, Any]]]:
    aliases = build_mention_aliases(entities)
    books = [
        {
            "id": book_id,
            "title": title,
            "order": index + 1,
            "sourceDataset": "sna_lotr_books",
        }
        for index, (book_id, title, _path) in enumerate(BOOK_SOURCES)
    ]
    chapters: list[ChapterDraft] = []
    documents: list[TextDocumentDraft] = []
    movies: dict[str, dict[str, Any]] = {}
    global_chapter = 0
    sequence = 0

    for book_id, book_title, path in BOOK_SOURCES:
        book_chapters, sections, global_chapter = chapter_sections(book_id, book_title, path, global_chapter)
        chapters.extend(book_chapters)
        for section in sections:
            chapter = section["chapter"]
            for chunk in chunk_text(section["text"]):
                sequence += 1
                mentions = detect_mentions(chunk, aliases)
                documents.append(
                    TextDocumentDraft(
                        id=f"{book_id}_chunk_{sequence:05d}",
                        labels={"TextChunk"},
                        text=chunk,
                        source_type="book",
                        source_title=book_title,
                        sequence=sequence,
                        mentions=mentions,
                        book_id=book_id,
                        chapter_id=chapter.id if chapter else None,
                        chapter_title=chapter.title if chapter else None,
                    )
                )

    with SNA_FILES["scripts"].open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            raw_speaker = (row.get("char") or "").strip()
            text = clean_dialogue(row.get("dialog") or "")
            movie_title = (row.get("movie") or "").strip()
            if not text or missing_name(raw_speaker):
                continue
            speaker = canonical_name(raw_speaker)
            add_entity(entities, speaker, {"Character"}, "sna_lotr_scripts", aliases={raw_speaker})
            movie_id = slug(movie_title)
            movies.setdefault(
                movie_id,
                {
                    "id": movie_id,
                    "title": movie_title,
                    "order": MOVIE_ORDER.get(movie_title, 99),
                    "sourceDataset": "sna_lotr_scripts",
                },
            )
            mentions = detect_mentions(text, aliases)
            mentions.add(speaker)
            line_number = parse_int(row.get("") or row.get("Index"))
            sequence += 1
            documents.append(
                TextDocumentDraft(
                    id=f"dialogue_{movie_id}_{idx:05d}",
                    labels={"DialogueLine"},
                    text=text,
                    source_type="dialogue",
                    source_title=movie_title,
                    sequence=sequence,
                    mentions=mentions,
                    movie_id=movie_id,
                    speaker=speaker,
                    line_number=line_number,
                )
            )

    return books, chapters, documents, list(movies.values())


def clean_dialogue(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def chapter_similarity_relationships(chapters: list[ChapterDraft]) -> list[dict[str, Any]]:
    by_global = {chapter.global_index: chapter.id for chapter in chapters}
    rels: list[dict[str, Any]] = []
    with SNA_FILES["chapter_edges"].open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_number = chapter_number(row.get("From"))
            target_number = chapter_number(row.get("To"))
            if source_number not in by_global or target_number not in by_global:
                continue
            if source_number == target_number:
                continue
            rels.append(
                {
                    "source": by_global[source_number],
                    "target": by_global[target_number],
                    "weight": safe_float(row.get("weight"), default=1.0) or 1.0,
                }
            )
    return rels


def chapter_number(value: Any) -> int | None:
    match = re.search(r"(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


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
    books: list[dict[str, Any]],
    chapters: list[ChapterDraft],
    documents: list[TextDocumentDraft],
    movies: list[dict[str, Any]],
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

    book_rows = [{"id": row["id"], "props": safe_props(row)} for row in books]
    movie_rows = [{"id": row["id"], "props": safe_props(row)} for row in movies]
    chapter_rows = [
        {
            "id": chapter.id,
            "bookId": chapter.book_id,
            "props": safe_props(
                {
                    "id": chapter.id,
                    "title": chapter.title,
                    "bookId": chapter.book_id,
                    "bookTitle": chapter.book_title,
                    "localIndex": chapter.local_index,
                    "globalIndex": chapter.global_index,
                    "sentiment": chapter.sentiment,
                    "sourceDataset": "sna_lotr_books",
                }
            ),
        }
        for chapter in chapters
    ]
    document_rows = [
        {
            "id": doc.id,
            "kind": "DialogueLine" if "DialogueLine" in doc.labels else "TextChunk",
            "bookId": doc.book_id,
            "chapterId": doc.chapter_id,
            "movieId": doc.movie_id,
            "speaker": doc.speaker,
            "props": safe_props(
                {
                    "id": doc.id,
                    "text": doc.text,
                    "sourceType": doc.source_type,
                    "sourceTitle": doc.source_title,
                    "sequence": doc.sequence,
                    "chapterTitle": doc.chapter_title,
                    "bookId": doc.book_id,
                    "chapterId": doc.chapter_id,
                    "movieId": doc.movie_id,
                    "speaker": doc.speaker,
                    "lineNumber": doc.line_number,
                    "mentionCount": len(doc.mentions),
                    "sourceDataset": "sna_lotr_books"
                    if doc.source_type == "book"
                    else "sna_lotr_scripts",
                }
            ),
        }
        for doc in documents
    ]
    mention_rows = [
        {"documentId": doc.id, "entity": entity}
        for doc in documents
        for entity in sorted(doc.mentions)
        if entity in entities
    ]
    speaker_rows = [
        {"documentId": doc.id, "speaker": doc.speaker}
        for doc in documents
        if doc.speaker and doc.speaker in entities
    ]
    chapter_edge_rows = chapter_similarity_relationships(chapters)

    with driver.session() as session:
        if reset:
            session.run("MATCH (n) DETACH DELETE n").consume()

        session.run(
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
        ).consume()
        session.run(
            "CREATE CONSTRAINT book_id IF NOT EXISTS FOR (b:Book) REQUIRE b.id IS UNIQUE"
        ).consume()
        session.run(
            "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE"
        ).consume()
        session.run(
            "CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE c.id IS UNIQUE"
        ).consume()
        session.run(
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:RetrievalDocument) REQUIRE d.id IS UNIQUE"
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
            if rel_type == "PREDICTED_LINK":
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (a:Entity {name: row.source})
                    MATCH (b:Entity {name: row.target})
                    MERGE (a)-[r:PREDICTED_LINK {method: row.props.method}]->(b)
                    SET r += row.props
                    """,
                    rows=rows,
                ).consume()
            else:
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

        session.run(
            """
            UNWIND $rows AS row
            MERGE (b:Book {id: row.id})
            SET b += row.props
            """,
            rows=book_rows,
        ).consume()

        session.run(
            """
            UNWIND $rows AS row
            MERGE (m:Movie {id: row.id})
            SET m += row.props
            """,
            rows=movie_rows,
        ).consume()

        session.run(
            """
            UNWIND $rows AS row
            MERGE (c:Chapter {id: row.id})
            SET c += row.props
            WITH c, row
            MATCH (b:Book {id: row.bookId})
            MERGE (c)-[:IN_BOOK {sourceDataset: 'sna_lotr_books'}]->(b)
            """,
            rows=chapter_rows,
        ).consume()

        session.run(
            """
            UNWIND $rows AS row
            MERGE (d:RetrievalDocument {id: row.id})
            SET d += row.props
            FOREACH (_ IN CASE WHEN row.kind = 'TextChunk' THEN [1] ELSE [] END | SET d:TextChunk)
            FOREACH (_ IN CASE WHEN row.kind = 'DialogueLine' THEN [1] ELSE [] END | SET d:DialogueLine)
            """,
            rows=document_rows,
        ).consume()

        session.run(
            """
            UNWIND $rows AS row
            MATCH (d:RetrievalDocument {id: row.id})
            MATCH (b:Book {id: row.bookId})
            MERGE (d)-[:IN_BOOK {sourceDataset: 'sna_lotr_books'}]->(b)
            """,
            rows=[row for row in document_rows if row["bookId"]],
        ).consume()

        session.run(
            """
            UNWIND $rows AS row
            MATCH (d:RetrievalDocument {id: row.id})
            MATCH (c:Chapter {id: row.chapterId})
            MERGE (d)-[:IN_CHAPTER {sourceDataset: 'sna_lotr_books'}]->(c)
            """,
            rows=[row for row in document_rows if row["chapterId"]],
        ).consume()

        session.run(
            """
            UNWIND $rows AS row
            MATCH (d:RetrievalDocument {id: row.id})
            MATCH (m:Movie {id: row.movieId})
            MERGE (d)-[:IN_MOVIE {sourceDataset: 'sna_lotr_scripts'}]->(m)
            """,
            rows=[row for row in document_rows if row["movieId"]],
        ).consume()

        session.run(
            """
            UNWIND $rows AS row
            MATCH (d:RetrievalDocument {id: row.documentId})
            MATCH (e:Entity {name: row.entity})
            MERGE (d)-[:MENTIONS {sourceDataset: 'sna_lotr_text'}]->(e)
            """,
            rows=mention_rows,
        ).consume()

        session.run(
            """
            UNWIND $rows AS row
            MATCH (d:RetrievalDocument {id: row.documentId})
            MATCH (e:Entity {name: row.speaker})
            MERGE (e)-[:SPEAKS_LINE {sourceDataset: 'sna_lotr_scripts'}]->(d)
            """,
            rows=speaker_rows,
        ).consume()

        session.run(
            """
            UNWIND $rows AS row
            MATCH (a:Chapter {id: row.source})
            MATCH (b:Chapter {id: row.target})
            MERGE (a)-[r:SIMILAR_CHAPTER]->(b)
            SET r.weight = row.weight,
                r.sourceDataset = 'sna_lotr_edges_chapters'
            """,
            rows=chapter_edge_rows,
        ).consume()

        node_count = session.run("MATCH (n:Entity) RETURN count(n) AS count").single()["count"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
        doc_count = session.run("MATCH (d:RetrievalDocument) RETURN count(d) AS count").single()["count"]

    driver.close()
    print(f"importado: {node_count} entidades, {doc_count} documentos, {rel_count} relacoes")


def build_graph() -> tuple[
    dict[str, EntityDraft],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[ChapterDraft],
    list[TextDocumentDraft],
    list[dict[str, Any]],
]:
    entities: dict[str, EntityDraft] = {}
    rels: list[dict[str, Any]] = []
    rels.extend(read_interactions(entities))
    rels.extend(read_properties(entities))
    rels.extend(read_sna_nodes(entities))
    rels.extend(read_sna_weighted_edges(entities))
    read_sna_sentiment(entities)
    rels.extend(read_sna_predictions(entities))
    rels.extend(read_ontology(entities))
    books, chapters, documents, movies = build_text_corpus(entities)
    return entities, rels, books, chapters, documents, movies


def main() -> int:
    parser = argparse.ArgumentParser(description="Importa o grafo hibrido LOTR para Neo4j.")
    parser.add_argument("--no-reset", action="store_true", help="nao apaga o grafo atual antes de importar")
    args = parser.parse_args()

    try:
        ensure_files()
        entities, rels, books, chapters, documents, movies = build_graph()
        write_graph(
            getenv("NEO4J_URI", NEO4J_URI),
            getenv("NEO4J_USERNAME", NEO4J_USERNAME),
            getenv("NEO4J_PASSWORD", NEO4J_PASSWORD),
            entities,
            rels,
            books,
            chapters,
            documents,
            movies,
            reset=not args.no_reset,
        )
    except Exception as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
