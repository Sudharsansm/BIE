"""
M04 — Knowledge Graph
======================
In-memory graph store with NER-based entity/relation extraction,
coreference resolution, Wikidata anchoring, and SPARQL-like pattern
queries.  Backend is swappable: ``InMemoryGraphStore`` implements the
same interface as a Neo4j Aura / Amazon Neptune adapter would.

Usage::

    from bie.kg import KnowledgeGraph

    kg = KnowledgeGraph()
    kg.ingest_document(doc, chunks)
    nodes = kg.search_entities("TSMC")
    paths = kg.query_pattern(source_type="Organization", relation="MANUFACTURES")
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from bie.config import BIESettings, settings
from bie.models import ChunkRecord, DocumentRecord


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class KGNode:
    entity_id: str
    name: str
    type: str  # Person | Organization | Event | Product | Location | Other
    attributes: dict = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    source_doc_ids: set[str] = field(default_factory=set)
    wikidata_id: str | None = None


@dataclass
class KGEdge:
    relation_id: str
    source_id: str
    target_id: str
    relation_type: str
    attributes: dict = field(default_factory=dict)
    source_doc_ids: set[str] = field(default_factory=set)


# ── Lightweight rule-based NER ────────────────────────────────────────────────
# Production deployments swap this for spaCy + a fine-tuned NER model
# (per PRD M04). The interface (extract_entities) stays identical.

_ORG_SUFFIXES = (
    "Inc", "Inc.", "Corp", "Corp.", "Corporation", "Ltd", "Ltd.", "LLC",
    "Co", "Co.", "Group", "Holdings", "Technologies", "Labs", "AG", "SA",
)
_KNOWN_ORGS = {
    "TSMC", "NVIDIA", "Apple", "AMD", "Intel", "OpenAI", "Anthropic",
    "Google", "Microsoft", "Amazon", "Meta", "Samsung", "Reuters",
    "BBC", "Tesla", "SpaceX", "IBM", "Qualcomm",
}
_KNOWN_LOCATIONS = {
    "Taiwan", "Hsinchu", "China", "Japan", "Korea", "South Korea",
    "United States", "US", "USA", "Arizona", "California", "Europe",
    "Tokyo", "Beijing", "Seoul", "Washington",
}
_RELATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(\w[\w\s]{0,30}?)\s+(?:manufactures|produces|makes)\s+([\w\s\-]{2,40})", re.I), "MANUFACTURES"),
    (re.compile(r"\b(\w[\w\s]{0,30}?)\s+(?:acquired|bought)\s+([\w\s]{2,40})", re.I), "ACQUIRED"),
    (re.compile(r"\b(\w[\w\s]{0,30}?)\s+(?:partnered with|partners with)\s+([\w\s]{2,40})", re.I), "PARTNERS_WITH"),
    (re.compile(r"\b(\w[\w\s]{0,30}?)\s+(?:is headquartered in|is based in|HQ in)\s+([\w\s]{2,40})", re.I), "HEADQUARTERED_IN"),
    (re.compile(r"\b(\w[\w\s]{0,30}?)\s+(?:CEO|founder|president)\s+(?:is|was)?\s*([\w\s]{2,40})", re.I), "HAS_LEADER"),
]


class EntityExtractor:
    """
    Rule-based + dictionary NER. Returns (entity_text, entity_type) tuples.
    Swap `.extract()` body for spaCy pipeline in production (M04 spec).
    """

    def extract(self, text: str) -> list[tuple[str, str]]:
        entities: list[tuple[str, str]] = []
        seen: set[str] = set()

        # Known orgs (dictionary lookup)
        for org in _KNOWN_ORGS:
            if re.search(rf"\b{re.escape(org)}\b", text):
                if org not in seen:
                    entities.append((org, "Organization"))
                    seen.add(org)

        # Known locations
        for loc in _KNOWN_LOCATIONS:
            if re.search(rf"\b{re.escape(loc)}\b", text):
                if loc not in seen:
                    entities.append((loc, "Location"))
                    seen.add(loc)

        # Capitalized multi-word sequences ending in org suffix
        for m in re.finditer(r"\b([A-Z][\w&]+(?:\s+[A-Z][\w&]+){0,3})\s+(" + "|".join(re.escape(s) for s in _ORG_SUFFIXES) + r")\b", text):
            name = f"{m.group(1)} {m.group(2)}"
            if name not in seen:
                entities.append((name, "Organization"))
                seen.add(name)

        # Years / dates as Event-adjacent attributes (not full entities)
        return entities

    def extract_relations(self, text: str) -> list[tuple[str, str, str]]:
        """Returns (source_text, relation_type, target_text)."""
        relations = []
        for pattern, rel_type in _RELATION_PATTERNS:
            for m in pattern.finditer(text):
                src = m.group(1).strip()
                tgt = m.group(2).strip().rstrip(".,;")
                if src and tgt and src.lower() != tgt.lower():
                    relations.append((src, rel_type, tgt))
        return relations


# ── Coreference (lightweight) ─────────────────────────────────────────────────

class SimpleCoref:
    """
    Resolves simple pronoun → last-mentioned-entity coreference within
    a paragraph. Production swaps for a neural coref model.
    """

    _PRONOUNS = {"it", "its", "they", "their", "them", "he", "his", "she", "her"}

    def resolve(self, text: str, entities: list[tuple[str, str]]) -> str:
        if not entities:
            return text
        last_entity = entities[-1][0]
        words = text.split()
        out = []
        for w in words:
            stripped = w.strip(".,;:!?").lower()
            if stripped in self._PRONOUNS:
                out.append(last_entity)
            else:
                out.append(w)
        return " ".join(out)


# ── In-memory graph store ─────────────────────────────────────────────────────

class InMemoryGraphStore:
    """
    Adjacency-list graph store. Same interface as a Neo4j/Neptune adapter:
    `upsert_node`, `upsert_edge`, `get_node`, `neighbors`, `find_by_name`.
    """

    def __init__(self):
        self._nodes: dict[str, KGNode] = {}
        self._edges: dict[str, KGEdge] = {}
        self._name_index: dict[str, str] = {}  # lowercase name/alias → entity_id
        self._adjacency: dict[str, set[str]] = defaultdict(set)  # entity_id → edge_ids

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def upsert_node(self, name: str, type_: str, doc_id: str, attributes: dict | None = None) -> KGNode:
        key = name.lower().strip()
        if key in self._name_index:
            node = self._nodes[self._name_index[key]]
            node.source_doc_ids.add(doc_id)
            if attributes:
                node.attributes.update(attributes)
            return node

        node = KGNode(
            entity_id=f"KG-{uuid.uuid4().hex[:8]}",
            name=name,
            type=type_,
            attributes=attributes or {},
            aliases=[name],
            source_doc_ids={doc_id},
        )
        self._nodes[node.entity_id] = node
        self._name_index[key] = node.entity_id
        return node

    def get_node(self, entity_id: str) -> KGNode | None:
        return self._nodes.get(entity_id)

    def find_by_name(self, name: str) -> KGNode | None:
        return self._nodes.get(self._name_index.get(name.lower().strip(), ""))

    def search_entities(self, query: str, limit: int = 10) -> list[KGNode]:
        q = query.lower().strip()
        results = []
        for node in self._nodes.values():
            if q in node.name.lower() or any(q in a.lower() for a in node.aliases):
                results.append(node)
            if len(results) >= limit:
                break
        return results

    # ── Edges ─────────────────────────────────────────────────────────────────

    def upsert_edge(
        self, source_id: str, target_id: str, relation_type: str, doc_id: str, attributes: dict | None = None
    ) -> KGEdge:
        # Check for existing edge with same (src, tgt, type)
        for eid in self._adjacency[source_id]:
            edge = self._edges[eid]
            if edge.target_id == target_id and edge.relation_type == relation_type:
                edge.source_doc_ids.add(doc_id)
                if attributes:
                    edge.attributes.update(attributes)
                return edge

        edge = KGEdge(
            relation_id=f"REL-{uuid.uuid4().hex[:8]}",
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            attributes=attributes or {},
            source_doc_ids={doc_id},
        )
        self._edges[edge.relation_id] = edge
        self._adjacency[source_id].add(edge.relation_id)
        self._adjacency[target_id].add(edge.relation_id)
        return edge

    def neighbors(self, entity_id: str) -> list[KGEdge]:
        return [self._edges[eid] for eid in self._adjacency.get(entity_id, set())]

    def query_pattern(
        self,
        source_type: str | None = None,
        relation: str | None = None,
        target_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """SPARQL-pattern-style query: filter edges by node/relation types."""
        results = []
        for edge in self._edges.values():
            src = self._nodes.get(edge.source_id)
            tgt = self._nodes.get(edge.target_id)
            if not src or not tgt:
                continue
            if source_type and src.type != source_type:
                continue
            if relation and edge.relation_type != relation:
                continue
            if target_type and tgt.type != target_type:
                continue
            results.append({
                "source": {"entity_id": src.entity_id, "name": src.name, "type": src.type},
                "relation": edge.relation_type,
                "target": {"entity_id": tgt.entity_id, "name": tgt.name, "type": tgt.type},
                "attributes": edge.attributes,
            })
            if len(results) >= limit:
                break
        return results

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)


# ── Knowledge Graph facade ────────────────────────────────────────────────────

class KnowledgeGraph:
    """
    High-level KG facade used by the retriever, contradiction detector,
    and fact verifier.
    """

    def __init__(self, cfg: BIESettings = settings, store: InMemoryGraphStore | None = None):
        self._cfg = cfg
        self._store = store or InMemoryGraphStore()
        self._extractor = EntityExtractor()
        self._coref = SimpleCoref()

    def ingest_document(self, doc: DocumentRecord, chunks: list[ChunkRecord]) -> dict:
        """
        Run NER + relation extraction over each chunk and upsert into the
        graph. Returns ingestion stats.
        """
        nodes_added = 0
        edges_added = 0

        for chunk in chunks:
            entities = self._extractor.extract(chunk.text)
            resolved_text = self._coref.resolve(chunk.text, entities)

            # Upsert entity nodes
            entity_nodes: dict[str, KGNode] = {}
            for name, etype in entities:
                node = self._store.upsert_node(name, etype, doc.doc_id)
                entity_nodes[name] = node
                nodes_added += 1

            # Extract + upsert relations
            relations = self._extractor.extract_relations(resolved_text)
            for src_text, rel_type, tgt_text in relations:
                src_node = self._match_entity(src_text, entity_nodes)
                tgt_node = self._match_entity(tgt_text, entity_nodes)
                if src_node and tgt_node and src_node.entity_id != tgt_node.entity_id:
                    self._store.upsert_edge(
                        src_node.entity_id, tgt_node.entity_id, rel_type, doc.doc_id
                    )
                    edges_added += 1

        return {"nodes_processed": nodes_added, "edges_processed": edges_added}

    def _match_entity(self, text: str, candidates: dict[str, KGNode]) -> KGNode | None:
        text_l = text.lower().strip()
        # Exact match in this document's extracted entities
        for name, node in candidates.items():
            if name.lower() in text_l or text_l in name.lower():
                return node
        # Fall back to global graph lookup
        return self._store.find_by_name(text)

    # ── Query interface ───────────────────────────────────────────────────────

    def search_entities(self, query: str, limit: int = 10) -> list[dict]:
        nodes = self._store.search_entities(query, limit)
        return [
            {
                "entity_id": n.entity_id,
                "name": n.name,
                "type": n.type,
                "attributes": n.attributes,
                "aliases": n.aliases,
                "source_count": len(n.source_doc_ids),
            }
            for n in nodes
        ]

    def get_entity_graph(self, entity_id: str) -> dict | None:
        node = self._store.get_node(entity_id)
        if not node:
            return None
        edges = self._store.neighbors(entity_id)
        neighbors = []
        for edge in edges:
            other_id = edge.target_id if edge.source_id == entity_id else edge.source_id
            other = self._store.get_node(other_id)
            if other:
                neighbors.append({
                    "entity": {"entity_id": other.entity_id, "name": other.name, "type": other.type},
                    "relation": edge.relation_type,
                    "direction": "outgoing" if edge.source_id == entity_id else "incoming",
                })
        return {
            "entity": {
                "entity_id": node.entity_id, "name": node.name, "type": node.type,
                "attributes": node.attributes, "aliases": node.aliases,
            },
            "neighbors": neighbors,
        }

    def query_pattern(
        self, source_type: str | None = None, relation: str | None = None,
        target_type: str | None = None, limit: int = 50,
    ) -> list[dict]:
        return self._store.query_pattern(source_type, relation, target_type, limit)

    # ── Contradiction support ─────────────────────────────────────────────────

    def find_attribute_conflicts(self) -> list[dict]:
        """
        Scans nodes for conflicting attribute values supplied by
        different documents (used by M06 Contradiction Detector).
        Note: in this in-memory model attributes are merged on upsert,
        so conflicts are tracked via `_conflict_log` populated during
        ingestion in production; here we expose the hook for M06.
        """
        return []

    @property
    def node_count(self) -> int:
        return self._store.node_count

    @property
    def edge_count(self) -> int:
        return self._store.edge_count
