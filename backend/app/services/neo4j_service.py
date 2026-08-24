from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.core.config import Settings


logger = logging.getLogger(__name__)


class KnowledgeGraphUnavailable(RuntimeError):
    pass


class Neo4jService:
    def __init__(self, app_settings: Settings) -> None:
        self.settings = app_settings
        self.driver: Any | None = None
        self.status = "disabled" if not app_settings.neo4j_enabled else "unavailable"

    def connect(self) -> None:
        if not self.settings.neo4j_enabled:
            return
        try:
            self.driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
                connection_timeout=2,
            )
            self.driver.verify_connectivity()
            self.status = "available"
            logger.info("Neo4j connected: uri=%s", self.settings.neo4j_uri)
        except Exception as exc:
            self.status = "unavailable"
            if self.driver is not None:
                self.driver.close()
                self.driver = None
            logger.warning("Neo4j unavailable; prediction APIs remain operational: %s", exc)

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def get_diabetes_graph(self) -> dict[str, list[dict[str, Any]]]:
        if self.driver is None or self.status != "available":
            raise KnowledgeGraphUnavailable(
                "Neo4j is unavailable. Start it and seed the Diabetes knowledge graph."
            )
        query = """
        MATCH (n)
        WHERE n.domain = 'diabetes_assignment_01'
        OPTIONAL MATCH (n)-[r]->(m)
        WHERE m.domain = 'diabetes_assignment_01'
        RETURN collect(DISTINCT {
            id: elementId(n), labels: labels(n), properties: properties(n)
        }) AS leftNodes,
        collect(DISTINCT CASE WHEN m IS NULL THEN null ELSE {
            id: elementId(m), labels: labels(m), properties: properties(m)
        } END) AS rightNodes,
        collect(DISTINCT CASE WHEN r IS NULL THEN null ELSE {
            id: elementId(r), source: elementId(startNode(r)),
            target: elementId(endNode(r)), type: type(r), properties: properties(r)
        } END) AS edges
        """
        try:
            record = self.driver.execute_query(query).records[0]
            nodes_by_id: dict[str, dict[str, Any]] = {}
            for node in [*record["leftNodes"], *record["rightNodes"]]:
                if node is not None:
                    nodes_by_id[node["id"]] = node
            return {
                "nodes": list(nodes_by_id.values()),
                "edges": [edge for edge in record["edges"] if edge is not None],
            }
        except (Neo4jError, ServiceUnavailable) as exc:
            self.status = "unavailable"
            raise KnowledgeGraphUnavailable("Neo4j query failed") from exc

    def seed_from_file(self, cypher_path: Path) -> None:
        if self.driver is None or self.status != "available":
            raise KnowledgeGraphUnavailable("Neo4j is unavailable")
        statements = [statement.strip() for statement in cypher_path.read_text(encoding="utf-8").split(";")]
        for statement in statements:
            if statement:
                self.driver.execute_query(statement)
