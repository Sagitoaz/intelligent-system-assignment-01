from __future__ import annotations

from app.core.config import settings
from app.services.neo4j_service import Neo4jService


def main() -> None:
    service = Neo4jService(settings)
    service.connect()
    try:
        cypher_path = settings.project_root / "knowledge_graph" / "cypher" / "diabetes_knowledge_graph.cypher"
        service.seed_from_file(cypher_path)
        graph = service.get_diabetes_graph()
        print(f"Knowledge graph seeded: {len(graph['nodes'])} nodes, {len(graph['edges'])} relationships")
    finally:
        service.close()


if __name__ == "__main__":
    main()
