import os
from typing import Any, Dict, List

from auth import verify_auth
from fastapi import APIRouter, Depends, HTTPException

NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

router = APIRouter(prefix="/memories", tags=["related"])

_driver = None


def _get_driver():
    global _driver
    if not NEO4J_URI:
        raise HTTPException(status_code=503, detail="Relation store is not configured (NEO4J_URI unset).")
    if _driver is None:
        from neo4j import GraphDatabase

        _driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", NEO4J_PASSWORD))
    return _driver


@router.get("/{memory_id}/related", summary="Memories related through the relation graph")
def related_memories(memory_id: str, depth: int = 1, _auth=Depends(verify_auth)) -> Dict[str, Any]:
    """Neighbours of a memory in the relation graph. depth 1 returns direct edges with
    type/direction/confidence; depth 2-3 walks paths and returns the closest hit per node."""
    depth = max(1, min(depth, 3))
    driver = _get_driver()
    results: List[Dict[str, Any]] = []
    with driver.session() as session:
        if depth == 1:
            records = session.run(
                "MATCH (m:Memory {id: $id})-[r]-(n:Memory) "
                "RETURN n.id AS id, n.snippet AS snippet, n.kind AS kind, n.project AS project, "
                "n.expired AS expired, type(r) AS relation, startNode(r).id = $id AS outgoing, "
                "r.confidence AS confidence ORDER BY r.confidence DESC LIMIT 50",
                id=memory_id,
            )
            results = [dict(rec) for rec in records]
        else:
            records = session.run(
                "MATCH p = (m:Memory {id: $id})-[*1..%d]-(n:Memory) WHERE n.id <> $id "
                "WITH n, [rel IN relationships(p) | type(rel)] AS path, length(p) AS distance "
                "ORDER BY distance ASC WITH n, collect({path: path, distance: distance})[0] AS best "
                "RETURN n.id AS id, n.snippet AS snippet, n.kind AS kind, n.project AS project, "
                "n.expired AS expired, best.distance AS distance, best.path AS path LIMIT 50" % depth,
                id=memory_id,
            )
            results = [dict(rec) for rec in records]
    return {"memory_id": memory_id, "depth": depth, "results": results}
