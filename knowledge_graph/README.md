# Diabetes Knowledge Graph

This Neo4j graph documents the intelligent system rather than asserting medical knowledge. Every stored fact comes from the final Diabetes notebook or the fitted pipeline: the selected model configuration, six raw features, preprocessing steps, target labels, training-selection experiment, held-out metrics, and fitted Random Forest feature importances.

## Schema

Node labels: `System`, `Model`, `Feature`, `Target`, `PipelineStep`, `Dataset`, `Experiment`, and `Metric`. All nodes also carry `AssignmentEntity` and `domain = "diabetes_assignment_01"` so API queries stay scoped.

Relationships: `USES_MODEL`, `USES_FEATURE`, `PREDICTS`, `HAS_PIPELINE_STEP`, `TRAINED_ON`, `EVALUATED_BY`, `SELECTED`, and `COMPARES_REPRESENTATION`.

`USES_FEATURE.importance` stores exact fitted impurity-based importance values. These values describe this model and do not imply causality or clinical importance.

## Seed locally

From the repository root, after Neo4j is healthy:

```powershell
docker compose run --rm neo4j-seed
```

The script uses `MERGE` and is safe to run repeatedly. Verify in Neo4j Browser at <http://localhost:7474>:

```cypher
MATCH (n {domain: 'diabetes_assignment_01'})
OPTIONAL MATCH (n)-[r]->()
RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS relationships;
```
