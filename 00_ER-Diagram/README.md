# 00 — Entity-Relationship Diagram

The data model for the platform. Defines the entities (stocks, news,
Reddit posts, market indicators), their attributes, and their
relationships — the shape that the later stages populate and query.

## Files

| File | Purpose |
| --- | --- |
| `ER.png` | Rendered entity-relationship diagram |
| `ER.puml` | PlantUML source (edit and re-render to update the diagram) |

## Why this is stage 0

Every data project should start with a model, not with code. The diagram
defines what data exists, how it connects, and what constraints apply —
before any schema is created, any pipeline is written, or any model is
trained.

## Rendering

```bash
plantuml -tpng ER.puml
