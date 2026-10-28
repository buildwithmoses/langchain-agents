"""Write accepted examples to a LangSmith dataset. Idempotent: an example mined
from a run that is already in the dataset is skipped."""

from __future__ import annotations


def upsert(dataset_name: str, examples: list[dict], description: str = "", client=None) -> dict:
    from langsmith import Client

    client = client or Client()
    if client.has_dataset(dataset_name=dataset_name):
        ds = client.read_dataset(dataset_name=dataset_name)
    else:
        ds = client.create_dataset(dataset_name, description=description or "Built by evalkit from mined traces")
    existing = {(e.metadata or {}).get("source_run_id") for e in client.list_examples(dataset_id=ds.id)}
    new = [e for e in examples if e["metadata"].get("source_run_id") not in existing]
    if new:
        client.create_examples(dataset_id=ds.id, examples=new)
    return {"dataset": dataset_name, "dataset_id": str(ds.id), "added": len(new),
            "skipped_existing": len(examples) - len(new)}
