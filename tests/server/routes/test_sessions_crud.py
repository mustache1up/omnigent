"""Tests for Sessions API CRUD endpoints (list, get, delete, patch).

Exercises the core session management routes through the ``client``
fixture. Since the lifespan event (which seeds agents) does not run
in test fixtures, we seed a test agent and conversation directly via
the stores.
"""

from __future__ import annotations

import httpx
import pytest_asyncio

from omnigent.db.utils import generate_agent_id
from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from omnigent.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)


@pytest_asyncio.fixture()
async def session_id(db_uri: str) -> str:
    """Seed a test agent and conversation, return the session ID."""
    agent_store = SqlAlchemyAgentStore(db_uri)
    conv_store = SqlAlchemyConversationStore(db_uri)
    agent_id = generate_agent_id()
    agent_store.create(agent_id, name="test-agent", bundle_location="test:///bundle")
    conv = conv_store.create_conversation(agent_id=agent_id)
    return conv.id


# ── GET /v1/sessions (list) ─────────────────────────────────────────


async def test_list_sessions_empty(client: httpx.AsyncClient) -> None:
    """Empty database returns an empty list."""
    resp = await client.get("/v1/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == []
    assert body["has_more"] is False


async def test_list_sessions_after_create(
    client: httpx.AsyncClient,
    session_id: str,
) -> None:
    """A created session appears in the list."""
    resp = await client.get("/v1/sessions")
    assert resp.status_code == 200
    body = resp.json()
    ids = [s["id"] for s in body["data"]]
    assert session_id in ids


async def test_list_sessions_pagination(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    """Pagination with limit returns at most N sessions."""
    agent_store = SqlAlchemyAgentStore(db_uri)
    conv_store = SqlAlchemyConversationStore(db_uri)
    agent_id = generate_agent_id()
    agent_store.create(agent_id, name="pag-agent", bundle_location="test:///bundle")
    conv_store.create_conversation(agent_id=agent_id)
    conv_store.create_conversation(agent_id=agent_id)

    resp = await client.get("/v1/sessions?limit=1")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 1


# ── GET /v1/sessions/{id} (get snapshot) ────────────────────────────


async def test_get_session(
    client: httpx.AsyncClient,
    session_id: str,
) -> None:
    """Get a session by ID returns its snapshot."""
    resp = await client.get(f"/v1/sessions/{session_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == session_id


async def test_get_session_not_found(client: httpx.AsyncClient) -> None:
    """Getting a nonexistent session returns 404."""
    resp = await client.get("/v1/sessions/conv_nonexistent_12345")
    assert resp.status_code == 404


# ── DELETE /v1/sessions/{id} ────────────────────────────────────────


async def test_delete_session(
    client: httpx.AsyncClient,
    session_id: str,
) -> None:
    """Deleting a session returns 200 with deleted: true."""
    resp = await client.delete(f"/v1/sessions/{session_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted"] is True


async def test_delete_session_not_found(client: httpx.AsyncClient) -> None:
    """Deleting a nonexistent session returns 404."""
    resp = await client.delete("/v1/sessions/conv_nonexistent_12345")
    assert resp.status_code == 404


# ── PATCH /v1/sessions/{id} ─────────────────────────────────────────


async def test_patch_session_title(
    client: httpx.AsyncClient,
    session_id: str,
) -> None:
    """Patching a session's title returns the updated session."""
    resp = await client.patch(
        f"/v1/sessions/{session_id}",
        json={"title": "New Title"},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200


async def test_patch_session_not_found(client: httpx.AsyncClient) -> None:
    """Patching a nonexistent session returns 404."""
    resp = await client.patch(
        "/v1/sessions/conv_nonexistent_12345",
        json={"title": "New Title"},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 404


# ── GET /v1/sessions/collections ────────────────────────────────────────


async def test_list_collections_empty(client: httpx.AsyncClient) -> None:
    """No collection labels anywhere → empty collection list."""
    resp = await client.get("/v1/sessions/collections")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_collections_returns_names_sorted(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    """Collections surface as a sorted list of names."""
    conv_store = SqlAlchemyConversationStore(db_uri)
    a = conv_store.create_conversation()
    b = conv_store.create_conversation()
    conv_store.set_labels(a.id, {"collection": "Sprint 42"})
    conv_store.set_labels(b.id, {"collection": "Customer X"})

    resp = await client.get("/v1/sessions/collections")
    assert resp.status_code == 200
    assert resp.json() == ["Customer X", "Sprint 42"]


# ── GET /v1/sessions?collection= (filter) ───────────────────────────────


async def test_list_sessions_filtered_by_collection(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    """``?collection=X`` returns only sessions in that collection."""
    agent_store = SqlAlchemyAgentStore(db_uri)
    conv_store = SqlAlchemyConversationStore(db_uri)
    # GET /v1/sessions filters has_agent_id=True, so bind the conversations to
    # a seeded agent — otherwise the list comes back empty.
    agent_id = generate_agent_id()
    agent_store.create(agent_id, name="collection-agent", bundle_location="test:///bundle")
    filed = conv_store.create_conversation(agent_id=agent_id)
    conv_store.create_conversation(agent_id=agent_id)  # unfiled
    conv_store.set_labels(filed.id, {"collection": "X"})

    resp = await client.get("/v1/sessions?collection=X")
    assert resp.status_code == 200
    ids = [s["id"] for s in resp.json()["data"]]
    assert ids == [filed.id]


async def test_list_sessions_empty_collection_returns_unfiled(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    """``?collection=`` (empty) returns only sessions with no collection label."""
    agent_store = SqlAlchemyAgentStore(db_uri)
    conv_store = SqlAlchemyConversationStore(db_uri)
    agent_id = generate_agent_id()
    agent_store.create(agent_id, name="collection-agent", bundle_location="test:///bundle")
    filed = conv_store.create_conversation(agent_id=agent_id)
    unfiled = conv_store.create_conversation(agent_id=agent_id)
    conv_store.set_labels(filed.id, {"collection": "X"})

    resp = await client.get("/v1/sessions?collection=")
    assert resp.status_code == 200
    ids = [s["id"] for s in resp.json()["data"]]
    assert unfiled.id in ids
    assert filed.id not in ids


# ── PATCH /v1/sessions/{id} collection label ────────────────────────────


async def test_patch_session_sets_collection_label(
    client: httpx.AsyncClient,
    session_id: str,
    db_uri: str,
) -> None:
    """PATCH with ``labels: {collection: X}`` upserts the collection label."""
    resp = await client.patch(
        f"/v1/sessions/{session_id}",
        json={"labels": {"collection": "Sprint 42"}},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200

    conv_store = SqlAlchemyConversationStore(db_uri)
    conv = conv_store.get_conversation(session_id)
    assert conv is not None
    assert conv.labels.get("collection") == "Sprint 42"


async def test_patch_session_empty_collection_removes_label(
    client: httpx.AsyncClient,
    session_id: str,
    db_uri: str,
) -> None:
    """PATCH with ``labels: {collection: ""}`` removes the collection label rather
    than persisting an empty value — so the session returns to Unfiled."""
    conv_store = SqlAlchemyConversationStore(db_uri)
    conv_store.set_labels(session_id, {"collection": "Sprint 42"})

    resp = await client.patch(
        f"/v1/sessions/{session_id}",
        json={"labels": {"collection": ""}},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200

    conv = conv_store.get_conversation(session_id)
    assert conv is not None
    assert "collection" not in conv.labels
