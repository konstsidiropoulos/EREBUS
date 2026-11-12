"""
EREBUS - Knowledge Vault
========================
A structured knowledge graph that replaces the flat fact list with something that
actually models Konstantinos's world: ENTITIES (people, projects, places,
preferences, things) connected by RELATIONSHIPS ("Konstantinos studies-at TU
Sofia"). Facts attach to entities. When Erebus talks, the RELEVANT slice of the
graph is pulled into his context — so he draws on what matters without carrying
all 400 facts every message.

Design goals:
- Dependency-free (plain JSON on disk, like the rest of Erebus). No database.
- Non-destructive: it IMPORTS the existing memory.json rather than replacing it.
- Safe: every operation is defensive; a corrupt vault never bricks the server.

Shape on disk (vault.json):
{
  "entities": {
     "<id>": {"id","name","type","facts":[...],"aliases":[...],"created","updated","mentions"}
  },
  "relations": [
     {"from":"<id>","type":"<verb>","to":"<id>","note":"","created"}
  ],
  "meta": {"version":1, "migrated_from_memory": true}
}

Entity types: person, project, place, preference, org, thing, event, concept.
"""

import json
import os
import re
import time
import unicodedata

ENTITY_TYPES = ("person", "project", "place", "preference",
                "org", "thing", "event", "concept")


class Vault:
    def __init__(self, path):
        self.path = path
        self.data = {"entities": {}, "relations": [], "meta": {"version": 1}}
        self.load()

    # ── persistence ───────────────────────────────────────────────────────────
    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.data = {
                    "entities": d.get("entities", {}),
                    "relations": d.get("relations", []),
                    "meta": d.get("meta", {"version": 1}),
                }
            except Exception as e:
                print(f"[VAULT] load failed ({e}); starting fresh")

    def save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)   # atomic
        except Exception as e:
            print(f"[VAULT] save failed: {e}")

    # ── entity helpers ────────────────────────────────────────────────────────
    def _norm(self, s):
        s = (s or "").strip().lower()
        s = "".join(c for c in unicodedata.normalize("NFKD", s)
                    if not unicodedata.combining(c))
        return re.sub(r"\s+", " ", s)

    def _make_id(self, name, etype):
        base = re.sub(r"[^a-z0-9]+", "_", self._norm(name)).strip("_") or "entity"
        cand = f"{etype}:{base}"
        i = 2
        while cand in self.data["entities"] and self.data["entities"][cand]["name"].lower() != name.lower():
            cand = f"{etype}:{base}_{i}"
            i += 1
        return cand

    def find_entity(self, name, etype=None):
        """Find an entity by name or alias (case/accent-insensitive)."""
        n = self._norm(name)
        for eid, e in self.data["entities"].items():
            if etype and e["type"] != etype:
                continue
            names = [e["name"]] + e.get("aliases", [])
            if any(self._norm(x) == n for x in names):
                return eid
        return None

    def upsert_entity(self, name, etype="thing", aliases=None):
        """Create or fetch an entity. Returns its id."""
        name = (name or "").strip()
        if not name:
            return None
        if etype not in ENTITY_TYPES:
            etype = "thing"
        eid = self.find_entity(name)   # match across any type first
        if eid:
            e = self.data["entities"][eid]
            e["mentions"] = e.get("mentions", 0) + 1
            e["updated"] = time.time()
            for a in (aliases or []):
                if a and self._norm(a) not in [self._norm(x) for x in [e["name"]] + e.get("aliases", [])]:
                    e.setdefault("aliases", []).append(a)
            return eid
        eid = self._make_id(name, etype)
        self.data["entities"][eid] = {
            "id": eid, "name": name, "type": etype,
            "facts": [], "aliases": aliases or [],
            "created": time.time(), "updated": time.time(), "mentions": 1,
        }
        return eid

    def add_fact(self, entity_name, fact, etype="thing"):
        """Attach a fact to an entity (creating the entity if needed)."""
        fact = (fact or "").strip()
        if not fact:
            return None
        eid = self.upsert_entity(entity_name, etype)
        e = self.data["entities"][eid]
        # avoid dupes
        if fact not in e["facts"]:
            e["facts"].append(fact)
            e["updated"] = time.time()
        return eid

    def add_relation(self, from_name, rel_type, to_name,
                     from_type="thing", to_type="thing", note=""):
        """Connect two entities: from --rel_type--> to."""
        fid = self.upsert_entity(from_name, from_type)
        tid = self.upsert_entity(to_name, to_type)
        if not fid or not tid:
            return False
        if fid == tid:
            return False   # never create a self-relation
        rel_type = (rel_type or "related_to").strip().lower().replace(" ", "_")
        for r in self.data["relations"]:
            if r["from"] == fid and r["to"] == tid and r["type"] == rel_type:
                return True   # already exists
        self.data["relations"].append({
            "from": fid, "type": rel_type, "to": tid,
            "note": note, "created": time.time(),
        })
        return True

    # ── retrieval ─────────────────────────────────────────────────────────────
    def relevant(self, query, limit=8):
        """Return the entities most relevant to `query`, with their facts and the
        relationships that touch them. This is what gets injected into context."""
        q = self._norm(query)
        q_tokens = set(t for t in re.split(r"[^a-z0-9]+", q) if len(t) > 2)
        scored = []
        for eid, e in self.data["entities"].items():
            score = 0
            names = [e["name"]] + e.get("aliases", [])
            n_norm = " ".join(self._norm(x) for x in names)
            # direct name mention is strong
            for name in names:
                if self._norm(name) and self._norm(name) in q:
                    score += 10
            # token overlap with name + facts
            hay = n_norm + " " + " ".join(self._norm(f) for f in e.get("facts", []))
            hay_tokens = set(re.split(r"[^a-z0-9]+", hay))
            score += len(q_tokens & hay_tokens)
            # recency / salience nudge
            score += min(e.get("mentions", 0), 5) * 0.1
            if score > 0:
                scored.append((score, eid))
        scored.sort(reverse=True)
        top = [eid for _, eid in scored[:limit]]
        return self._render(top)

    def _render(self, entity_ids):
        """Render a set of entities + their relations as a compact text block."""
        if not entity_ids:
            return ""
        idset = set(entity_ids)
        lines = []
        for eid in entity_ids:
            e = self.data["entities"].get(eid)
            if not e:
                continue
            head = f"{e['name']} ({e['type']})"
            if e.get("facts"):
                head += ": " + "; ".join(e["facts"][:6])
            lines.append(head)
        # relations among the selected entities (and to their direct neighbors)
        rel_lines = []
        for r in self.data["relations"]:
            if r["from"] in idset or r["to"] in idset:
                fe = self.data["entities"].get(r["from"])
                te = self.data["entities"].get(r["to"])
                if fe and te:
                    rel_lines.append(f"{fe['name']} {r['type'].replace('_',' ')} {te['name']}")
        out = "\n".join(f"- {l}" for l in lines)
        if rel_lines:
            # dedupe, cap
            seen = []
            for rl in rel_lines:
                if rl not in seen:
                    seen.append(rl)
            out += "\nConnections:\n" + "\n".join(f"- {l}" for l in seen[:10])
        return out

    def stats(self):
        return {
            "entities": len(self.data["entities"]),
            "relations": len(self.data["relations"]),
            "by_type": {t: sum(1 for e in self.data["entities"].values() if e["type"] == t)
                        for t in ENTITY_TYPES if any(e["type"] == t for e in self.data["entities"].values())},
        }


# ── migration from the old flat memory.json ───────────────────────────────────
def migrate_from_memory(vault: Vault, memory: dict):
    """Import the existing flat memory into the vault, non-destructively. Global
    facts become facts on a 'Konstantinos' person entity (or 'general' concept);
    per-person profiles become person entities with their facts. Returns a summary."""
    imported = {"facts": 0, "people": 0}

    # the owner
    owner_id = vault.upsert_entity("Konstantinos", "person",
                                   aliases=["Kostas", "Kostakis", "user", "owner"])

    # global facts -> attach to owner (they're things true about his world)
    for f in memory.get("facts", []):
        f = f.strip() if isinstance(f, str) else str(f)
        if f:
            vault.add_fact("Konstantinos", f, "person")
            imported["facts"] += 1

    # per-person profiles -> person entities
    owner_eid = vault.find_entity("Konstantinos", "person")
    for key, prof in (memory.get("people", {}) or {}).items():
        name = prof.get("name", key)
        pid = vault.upsert_entity(name, "person")
        for f in prof.get("facts", []):
            if f:
                vault.add_fact(name, f, "person")
                imported["facts"] += 1
        # relation: owner knows this person — but never a self-relation (skip if this
        # profile resolved to the owner entity itself, e.g. via an alias).
        if pid and pid != owner_eid:
            vault.add_relation("Konstantinos", "knows", name, "person", "person")
        imported["people"] += 1

    # seed the known project structure so the graph starts rich
    vault.upsert_entity("Erebus", "project", aliases=["the robot", "the companion"])
    vault.add_relation("Konstantinos", "builds", "Erebus", "person", "project")

    vault.data["meta"]["migrated_from_memory"] = True
    vault.save()
    return imported


if __name__ == "__main__":
    # self-test
    import tempfile
    d = tempfile.mkdtemp()
    v = Vault(os.path.join(d, "vault.json"))

    print("=== build a small graph ===")
    v.add_fact("Konstantinos", "is a master's student at TU Sofia", "person")
    v.add_fact("Konstantinos", "lives in Sofia, Bulgaria", "person")
    v.add_relation("Konstantinos", "studies_at", "TU Sofia", "person", "org")
    v.add_relation("Konstantinos", "builds", "Erebus", "person", "project")
    v.add_fact("Erebus", "is a monochrome animatronic companion", "project")
    v.save()
    print(v.stats())

    print("\n=== retrieval: 'tell me about erebus' ===")
    print(v.relevant("tell me about erebus"))

    print("\n=== retrieval: 'where does konstantinos study' ===")
    print(v.relevant("where does konstantinos study"))

    print("\n=== migration test ===")
    fake_mem = {"facts": ["likes strong coffee", "drives a Porsche"],
                "people": {"maria": {"name": "Maria", "facts": ["is his sister"]}}}
    v2 = Vault(os.path.join(d, "vault2.json"))
    print("imported:", migrate_from_memory(v2, fake_mem))
    print(v2.stats())
    print(v2.relevant("who is maria"))
