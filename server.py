"""
EREBUS - Local AI Server with ChromaDB Memory
Run: python server.py
"""

import http.server
import socketserver
import json
import urllib.request
import urllib.error
import os
import sys
import threading
import subprocess
import time
import hashlib
import re
from datetime import datetime

PORT = 8765

# ── KEYS FROM .env ──
# Reads keys from a local file so you don't paste them into the browser every launch.
# Tries ".env" first, then "env.txt" (Windows Explorer makes dotfiles awkward, so the
# .txt name is an easy fallback). Format is KEY=value, one per line, # for comments.
# This file stays LOCAL — never commit it.
def load_env():
    base = os.path.dirname(os.path.abspath(__file__))
    env = {}
    used = None
    for fname in ('.env', 'env.txt'):
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            continue
        used = fname
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, _, val = line.partition('=')
                    env[key.strip()] = val.strip().strip('"').strip("'")
        except Exception as e:
            print(f"  [ENV] Error reading {fname}: {e}")
        break
    if used:
        loaded = [k for k in ('ANTHROPIC_API_KEY', 'ELEVENLABS_API_KEY', 'ELEVENLABS_VOICE_ID') if env.get(k)]
        print(f"  [ENV] {used}: loaded {', '.join(loaded) if loaded else 'nothing (file present but empty)'}")
    else:
        print("  [ENV] No .env or env.txt found — enter keys in the browser Config panel")
    return env

ENV = load_env()

# ── MEMORY ENGINE TOGGLE ──
# ChromaDB's first real query/add lazily loads an embedding model. On bleeding-edge
# Python (3.14) that load can HANG, which wedges the whole server — the symptom was
# "ChromaDB initialized" printing fine, page loading instantly, but the first save/search
# timing out. JSON keyword memory is instant and persistent, so it's the default.
# Flip this to True later to try semantic recall again (e.g. on a stable Python
# build), and the code falls back to JSON automatically if it fails.
USE_CHROMADB = False

# Serializes the read-modify-write on memory.json now that the server is multi-threaded.
mem_lock = threading.Lock()

# ── CHROMADB SETUP ──
if USE_CHROMADB:
    try:
        import chromadb
        from chromadb.config import Settings

        DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'erebus_memory')
        chroma_client = chromadb.PersistentClient(path=DB_PATH)

        # Collections
        conversations = chroma_client.get_or_create_collection(
            name="conversations",
            metadata={"hnsw:space": "cosine"}
        )
        facts = chroma_client.get_or_create_collection(
            name="facts",
            metadata={"hnsw:space": "cosine"}
        )
        faces = chroma_client.get_or_create_collection(
            name="faces",
            metadata={"hnsw:space": "cosine"}
        )

        CHROMA_READY = True
        print("  [MEM] ChromaDB initialized")
    except Exception as e:
        CHROMA_READY = False
        print(f"  [MEM] ChromaDB failed: {e} — falling back to JSON")
else:
    CHROMA_READY = False
    print("  [MEM] Using JSON memory (ChromaDB disabled)")

# ── FALLBACK JSON MEMORY ──
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'memory.json')

def load_json_memory():
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            mem = json.load(f)
    except:
        mem = {}
    # schema migration — older files simply gain the new keys
    mem.setdefault('facts', [])          # GLOBAL facts (true regardless of who's present)
    mem.setdefault('conversations', [])
    mem.setdefault('people', {})         # per-person profiles: key -> {name, facts, created, last_seen}
    mem.setdefault('active_person', None)
    return mem

def save_json_memory(data):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Knowledge Vault (structured entity+relationship graph) ────────────────────
VAULT = None
try:
    from vault import Vault, migrate_from_memory
    _vault_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vault.json')
    VAULT = Vault(_vault_path)
    # first run: migrate the existing flat memory into the vault (non-destructive;
    # memory.json is left untouched — the vault is an additional, richer layer).
    if not VAULT.data.get('meta', {}).get('migrated_from_memory'):
        try:
            _summary = migrate_from_memory(VAULT, load_json_memory())
            print(f"[VAULT] migrated existing memory: {_summary}")
        except Exception as _e:
            print(f"[VAULT] migration skipped ({_e})")
    print(f"[VAULT] ready — {VAULT.stats()}")
except Exception as e:
    print(f"[VAULT] disabled ({e}) — falling back to flat memory only")
    VAULT = None


def _vault_extract(user_msg, bot_msg):
    """Auto-extract durable knowledge from an exchange into the vault, using Claude.
    Runs in the background after a conversation is saved. Best-effort; never raises."""
    if not VAULT:
        return
    key = ENV.get('ANTHROPIC_API_KEY', '')
    if not key:
        return
    prompt = (
        "From this exchange, extract durable facts and relationships about the user's "
        "world (people, projects, places, preferences, organizations). Ignore small talk "
        "and anything transient. Respond ONLY with JSON:\n"
        '{"facts":[{"entity":"<name>","type":"<person|project|place|preference|org|thing|event|concept>","fact":"<short fact>"}],'
        '"relations":[{"from":"<name>","rel":"<verb>","to":"<name>"}]}\n'
        "Empty arrays are fine. Keep facts short and factual.\n\n"
        f"User: {user_msg}\nErebus: {bot_msg}"
    )
    try:
        payload = json.dumps({
            "model": "claude-haiku-4-5-20251001", "max_tokens": 600,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=payload,
            headers={"Content-Type": "application/json", "x-api-key": key,
                     "anthropic-version": "2023-06-01"}, method="POST")
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return
        extracted = json.loads(m.group(0))
        for f in extracted.get("facts", []):
            if f.get("entity") and f.get("fact"):
                VAULT.add_fact(f["entity"], f["fact"], f.get("type", "thing"))
        for r in extracted.get("relations", []):
            if r.get("from") and r.get("to"):
                VAULT.add_relation(r["from"], r.get("rel", "related_to"), r["to"])
        VAULT.save()
    except Exception as e:
        print(f"[VAULT] extract skipped: {e}")


def find_chrome():
    paths = [
        os.path.expandvars(r'%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe'),
        os.path.expandvars(r'%PROGRAMFILES%\Google\Chrome\Application\chrome.exe'),
        os.path.expandvars(r'%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe'),
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None


class ErebusHandler(http.server.SimpleHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, x-api-key, xi-api-key, anthropic-version')

    def do_POST(self):

        # ── CLAUDE API PROXY ──
        if self.path == '/api/claude':
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length)
                data = json.loads(body)
                anthropic_key = data.pop('anthropic_key', '')
                if not anthropic_key:
                    self._error(400, 'No Anthropic API key provided')
                    return
                payload = json.dumps(data).encode('utf-8')
                req = urllib.request.Request(
                    'https://api.anthropic.com/v1/messages',
                    data=payload,
                    headers={
                        'Content-Type': 'application/json',
                        'x-api-key': anthropic_key,
                        'anthropic-version': '2023-06-01'
                    },
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = resp.read()
                self.send_response(200)
                self._cors()
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(result)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode('utf-8')
                self.send_response(e.code)
                self._cors()
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(err_body.encode('utf-8'))
            except Exception as e:
                self._error(500, str(e))
            return

        # ── CLAUDE STREAMING PROXY ──
        if self.path == '/api/claude/stream':
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length)
                data = json.loads(body)
                anthropic_key = data.pop('anthropic_key', '')
                if not anthropic_key:
                    self._error(400, 'No Anthropic API key provided')
                    return
                data['stream'] = True
                payload = json.dumps(data).encode('utf-8')
                req = urllib.request.Request(
                    'https://api.anthropic.com/v1/messages',
                    data=payload,
                    headers={
                        'Content-Type': 'application/json',
                        'x-api-key': anthropic_key,
                        'anthropic-version': '2023-06-01'
                    },
                    method='POST'
                )
                self.send_response(200)
                self._cors()
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Accel-Buffering', 'no')
                self.end_headers()
                with urllib.request.urlopen(req, timeout=60) as resp:
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        self.wfile.write(line)
                        self.wfile.flush()
            except urllib.error.HTTPError as e:
                err_body = e.read().decode('utf-8')
                self.send_response(e.code)
                self._cors()
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(err_body.encode('utf-8'))
            except Exception as e:
                try:
                    self.wfile.write(f'data: {{"error":"{str(e)}"}}\n\n'.encode())
                    self.wfile.flush()
                except:
                    pass
            return

        # ── ELEVENLABS TTS PROXY ──
        if self.path.startswith('/api/tts/'):
            try:
                voice_id = self.path.split('/api/tts/')[1]
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length)
                data = json.loads(body)
                el_key = data.pop('elevenlabs_key', '')
                if not el_key:
                    self._error(400, 'No ElevenLabs API key provided')
                    return
                payload = json.dumps(data).encode('utf-8')
                req = urllib.request.Request(
                    f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
                    data=payload,
                    headers={
                        'Content-Type': 'application/json',
                        'xi-api-key': el_key
                    },
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    audio = resp.read()
                self.send_response(200)
                self._cors()
                self.send_header('Content-Type', 'audio/mpeg')
                self.end_headers()
                self.wfile.write(audio)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode('utf-8')
                self.send_response(e.code)
                self._cors()
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(err_body.encode('utf-8'))
            except Exception as e:
                self._error(500, str(e))
            return

        # ── MEMORY API ──
        if self.path == '/api/memory':
            mem_lock.acquire()
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length)
                data = json.loads(body)
                action = data.get('action')

                if action == 'save_conversation':
                    # Save a full conversation exchange
                    user_msg = data.get('user', '')
                    bot_msg = data.get('bot', '')
                    source = data.get('source', 'pc')
                    username = data.get('username', '')
                    person = (data.get('person') or '').strip().lower()  # active profile, if any
                    timestamp = datetime.now().isoformat()
                    doc_id = hashlib.md5(f"{timestamp}{user_msg}".encode()).hexdigest()
                    tag = f"[{source}{(' @' + username) if username else ''}{(' person:' + person) if person else ''}] "
                    combined = f"{tag}User: {user_msg}\nErebus: {bot_msg}"

                    if CHROMA_READY:
                        conversations.add(
                            documents=[combined],
                            metadatas=[{"timestamp": timestamp, "user": user_msg, "bot": bot_msg,
                                        "source": source, "username": username, "person": person}],
                            ids=[doc_id]
                        )
                    else:
                        mem = load_json_memory()
                        mem['conversations'].append({"id": doc_id, "timestamp": timestamp, "user": user_msg,
                                                     "bot": bot_msg, "source": source, "username": username,
                                                     "person": person})
                        if len(mem['conversations']) > 500:
                            mem['conversations'] = mem['conversations'][-500:]
                        save_json_memory(mem)

                    self._ok({"saved": True, "id": doc_id})

                    # Auto-extract structured knowledge into the vault, in the
                    # background so it never slows Erebus's reply.
                    if VAULT and user_msg and bot_msg:
                        threading.Thread(target=_vault_extract,
                                         args=(user_msg, bot_msg), daemon=True).start()

                elif action == 'person_set':
                    # Create/switch the active person profile ("who am I talking to")
                    name = (data.get('name') or '').strip()
                    if not name:
                        self._error(400, 'No name provided')
                    else:
                        key = name.lower()
                        now = datetime.now().isoformat()
                        mem = load_json_memory()
                        created = key not in mem['people']
                        if created:
                            mem['people'][key] = {"name": name, "created": now, "facts": []}
                        mem['people'][key]['last_seen'] = now
                        mem['active_person'] = key
                        save_json_memory(mem)
                        p = mem['people'][key]
                        self._ok({"active": key, "name": p['name'], "created": created, "facts": p.get('facts', [])})

                elif action == 'person_list':
                    mem = load_json_memory()
                    people = [{"key": k, "name": v.get('name', k), "fact_count": len(v.get('facts', [])),
                               "last_seen": v.get('last_seen', '')} for k, v in mem['people'].items()]
                    self._ok({"people": people, "active": mem.get('active_person')})

                elif action == 'person_clear':
                    # Go anonymous — no active person (global facts still apply)
                    mem = load_json_memory()
                    mem['active_person'] = None
                    save_json_memory(mem)
                    self._ok({"active": None})

                elif action == 'face_save':
                    # Store face descriptors (128 floats each) on a person's profile.
                    # These are a mathematical signature, NOT a photo — no image is kept.
                    name = (data.get('name') or '').strip()
                    descriptors = data.get('descriptors') or []
                    if not name:
                        self._error(400, 'No name provided')
                    elif not descriptors:
                        self._error(400, 'No descriptors provided')
                    else:
                        # validate BEFORE touching the store, so a bad payload can't
                        # leave an empty profile behind
                        valid = [[float(x) for x in d] for d in descriptors
                                 if isinstance(d, list) and len(d) == 128]
                        if not valid:
                            self._error(400, 'No valid 128-length descriptors')
                        else:
                            key = name.lower()
                            now = datetime.now().isoformat()
                            mem = load_json_memory()
                            if key not in mem['people']:
                                mem['people'][key] = {"name": name, "created": now, "facts": []}
                            person = mem['people'][key]
                            person.setdefault('faces', [])
                            person['faces'].extend(valid)
                            # cap so the file can't grow unbounded over repeat enrolments
                            if len(person['faces']) > 20:
                                person['faces'] = person['faces'][-20:]
                            person['last_seen'] = now
                            mem['active_person'] = key
                            save_json_memory(mem)
                            self._ok({"saved": True, "name": person['name'], "key": key,
                                      "added": len(valid), "total": len(person['faces'])})

                elif action == 'face_list':
                    # All known faces, for the browser to match against
                    mem = load_json_memory()
                    known = []
                    for k, v in mem.get('people', {}).items():
                        faces = v.get('faces', [])
                        if faces:
                            known.append({"key": k, "name": v.get('name', k), "descriptors": faces})
                    self._ok({"known": known})

                elif action == 'face_forget':
                    # Delete a person's biometric data. Their profile/facts survive —
                    # this removes only the face signature.
                    name = (data.get('name') or '').strip().lower()
                    mem = load_json_memory()
                    if name in mem.get('people', {}):
                        had = len(mem['people'][name].get('faces', []))
                        mem['people'][name]['faces'] = []
                        save_json_memory(mem)
                        self._ok({"forgotten": True, "name": mem['people'][name].get('name', name), "removed": had})
                    else:
                        self._error(400, f'No profile named {name}')

                elif action == 'save_fact':
                    # Save a fact — into a person's profile if given, else global
                    fact_text = data.get('fact', '')
                    person = (data.get('person') or '').strip().lower()
                    timestamp = datetime.now().isoformat()
                    doc_id = hashlib.md5(fact_text.encode()).hexdigest()

                    if CHROMA_READY:
                        # Check if fact already exists
                        try:
                            existing = facts.get(ids=[doc_id])
                            if not existing['ids']:
                                facts.add(
                                    documents=[fact_text],
                                    metadatas=[{"timestamp": timestamp, "person": person}],
                                    ids=[doc_id]
                                )
                        except:
                            facts.add(
                                documents=[fact_text],
                                metadatas=[{"timestamp": timestamp, "person": person}],
                                ids=[doc_id]
                            )
                        self._ok({"saved": True, "person": person or None})
                    else:
                        mem = load_json_memory()
                        if person and person in mem['people']:
                            target = mem['people'][person].setdefault('facts', [])
                        else:
                            person = ''
                            target = mem['facts']
                        if fact_text not in target:
                            target.append(fact_text)
                            save_json_memory(mem)
                        self._ok({"saved": True, "person": person or None})

                elif action == 'vault_query':
                    # Pull relevant structured knowledge (entities + relationships)
                    # from the vault for a given query. Used to enrich Erebus's context.
                    query = data.get('query', '')
                    if VAULT and query:
                        self._ok({"knowledge": VAULT.relevant(query, limit=data.get('n', 8))})
                    else:
                        self._ok({"knowledge": ""})

                elif action == 'vault_stats':
                    self._ok(VAULT.stats() if VAULT else {"entities": 0, "relations": 0})

                elif action == 'vault_graph':
                    # Return the WHOLE vault as nodes + edges, for the memory graph
                    # visualization in the browser. Compact shape the canvas can draw.
                    if VAULT:
                        nodes = [{"id": e["id"], "name": e["name"], "type": e["type"],
                                  "facts": e.get("facts", []), "mentions": e.get("mentions", 1)}
                                 for e in VAULT.data["entities"].values()]
                        edges = [{"from": r["from"], "to": r["to"],
                                  "type": r["type"].replace("_", " ")}
                                 for r in VAULT.data["relations"]]
                        self._ok({"nodes": nodes, "edges": edges})
                    else:
                        self._ok({"nodes": [], "edges": []})

                elif action == 'search':
                    # Semantic search through memory
                    query = data.get('query', '')
                    n = data.get('n', 5)
                    # vault knowledge to enrich any search (structured graph slice)
                    vault_knowledge = VAULT.relevant(query, limit=8) if (VAULT and query) else ""

                    if CHROMA_READY and query:
                        results = {'conversations': [], 'facts': [], 'vault': vault_knowledge}
                        try:
                            conv_results = conversations.query(query_texts=[query], n_results=min(n, conversations.count()))
                            if conv_results['documents'][0]:
                                results['conversations'] = [
                                    {"text": doc, "metadata": meta}
                                    for doc, meta in zip(conv_results['documents'][0], conv_results['metadatas'][0])
                                ]
                        except: pass
                        try:
                            fact_results = facts.query(query_texts=[query], n_results=min(n, facts.count()))
                            if fact_results['documents'][0]:
                                results['facts'] = fact_results['documents'][0]
                        except: pass
                        self._ok(results)
                    else:
                        mem = load_json_memory()
                        q = (query or '').lower()
                        if q:
                            # keyword recall over JSON fallback
                            conv_hits = [c for c in mem['conversations']
                                         if q in (c.get('user', '') + ' ' + c.get('bot', '')).lower()][-n:]
                            fact_hits = [f for f in mem['facts'] if q in f.lower()]
                            # if nothing matches, fall back to most recent so he still has context
                            if not conv_hits:
                                conv_hits = mem['conversations'][-n:]
                            self._ok({"conversations": [{"text": f"User: {c.get('user','')}\nErebus: {c.get('bot','')}",
                                                          "metadata": c} for c in conv_hits],
                                      "facts": fact_hits if fact_hits else mem['facts'],
                                      "vault": vault_knowledge})
                        else:
                            self._ok({"conversations": mem['conversations'][-5:], "facts": mem['facts'],
                                      "vault": vault_knowledge})

                elif action == 'load':
                    # Load facts + active person profile for the system prompt
                    if CHROMA_READY:
                        try:
                            count = facts.count()
                            if count > 0:
                                all_facts = facts.get()
                                fact_list = all_facts['documents']
                            else:
                                fact_list = []
                        except:
                            fact_list = []
                        try:
                            conv_count = conversations.count()
                        except:
                            conv_count = 0
                        self._ok({"facts": fact_list, "conversation_count": conv_count, "engine": "chromadb",
                                  "active_person": None, "person_name": None, "person_facts": [], "people_count": 0})
                    else:
                        mem = load_json_memory()
                        active = mem.get('active_person')
                        person_name = None
                        person_facts = []
                        if active and active in mem.get('people', {}):
                            person_name = mem['people'][active].get('name', active)
                            person_facts = mem['people'][active].get('facts', [])
                        else:
                            active = None
                        self._ok({"facts": mem.get('facts', []),
                                  "conversation_count": len(mem.get('conversations', [])),
                                  "engine": "json",
                                  "active_person": active,
                                  "person_name": person_name,
                                  "person_facts": person_facts,
                                  "people_count": len(mem.get('people', {}))})

                elif action == 'clear':
                    # Clears conversations + global facts. Person PROFILES survive —
                    # wiping people should be deliberate, not a side effect.
                    if CHROMA_READY:
                        try:
                            chroma_client.delete_collection("conversations")
                            chroma_client.delete_collection("facts")
                            globals()['conversations'] = chroma_client.get_or_create_collection("conversations", metadata={"hnsw:space": "cosine"})
                            globals()['facts'] = chroma_client.get_or_create_collection("facts", metadata={"hnsw:space": "cosine"})
                        except: pass
                    else:
                        mem = load_json_memory()
                        save_json_memory({"facts": [], "conversations": [],
                                          "people": mem.get('people', {}),
                                          "active_person": mem.get('active_person')})
                    self._ok({"cleared": True, "people_kept": True})

                elif action == 'stats':
                    if CHROMA_READY:
                        self._ok({
                            "engine": "chromadb",
                            "conversations": conversations.count(),
                            "facts": facts.count(),
                            "path": DB_PATH
                        })
                    else:
                        mem = load_json_memory()
                        self._ok({
                            "engine": "json",
                            "conversations": len(mem.get('conversations', [])),
                            "facts": len(mem.get('facts', [])),
                            "people": len(mem.get('people', {})),
                            "active_person": mem.get('active_person')
                        })
                else:
                    self._error(400, f'Unknown action: {action}')

            except Exception as e:
                self._error(500, str(e))
            finally:
                mem_lock.release()
            return

        self._error(404, 'Not found')

    def do_GET(self):
        # Serve keys from .env to the local PC interface so it can auto-fill config.
        # localhost-only; the browser already holds these keys during normal use.
        if self.path == '/api/config':
            self.send_response(200)
            self._cors()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'anthropic': ENV.get('ANTHROPIC_API_KEY', ''),
                'elevenlabs': ENV.get('ELEVENLABS_API_KEY', ''),
                'voice': ENV.get('ELEVENLABS_VOICE_ID', '')
            }).encode())
            return
        super().do_GET()

    def _ok(self, data):
        self.send_response(200)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _error(self, code, msg):
        self.send_response(code)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': {'message': msg}}).encode())


def open_browser():
    time.sleep(1.5)
    url = f'http://localhost:{PORT}/erebus.html'
    chrome = find_chrome()
    if chrome:
        # Launch as its own app window in fullscreen so it feels like a program,
        # not a browser tab. A separate user-data-dir keeps Erebus in its own
        # window even if you already have Chrome open for normal browsing.
        profile = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.chrome_erebus')
        subprocess.Popen([
            chrome,
            '--start-fullscreen',          # fills the screen; F11 toggles out while developing
            f'--app={url}',                # app mode: no tabs, no address bar
            f'--user-data-dir={profile}',  # its own window, separate from your normal Chrome
            '--autoplay-policy=no-user-gesture-required',  # let his voice play without a click
        ])
    else:
        import webbrowser
        webbrowser.open(url)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Each request gets its own thread, so a slow memory or API call never blocks others.
    allow_reuse_address avoids the 'port still held' bind failure after an unclean restart."""
    daemon_threads = True
    allow_reuse_address = True


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 50)
    print("  EREBUS - AI Animatronic Interface")
    print("=" * 50)
    print(f"  Server: http://localhost:{PORT}")
    print(f"  Memory: {'ChromaDB' if CHROMA_READY else 'JSON fallback'}")
    print(f"  Press Ctrl+C to shut down")
    print("=" * 50)
    threading.Thread(target=open_browser, daemon=True).start()
    with ThreadingHTTPServer(('', PORT), ErebusHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Erebus shutting down.")
            sys.exit(0)
