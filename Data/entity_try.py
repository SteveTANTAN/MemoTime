import os, re, json, argparse, sqlite3
from typing import List, Tuple, Dict, Any, Iterable, Set, Optional
import numpy as np

# ============== optional: FAISS ==============
try:
    import faiss  # pip install faiss-cpu
except Exception:
    faiss = None

# ============== OpenAI client ==============
try:
    from openai import OpenAI
    _openai_client = None
    def _client():
        global _openai_client
        if _openai_client is None:
            from memotime.kg_agent.llm import DEFAULT_OPENAI_API_KEY
            api_key = os.getenv("OPENAI_API_KEY") or DEFAULT_OPENAI_API_KEY
            _openai_client = OpenAI(api_key=api_key)
        return _openai_client
except Exception:
    OpenAI = None
    def _client():
        raise RuntimeError("openai package not installed. pip install openai")

EMBED_MODEL = "text-embedding-3-large"
RERANK_MODEL = "gpt-4o-mini"

# ============== SQLite basics ==============
def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def fetch_entities(db_path: str) -> List[Tuple[int, str]]:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM entities ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()
    return rows  # [(id, name), ...]

    # ============== 3-hop connected and subgraph ==============
def _neighbors(conn: sqlite3.Connection, eid: int) -> List[int]:
    cur = conn.cursor()
    cur.execute("SELECT tail_id FROM edges WHERE head_id=?", (eid,))
    outs = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT head_id FROM edges WHERE tail_id=?", (eid,))
    ins = [r[0] for r in cur.fetchall()]
    return list(set(outs + ins))

def within_k(db_path: str, ids: List[int], k: int = 3) -> bool:
    if len(ids) <= 1:
        return True
    conn = _connect(db_path)
    seen: Set[int] = set()
    frontier: Set[int] = {ids[0]}
    for _ in range(k + 1):
        if not frontier:
            break
        nxt: Set[int] = set()
        for u in list(frontier):
            if u in seen:
                continue
            seen.add(u)
            for v in _neighbors(conn, u):
                if v not in seen:
                    nxt.add(v)
        frontier = nxt
    conn.close()
    return all(x in seen for x in ids[1:])

def extract_3hop_subgraph(db_path: str, seeds: Iterable[int], k: int = 3) -> List[Tuple[int, str, int, str]]:
    conn = _connect(db_path)
    visited: Set[int] = set()
    frontier: Set[int] = set(seeds)
    depth = 0
    while frontier and depth <= k:
        nxt: Set[int] = set()
        for u in frontier:
            if u in visited:
                continue
            visited.add(u)
            for v in _neighbors(conn, u):
                if v not in visited:
                    nxt.add(v)
        frontier = nxt
        depth += 1
    cur = conn.cursor()
    if not visited:
        conn.close()
        return []
    marks = ",".join("?" for _ in visited)
    params = tuple(visited)
    cur.execute(
        f"SELECT head_id, relation, tail_id, time FROM edges "
        f"WHERE head_id IN ({marks}) OR tail_id IN ({marks})",
        params + params
    )
    rows = cur.fetchall()
    conn.close()
    return rows  # [(h, r, t, time), ...]

# ============== text normalization and heuristic keywords ==============
_STOPWORDS = {
    "the","a","an","of","and","or","to","in","on","for","with","by","at","from",
    "as","is","was","were","be","been","being","who","whom","which","that","this",
    "these","those","after","before","first","last","when","what","where","why",
    "how","did","do","does","done","between","into"
}

def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalized_key(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^0-9a-z\s\-\_()/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def heuristic_keywords_stronger(question: str, max_len: int = 8) -> List[str]:
    q = question.strip()
    spans = set()

    # continuous uppercase开头词（允许少量小词）
    for m in re.finditer(r"\b([A-Z][a-zA-Z']+(?:\s+(?:of|and|the|to|for|in|on|&)\s+)?[A-Z][a-zA-Z'()\-]+(?:\s+[A-Z][a-zA-Z'()\-]+){0,6})\b", q):
        spans.add(normalize_text(m.group(0)))

    # organization suffix extension
    org_suffix = r"(Ministry|Council|Cabinet|Department|Government|University|Commission|Office|Party|Bank|Company|Corporation|Agency|Bureau|Authority|Committee|Parliament|Assembly|Embassy|Consulate)"
    for m in re.finditer(rf"\b([A-Z][\w()\-']+(?:\s+[A-Z][\w()\-']+){{0,7}}\s+{org_suffix}\b)", q):
        spans.add(normalize_text(m.group(1)))

    # comma inversion: X, Country  → X；X Country
    for m in re.finditer(r"\b([A-Z][\w()\-']+(?:\s+[A-Z][\w()\-']+)*)\s*,\s*([A-Z][\w()\-']+)\b", q):
        a, b = normalize_text(m.group(1)), normalize_text(m.group(2))
        spans.add(a); spans.add(f"{a} {b}")

    # bracket subtitle: A (B) → A, B, A B
    for m in re.finditer(r"\b([A-Z][\w\-']+(?:\s+[A-Z][\w\-']+)*)\s*\(([^)]+)\)", q):
        a, b = normalize_text(m.group(1)), normalize_text(m.group(2))
        spans.add(a); spans.add(b); spans.add(f"{a} {b}")

    # person name: first name-last name/with middle name
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+)\b", q):
        spans.add(normalize_text(m.group(0)))

    # window extension
    tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z()\-']+", q) if t.lower() not in _STOPWORDS]
    for i in range(len(tokens)):
        if tokens[i][0].isupper():
            spans.add(tokens[i])
            for w in (2,3,4):
                if i+w <= len(tokens):
                    frag = " ".join(tokens[i:i+w])
                    if sum(tok[0].isupper() for tok in tokens[i:i+w]) >= 1:
                        spans.add(frag)

    spans = {s.strip(" ?!.") for s in spans if len(s.split()) <= max_len}
    bag, seen = [], set()
    for s in spans:
        k = normalized_key(s)
        if k and k not in seen:
            seen.add(k); bag.append(s)
    return bag

# ============== LLM keywords ==============
def llm_keywords(question: str, max_kw: int = 8) -> List[str]:
    cli = _client()
    sys = (
        "Extract entity-like keywords or aliases from the question. "
        "Return JSON with key 'keywords' as an array. No extra text. "
        "For example, question: 'After the Danish Ministry of Defence and Security, who was the first to visit Iraq?' -> keywords: ['Danish Ministry of Defence and Security', 'Iraq']"
    )

    
    user = {"question": question, "max": max_kw}
    resp = cli.chat.completions.create(
        model=RERANK_MODEL,
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":json.dumps(user, ensure_ascii=False)}],
        temperature=0,
        response_format={"type":"json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        kws = data.get("keywords", [])
        out, seen = [], set()
        for k in kws:
            k = re.sub(r"\s+", " ", str(k)).strip().strip(".,;:!?")
            nk = normalized_key(k)
            if nk and nk not in seen:
                seen.add(nk); out.append(k)
        return out[:max_kw]
    except Exception:
        return []

def combined_keywords(question: str, llm_max: int = 8, heur_max_len: int = 8) -> List[str]:
    try:
        kw_llm = llm_keywords(question, max_kw=llm_max) or []
    except Exception:
        kw_llm = []
    kw_heur = heuristic_keywords_stronger(question, max_len=heur_max_len)
    bag, seen = [], set()
    for k in kw_llm + kw_heur:
        k = re.sub(r"\s+", " ", str(k)).strip().strip(".,;:!?")
        nk = normalized_key(k)
        if nk and nk not in seen:
            seen.add(nk); bag.append(k)
    return bag

# ============== literal hit (problem原文直接匹配实体) ==============
def _norm_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s\-_]+", " ", s)
    s = re.sub(r"[^\w\s():/]", "", s)
    return s

def _trigram_set(s: str):
    s = _norm_name(s).replace(" ", "")
    if len(s) <= 2:
        return {s}
    return {s[i:i+3] for i in range(len(s)-2)}

def _name_sim(a: str, b: str) -> float:
    A, B = _trigram_set(a), _trigram_set(b)
    if not A or not B:
        return 0.0
    j = len(A & B) / len(A | B)
    l = min(len(a), len(b)) / max(len(a), len(b))
    return 0.7 * j + 0.3 * l

def literal_hit_entities(db_path: str, question: str, sim_th: float = 0.85, span_only: bool = True):
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM entities ORDER BY id ASC")
    ents = cur.fetchall()
    conn.close()

    q_raw = question
    q_norm = _norm_name(question)

    spans = []
    if span_only:
        for m in re.finditer(r"\b([A-Z][a-zA-Z']+(?:\s+(?:of|and|the|to|for|in|on)\s+)?[A-Z][a-zA-Z'()\-]+(?:\s+[A-Z][a-zA-Z'()\-]+){0,4})\b", q_raw):
            spans.append(m.group(0).strip())
        spans.append(q_raw)
    else:
        spans = [q_raw]

    hits = []
    seen = set()
    for eid, name in ents:
        ok = False
        if name.lower() in q_raw.lower():
            ok = True
        else:
            for sp in spans:
                if name.lower() in sp.lower():
                    ok = True; break
                if _name_sim(name, sp) >= sim_th:
                    ok = True; break
            if not ok and _name_sim(name, q_norm) >= sim_th:
                ok = True
        if ok and eid not in seen:
            seen.add(eid)
            hits.append((eid, name))
    return hits

# ============== OpenAI embedding and reranking ==============
def openai_embed(texts: List[str]) -> np.ndarray:
    cli = _client()
    out = []
    for i in range(0, len(texts), 256):
        batch = texts[i:i+256]
        resp = cli.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([v.embedding for v in resp.data])
    arr = np.array(out, dtype=np.float32)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    return arr

def llm_rerank(query: str, candidates: List[str]) -> List[float]:
    if not candidates:
        return []
    cli = _client()
    chunk = candidates[:50]
    sys = (
        "You are a ranking function. Score how relevant a candidate entity name is to the query. "
        "Return a JSON object with key 'scores' as an array of numbers in [0,1] aligned with the input order."
    )
    user = {"query": query, "candidates": chunk}
    resp = cli.chat.completions.create(
        model=RERANK_MODEL,
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":json.dumps(user, ensure_ascii=False)}],
        temperature=0,
        response_format={"type":"json_object"}
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        scores = data.get("scores", [])
        if len(scores) != len(chunk):
            scores = [0.5]*len(chunk)
        return [float(x) for x in scores]
    except Exception:
        return [0.5]*len(chunk)

# ============== ANN 索引（HNSW 或 IVF-PQ） ==============
class ANNIndex:
    def __init__(self, index_dir: str, prefer_ivfpq: bool = True, nprobe: int = 16):
        self.index_dir = index_dir
        self.E = np.load(os.path.join(index_dir, "entity_embs.npy"))
        if self.E.dtype != np.float32:
            self.E = self.E.astype("float32")
        self.use_faiss = False
        self.faiss_index = None
        self.is_ivfpq = False
        self.nprobe = nprobe
        if faiss is not None:
            ivf = os.path.join(index_dir, "entity_ivfpq.index")
            hnsw = os.path.join(index_dir, "entity_hnsw.index")
            path = ivf if (prefer_ivfpq and os.path.exists(ivf)) else (hnsw if os.path.exists(hnsw) else None)
            if path:
                self.faiss_index = faiss.read_index(path)
                self.use_faiss = True
                self.is_ivfpq = "IVF" in str(type(self.faiss_index))
                if self.is_ivfpq:
                    try:
                        self.faiss_index.nprobe = nprobe
                    except Exception:
                        pass

    def search(self, q: np.ndarray, topk: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        if q.ndim == 1:
            q = q[None, :]
        if self.use_faiss:
            D, I = self.faiss_index.search(q, topk)
            return I, D
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        En = self.E / (np.linalg.norm(self.E, axis=1, keepdims=True) + 1e-12)
        S = qn @ En.T
        I = np.argsort(-S, axis=1)[:, :topk]
        D = np.take_along_axis(S, I, axis=1)
        return I, D

# ============== build entity index (subcommand) ==============
def cmd_build_index(db_path: str, out_dir: str, fp16: bool = False,
                    index_type: str = "ivfpq", nlist: int = 4096, m: int = 64, nbits: int = 8):
    os.makedirs(out_dir, exist_ok=True)
    ents = fetch_entities(db_path)
    ids = [i for i, _ in ents]
    names = [n for _, n in ents]
    E = openai_embed(names)  # (N, 3072)
    if fp16:
        E = E.astype("float16")
    np.save(os.path.join(out_dir, "entity_embs.npy"), E)
    with open(os.path.join(out_dir, "entity_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"num_entities": len(ents), "ids": ids}, f)

    if faiss is None:
        print("faiss not installed, only save vectors; will use numpy retrieval.")
        return

    Ef32 = E.astype("float32")
    d = Ef32.shape[1]

    if index_type.lower() == "ivfpq":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, int(nlist), int(m), int(nbits))
        index.metric_type = faiss.METRIC_INNER_PRODUCT
        train_samples = Ef32 if Ef32.shape[0] <= 100000 else Ef32[np.random.choice(Ef32.shape[0], 100000, replace=False)]
        index.train(train_samples)
        index.add(Ef32)
        index.nprobe = 16
        faiss.write_index(index, os.path.join(out_dir, "entity_ivfpq.index"))
        print(f"Built IVF-PQ index: N={Ef32.shape[0]}, dim={d}, nlist={nlist}, m={m}, nbits={nbits}")
    else:
        index = faiss.IndexHNSWFlat(d, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64
        index.add(Ef32)
        faiss.write_index(index, os.path.join(out_dir, "entity_hnsw.index"))
        print(f"Built HNSW index: N={Ef32.shape[0]}, dim={d}")
def _load_meta(index_dir: str):
    meta_path = os.path.join(index_dir, "entity_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # pos2dbid: vector position index -> database id
    pos2dbid = meta["ids"]
    return pos2dbid
# ============== candidate generation (LLM+heuristic keywords → ANN → literal hit priority → optional LLM reranking) ==============
# def keyword_candidates_openai_with_keywords(
#     db_path: str, index_dir: str, question: str, keywords: list[str],
#     ann_topk_per_kw: int = 50, no_llm_rerank: bool = False, rerank_k: int = 20,
#     literal_sim_th: float = 0.85
# ):
#     ents = fetch_entities(db_path)
#     names = [n for _, n in ents]
#     ann = ANNIndex(index_dir)

#     # A) literal hit (treat problem原文当 span)
#     literal_hits = literal_hit_entities(db_path, question, sim_th=literal_sim_th, span_only=True)

#     # B) keywords list
#     kws = keywords if keywords else [question]

#     # C) per keyword ANN
#     q_embs = openai_embed(kws)
#     I, _ = ann.search(q_embs, topk=ann_topk_per_kw)

#     cand: Dict[int, Dict[str, Any]] = {}  # eid_pos -> info
#     for eid, name in literal_hits:
#         cand[eid] = {"name": name, "kws": set(["__literal__"]), "seed": True}

#     for ki, row in enumerate(I):
#         kw = kws[ki]
#         for eid_pos in row.tolist():
#             if eid_pos not in cand:
#                 cand[eid_pos] = {"name": names[eid_pos], "kws": set(), "seed": False}
#             cand[eid_pos]["kws"].add(kw)

#     ranked_by_freq = sorted(
#         cand.items(),
#         key=lambda x: (0 if x[1]["seed"] else 1, -len(x[1]["kws"]), x[0])
#     )

#     head = ranked_by_freq[:rerank_k]
#     tail = ranked_by_freq[rerank_k:]

#     if not no_llm_rerank and head:
#         head_ids = [eid for eid,_ in head]
#         head_names = [cand[eid]["name"] for eid in head_ids]
#         scores = llm_rerank(question, head_names)
#         head_ranked = sorted(zip(head_ids, head_names, scores), key=lambda x: -x[2])
#         ranked = head_ranked + [(eid, cand[eid]["name"], 0.0) for eid,_ in tail]
#     else:
#         ranked = [(eid, cand[eid]["name"],
#                    float(len(cand[eid]["kws"]) + (10 if cand[eid]["seed"] else 0)))
#                   for eid,_ in ranked_by_freq]

#     out = []
#     for eid, name, sc in ranked:
#         kws_clean = sorted([kw for kw in cand[eid]["kws"] if kw != "__literal__"])
#         out.append({
#             "id": ents[eid][0],
#             "name": name,
#             "score": float(sc),
#             "kws": kws_clean,
#             "seed": cand[eid]["seed"]
#         })
#     return {"keywords": kws, "candidates": out, "literal_hits": [n for _, n in literal_hits]}
def keyword_candidates_openai_with_keywords(
    db_path: str, index_dir: str, question: str, keywords: List[str],
    ann_topk_per_kw: int = 50, no_llm_rerank: bool = False, rerank_k: int = 20,
    literal_sim_th: float = 0.85
):
    ents = fetch_entities(db_path)                 # [(db_id, name), ...]
    id2name = {dbid: name for dbid, name in ents}  # convenient to get name
    pos2dbid = _load_meta(index_dir)               # key: position->db_id
    ann = ANNIndex(index_dir)

    # A) literal hit (return database id)
    literal_hits = literal_hit_entities(db_path, question, sim_th=literal_sim_th, span_only=True)

    # B) keywords list
    kws = keywords if keywords else [question]

    # C) per keyword ANN
    q_embs = openai_embed(kws)
    I, _ = ann.search(q_embs, topk=ann_topk_per_kw)

    # use "database id" as cand key, avoid confusion
    cand: Dict[int, Dict[str, Any]] = {}  # db_id -> info

    # merge function: if exists, merge keywords set and seed mark
    def _merge(db_id: int, name: str, kw: Optional[str], is_seed: bool):
        if db_id not in cand:
            cand[db_id] = {"name": name, "kws": set(), "seed": False}
        if kw is not None:
            cand[db_id]["kws"].add(kw)
        if is_seed:
            cand[db_id]["seed"] = True

    # merge literal hit (database id)
    for db_id, name in literal_hits:
        _merge(db_id, name, None, True)

    # merge ANN result (first map position index to database id)
    for ki, row in enumerate(I):
        kw = kws[ki]
        for pos in row.tolist():
            if pos < 0 or pos >= len(pos2dbid):
                continue
            db_id = pos2dbid[pos]
            name = id2name.get(db_id, f"<id:{db_id}>")
            _merge(db_id, name, kw, False)

    # sort by rules (seed priority, more keywords priority)
    ranked_by_freq = sorted(
        cand.items(),
        key=lambda x: (0 if x[1]["seed"] else 1, -len(x[1]["kws"]), x[0])
    )

    head = ranked_by_freq[:rerank_k]
    tail = ranked_by_freq[rerank_k:]

    if not no_llm_rerank and head:
        head_ids = [db_id for db_id, _ in head]
        head_names = [cand[db_id]["name"] for db_id in head_ids]
        scores = llm_rerank(question, head_names)
        head_ranked = sorted(zip(head_ids, head_names, scores), key=lambda x: -x[2])
        ranked = head_ranked + [(db_id, cand[db_id]["name"], 0.0) for db_id, _ in tail]
    else:
        ranked = [
            (db_id, info["name"], float(len(info["kws"]) + (10 if info["seed"] else 0)))
            for db_id, info in ranked_by_freq
        ]

    out = []
    for db_id, name, sc in ranked:
        kws_clean = sorted([kw for kw in cand[db_id]["kws"]])
        out.append({
            "id": db_id,         # directly output database id
            "name": name,
            "score": float(sc),
            "kws": kws_clean,
            "seed": cand[db_id]["seed"]
        })
    return {"keywords": kws, "candidates": out, "literal_hits": [n for _, n in literal_hits]}
# ============== dynamic topic selection (including seed priority and 3-hop尽量连通) ==============
def link_topics_via_combined_keywords_dynamic(
    db_path: str, index_dir: str, question: str,
    llm_max: int = 8, heur_max_len: int = 8,
    ann_topk_per_kw: int = 50, no_llm_rerank: bool = False, rerank_k: int = 20,
    max_topics: int = 6, min_topics: int = 1, try_limit: int = 200
):
    kws = combined_keywords(question, llm_max=llm_max, heur_max_len=heur_max_len)
    
    
    pack = keyword_candidates_openai_with_keywords(
        db_path, index_dir, question, kws,
        ann_topk_per_kw=ann_topk_per_kw,
        no_llm_rerank=no_llm_rerank, rerank_k=rerank_k
    )
    cands = pack["candidates"]

    seeds = [c for c in cands if c.get("seed")]
    covered_any = set()
    for c in cands:
        covered_any.update(c["kws"])
    base_target = max(min_topics, min(len(covered_any), max_topics))
    target = max(base_target, len(seeds))

    chosen = []
    covered = set()

    def _can_add(eid):
        ids = [x[0] for x in chosen] + [eid]
        return within_k(db_path, ids, k=3)

    # first put all seeds (like person name directly appearing in the question)
    for s in seeds:
        chosen.append((s["id"], s["name"], 1, set(s["kws"])))
        covered.update(s["kws"])

    # then fill other candidates, prioritize keyword coverage,尽量保持三跳连通
    for c in cands[:try_limit]:
        if c.get("seed"):
            continue
        if len(chosen) >= target:
            break
        eid, name = c["id"], c["name"]
        new_kws = set(c["kws"]) - covered
        if new_kws and _can_add(eid):
            chosen.append((eid, name, c["score"], set(new_kws)))
            covered.update(new_kws)
        elif not new_kws and _can_add(eid):
            chosen.append((eid, name, c["score"], set()))
        elif len(chosen) < target:
            chosen.append((eid, name, c["score"], set(new_kws)))
            covered.update(new_kws)

    return [(eid, name, float(score)) for eid, name, score, _ in chosen[:max_topics]]

# ============== name/time loose matching and quick screening ==============
_MONTHS = {
    "january":1,"jan":1,"february":2,"feb":2,"march":3,"mar":3,"april":4,"apr":4,"may":5,
    "june":6,"jun":6,"july":7,"jul":7,"august":8,"aug":8,"september":9,"sep":9,"sept":9,
    "october":10,"oct":10,"november":11,"nov":11,"december":12,"dec":12,
}

def _parse_time_loose(text: str):
    if not text:
        return (None, None, None)
    s = text.strip().lower()
    s = re.sub(r"[,]", " ", s)
    s = re.sub(r"\s+", " ", s)

    m = re.fullmatch(r"(\d{4})", s)
    if m:
        return (int(m.group(1)), None, None)
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})", s)
    if m:
        y, mth = int(m.group(1)), int(m.group(2))
        return (y, mth if 1 <= mth <= 12 else None, None)
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mth <= 12 and 1 <= d <= 31:
            return (y, mth, d)
        return (y, mth if 1 <= mth <= 12 else None, None)

    toks = s.split()
    if len(toks) == 2 and toks[0] in _MONTHS and re.fullmatch(r"\d{4}", toks[1] or ""):
        return (int(toks[1]), _MONTHS[toks[0]], None)
    if len(toks) == 3:
        if toks[0] in _MONTHS and toks[1].isdigit() and re.fullmatch(r"\d{4}", toks[2] or ""):
            return (int(toks[2]), _MONTHS[toks[0]], int(toks[1]))
        if toks[1] in _MONTHS and toks[0].isdigit() and re.fullmatch(r"\d{4}", toks[2] or ""):
            return (int(toks[2]), _MONTHS[toks[1]], int(toks[0]))

    y = re.search(r"\b(19\d{2}|20\d{2})\b", s)
    if y:
        return (int(y.group(1)), None, None)
    return (None, None, None)

def _time_match_at_level(answer_time: str, edge_time: str, level: str) -> bool:
    ay, am, ad = _parse_time_loose(answer_time)
    ey, em, ed = _parse_time_loose(edge_time)
    if not ey:
        return False
    level = (level or "day").lower()
    if level == "year":
        return (ay is not None) and (ey == ay)
    if level == "month":
        return (ay is not None and am is not None) and (ey == ay and em == am)
    return (ay is not None and am is not None and ad is not None) and (ey == ay and em == am and ed == ad)

def _subgraph_nodes_from_edges(sub_edges):
    nodes = set()
    for h, _, t, _ in sub_edges:
        nodes.add(h); nodes.add(t)
    return nodes

def _resolve_answer_entities_by_name(db_path: str, answers: List[str], sim_th: float = 0.60):
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM entities ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()

    out_ids = []
    for ans in answers:
        best = (-1.0, None)
        for eid, name in rows:
            s = _name_sim(ans, name)
            if s > best[0]:
                best = (s, eid)
        out_ids.append(best[1] if best[0] >= sim_th else None)
    return out_ids

def quick_answer_screen(db_path: str, sub_edges, answer_type: str, answers: List[str], time_level: str):
    answer_type = (answer_type or "").strip().lower()
    time_level = (time_level or "").strip().lower()
    if not answers:
        return {"reachable": False, "reason": "no answers provided"}

    if answer_type == "entity":
        nodes = _subgraph_nodes_from_edges(sub_edges)
        ans_ids = _resolve_answer_entities_by_name(db_path, answers)
        for aid in ans_ids:
            if aid is not None and aid in nodes:
                return {"reachable": True, "reason": "entity present in subgraph"}
        return {"reachable": False, "reason": "entity not in 3-hop subgraph"}

    if answer_type == "time":
        for _, _, _, ts in sub_edges:
            for ans in answers:
                if _time_match_at_level(ans, ts, time_level or "day"):
                    return {"reachable": True, "reason": f"time matches at {time_level or 'day'} level"}
        return {"reachable": False, "reason": "time not matched in 3-hop subgraph"}

    return {"reachable": True, "reason": "skip check for non-entity/time"}

# ============== adaptive topic selection + quick screening ==============
def choose_topics_with_screen_dynamic(
    db_path: str, index_dir: str, question_item: dict,
    llm_max: int = 8, heur_max_len: int = 8,
    ann_topk_per_kw: int = 50, no_llm_rerank: bool = False, rerank_k: int = 20,
    max_topics: int = 6, min_topics: int = 1, try_limit: int = 200
):
    topics = link_topics_via_combined_keywords_dynamic(
        db_path, index_dir, question_item["question"],
        llm_max=llm_max, heur_max_len=heur_max_len,
        ann_topk_per_kw=ann_topk_per_kw, no_llm_rerank=no_llm_rerank,
        rerank_k=rerank_k, max_topics=max_topics, min_topics=min_topics,
        try_limit=try_limit
    )
    sub_edges = extract_3hop_subgraph(db_path, [x[0] for x in topics], k=3)
    screen = quick_answer_screen(
        db_path, sub_edges,
        answer_type=question_item.get("answer_type"),
        answers=question_item.get("answers") or [],
        time_level=question_item.get("time_level") or "day",
    )
    if screen["reachable"]:
        return topics, sub_edges, screen

    # if not passed, try to add more candidates (keep 3-hop connected)
    pack = keyword_candidates_openai_with_keywords(
        db_path, index_dir, question_item["question"],
        combined_keywords(question_item["question"], llm_max=llm_max, heur_max_len=heur_max_len),
        ann_topk_per_kw=ann_topk_per_kw, no_llm_rerank=no_llm_rerank, rerank_k=rerank_k
    )
    cands = pack["candidates"]
    chosen_ids = {x[0] for x in topics}
    def _can_add(eid):
        ids = [x[0] for x in topics] + [eid]
        return within_k(db_path, ids, k=3)

    for c in cands:
        if len(topics) >= max_topics:
            break
        if c["id"] in chosen_ids:
            continue
        if not _can_add(c["id"]):
            continue
        topics.append((c["id"], c["name"], c["score"]))

        sub_edges = extract_3hop_subgraph(db_path, [x[0] for x in topics], k=3)
        screen = quick_answer_screen(
            db_path, sub_edges,
            answer_type=question_item.get("answer_type"),
            answers=question_item.get("answers") or [],
            time_level=question_item.get("time_level") or "day",
        )
        if screen["reachable"]:
            return topics, sub_edges, screen

    return topics, sub_edges, screen

# ============== CLI ==============
def cmd_run(db_path: str, index_dir: str, questions_path: str, out_path: str,
            llm_max: int, heur_max_len: int, ann_topk_per_kw: int, rerank_k: int,
            no_llm_rerank: bool, max_topics: int, min_topics: int, try_limit: int):
    data = json.load(open(questions_path, "r", encoding="utf-8"))
    out = []
    for q in data:
        topics, sub_edges, screen = choose_topics_with_screen_dynamic(
            db_path, index_dir, q,
            llm_max=llm_max, heur_max_len=heur_max_len,
            ann_topk_per_kw=ann_topk_per_kw, rerank_k=rerank_k,
            no_llm_rerank=no_llm_rerank,
            max_topics=max_topics, min_topics=min_topics, try_limit=try_limit
        )
        out.append({
            "quid": q["quid"],
            "question": q["question"],
            "answer_type": q.get("answer_type"),
            "time_level": q.get("time_level"),
            "qtype": q.get("qtype"),
            "qlabel": q.get("qlabel"),
            "topics": [{"id": eid, "name": name, "score": score} for eid, name, score in topics if score > 0],
            "subgraph_edges": sub_edges[:3],
            "quick_screen": screen
        })
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")

def main():
    ap = argparse.ArgumentParser(description="KG bootstrap with LLM+heuristic keywords, ANN, rerank, 3-hop and quick screening")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("build-index", help="Build entity embeddings and FAISS index")
    p1.add_argument("--db", required=True)
    p1.add_argument("--out_dir", required=True)
    p1.add_argument("--fp16", action="store_true")
    p1.add_argument("--index_type", default="ivfpq", choices=["ivfpq","hnsw"])
    p1.add_argument("--nlist", type=int, default=4096)
    p1.add_argument("--m", type=int, default=64)
    p1.add_argument("--nbits", type=int, default=8)

    p2 = sub.add_parser("run", help="Run topic selection and subgraph extraction")
    p2.add_argument("--db", required=True)
    p2.add_argument("--index_dir", required=True)
    p2.add_argument("--questions", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--llm_max", type=int, default=8)
    p2.add_argument("--heur_max_len", type=int, default=8)
    p2.add_argument("--ann_topk_per_kw", type=int, default=50)
    p2.add_argument("--rerank_k", type=int, default=20)
    p2.add_argument("--no_llm_rerank", action="store_true")
    p2.add_argument("--max_topics", type=int, default=6)
    p2.add_argument("--min_topics", type=int, default=1)
    p2.add_argument("--try_limit", type=int, default=200)

    args = ap.parse_args()
    if args.cmd == "build-index":
        cmd_build_index(args.db, args.out_dir, fp16=args.fp16,
                        index_type=args.index_type, nlist=args.nlist, m=args.m, nbits=args.nbits)
    elif args.cmd == "run":
        cmd_run(args.db, args.index_dir, args.questions, args.out,
                llm_max=args.llm_max, heur_max_len=args.heur_max_len,
                ann_topk_per_kw=args.ann_topk_per_kw, rerank_k=args.rerank_k,
                no_llm_rerank=args.no_llm_rerank, max_topics=args.max_topics,
                min_topics=args.min_topics, try_limit=args.try_limit)
    else:
        raise ValueError("unknown subcommand")

if __name__ == "__main__":
    # temporary test
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "test":
        print("Testing llm_keywords function...")
        test_questions = [
            "After the Danish Ministry of Defence and Security, who was the first to visit Iraq?",
            "What is the capital of France?",
            "Who won the Nobel Prize in Physics in 2023?"
        ]
        for q in test_questions:
            print(f"\nQuestion: {q}")
            keywords = llm_keywords(q)
            print(f"Keywords: {keywords}")
    else:
        main()