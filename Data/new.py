# file: kg_semantic_candidates.py
"""
- build-index:
    1) read entity table entities(id INTEGER PRIMARY KEY, name TEXT) from SQLite.
    2) embed entity names with OpenAI text-embedding-3-large, save to out_dir/entity_embs.npy.
    3) write out_dir/entity_meta.json (contains entity id order, for pos<->id mapping).
    4) optional: build FAISS HNSW index and write to out_dir/entity_hnsw.index (recommended).
- run:
    1) read questions from questions.json; each item expects to contain field "keywords": ["...","..."] (you have manually selected keywords).
       if "keywords" is missing, it will fall back to using question text itself as the only keyword.
    2) do "pure semantic matching" for each question's keywords: keyword embedding ↔ all entity embeddings.
       first use HNSW to get per_kw_topk nearest neighbors for each keyword, then do exact dot product on the union,
       get the final score for each entity based on aggregation strategy (max/mean/sum/softmax), and return the topk.
       get the final score for each entity based on aggregation strategy (max/mean/sum/softmax), and return the topk.
    3) output a JSON file, containing candidates list (id, name, score) for each question.

dependencies:
  pip install openai faiss-cpu

environment variables:
  OPENAI_API_KEY=...
"""

import os, re, json, argparse, sqlite3
from typing import List, Tuple, Dict, Any, Optional
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

def canon_text(s: str) -> str:
    """
    for input to embedding: replace underscore/hyphen with space, compress multiple spaces, remove leading and trailing whitespace.
    note: only affect embedding input, not change the original name in the database.
    """
    s = str(s)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

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

# ============== path and mapping ==============
def _meta_path(index_dir: str) -> str:
    return os.path.join(index_dir, "entity_meta.json")

def load_entity_ids(index_dir: str):
    with open(_meta_path(index_dir), "r", encoding="utf-8") as f:
        meta = json.load(f)
    ids = [int(x) for x in meta["ids"]]
    pos2id = np.array(ids, dtype=np.int64)
    id2pos = {int(eid): i for i, eid in enumerate(ids)}
    return pos2id, id2pos

# ============== OpenAI embedding ==============
def openai_embed(texts: List[str]) -> np.ndarray:
    """
    return L2 normalized embedding matrix (N, D); will call API in batches of 256.
    """
    cli = _client()
    out = []
    for i in range(0, len(texts), 256):
        batch = texts[i:i+256]
        resp = cli.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([v.embedding for v in resp.data])
    arr = np.array(out, dtype=np.float32)
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
    return arr

# ============== build index (HNSW optional) ==============
def cmd_build_index(db_path: str, out_dir: str, fp16: bool = False,
                    build_hnsw: bool = True, M: int = 64, ef_construction: int = 400, ef_search: int = 128):
    os.makedirs(out_dir, exist_ok=True)
    ents = fetch_entities(db_path)
    ids = [i for i, _ in ents]
    names = [n for _, n in ents]
    names_for_embed = [canon_text(n) for n in names]
    # 1) entity name embedding
    E = openai_embed(names_for_embed)  # (N, 3072)
    if fp16:
        E = E.astype("float16")
    np.save(os.path.join(out_dir, "entity_embs.npy"), E)

    # 2) write meta (pos <-> id)
    with open(_meta_path(out_dir), "w", encoding="utf-8") as f:
        json.dump({"num_entities": len(ents), "ids": ids}, f)

    # 3) (optional) build HNSW
    if build_hnsw and faiss is not None:
        Ef32 = E.astype("float32")
        d = Ef32.shape[1]
        index = faiss.IndexHNSWFlat(d, M)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(Ef32)
        faiss.write_index(index, os.path.join(out_dir, "entity_hnsw.index"))
        print(f"[HNSW] Built: N={Ef32.shape[0]}, dim={d}, M={M}, efC={ef_construction}, efS={ef_search}")
    elif build_hnsw and faiss is None:
        print("[HNSW] faiss not installed, skip; will use exact matrix multiplication at runtime.")

    print(f"[OK] Embeddings saved to {out_dir}")

# ============== pure semantic candidates (only based on your selected keywords) ==============
def semantic_candidates_by_keywords(
    db_path: str,
    index_dir: str,
    keywords: List[str],
    topk: int = 200,
    per_kw_topk: int = 500,     # how many nearest neighbors to get for each keyword
    agg: str = "max",           # max / mean / sum / softmax
    exclude_ids: Optional[List[int]] = None,
    use_faiss: bool = True
) -> List[Tuple[int, str, float]]:
    """
    return [(entity_id, entity_name, score)], only based on semantic similarity:
      1) do vector retrieval for each keyword (HNSW)
      2) merge candidates, then do exact scoring on the candidate subset and get the final score based on the aggregation strategy
      3) sort and get the topk
    """
    # read entities
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM entities ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()
    id2name = {int(e): n for e, n in rows}

    # load entity embeddings
    E = np.load(os.path.join(index_dir, "entity_embs.npy"))
    if E.dtype != np.float32:
        E = E.astype("float32")
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    # pos<->id
    pos2id, id2pos = load_entity_ids(index_dir)

    # keywords
    raw_kw = [k for k in (keywords or []) if isinstance(k, str) and k.strip()]
    if not raw_kw:
        return []

    # keyword variants: original + underscore + normalized (space/hyphen→space)
    aug = []
    seen = set()
    for k in raw_kw:
        variants = [
            str(k),
            str(k).replace(" ", "_"),
            canon_text(k),
        ]
        for v in variants:
            v2 = re.sub(r"\s+", " ", v).strip()
            if v2 and v2 not in seen:
                seen.add(v2); aug.append(v2)

    kw = aug
    K = openai_embed(kw)
 # (K, D), already normalized
    K = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)

    # candidate union
    cand_pos = set()
    hnsw_path = os.path.join(index_dir, "entity_hnsw.index")
    if use_faiss and faiss is not None and os.path.exists(hnsw_path):
        index = faiss.read_index(hnsw_path)
        try:
            index.hnsw.efSearch = max(128, per_kw_topk)
        except Exception:
            pass
        _, I = index.search(K, per_kw_topk)  # I: (K, per_kw_topk) of positions
        for row in I:
            for p in row.tolist():
                if p >= 0:
                    cand_pos.add(int(p))
    else:
        # exact: K @ En^T, get the top per_kw_topk for each keyword
        sims_full = K @ En.T  # (K, N)
        for i in range(sims_full.shape[0]):
            idx = np.argpartition(-sims_full[i], per_kw_topk)[:per_kw_topk]
            cand_pos.update(int(x) for x in idx.tolist())

    if not cand_pos:
        return []

    cand_pos = sorted(cand_pos)
    # do exact scoring and aggregation on the candidate subset
    sims = K @ En[cand_pos].T  # (K, C)
    if agg == "mean":
        s = sims.mean(axis=0)
    elif agg == "sum":
        s = sims.sum(axis=0)
    elif agg == "softmax":
        w = np.exp(sims - sims.max(axis=0, keepdims=True))
        s = (w * sims).sum(axis=0) / (w.sum(axis=0) + 1e-12)
    else:  # "max"
        s = sims.max(axis=0)

    # exclude existing ids (optional)
    excl = set(int(x) for x in (exclude_ids or []))
    kw_norm_set = {canon_text(x).lower() for x in kw}

    items = []
    for j, pos in enumerate(cand_pos):
        eid = int(pos2id[pos])
        if eid in excl:
            continue
        name_norm = canon_text(id2name.get(eid, ""))
        bonus = 0.02 if name_norm.lower() in kw_norm_set else 0.0
        
        items.append((eid, id2name.get(eid, ""), float(s[j]) + bonus))

    items.sort(key=lambda x: -x[2])
    return items[:topk]
from entity_try import combined_keywords
# ============== run: read keywords from question file, output candidates ==============
def _keywords_from_item(item: Dict[str, Any]) -> List[str]:
    # prioritize using the "keywords" you provided
    kws = item.get("keywords")
    kws = combined_keywords(item.get("question"))
    if isinstance(kws, list) and kws:
        return [str(x) for x in kws if isinstance(x, (str, int, float))]
    # if no keywords, fall back to using question text itself as the only keyword
    q = item.get("question") or ""
    q = re.sub(r"\s+", " ", str(q)).strip()
    return [q] if q else []

# def cmd_run(db_path: str, index_dir: str, questions_path: str, out_path: str,
#             topk: int, per_kw_topk: int, agg: str, no_faiss: bool):
#     data = json.load(open(questions_path, "r", encoding="utf-8"))
#     # read once entity name mapping, for backfilling
#     conn = _connect(db_path)
#     cur = conn.cursor()
#     cur.execute("SELECT id, name FROM entities")
#     id_name = {int(eid): nm for eid, nm in cur.fetchall()}
#     conn.close()

#     out = []
#     for q in data[:500]:
#         kws = _keywords_from_item(q)
#         cands = semantic_candidates_by_keywords(
#             db_path, index_dir, kws,
#             topk=topk, per_kw_topk=per_kw_topk, agg=agg,
#             exclude_ids=[], use_faiss=(not no_faiss)
#         )
#         out.append({
#             "quid": q.get("quid"),
#             "question": q.get("question"),
#             "keywords_used": kws,
#             "candidates": [
#                 {"id": int(eid), "name": id_name.get(int(eid), ""), "score": float(score)}
#                 for eid, _nm, score in cands if score > 0.7
#             ]
#         })
#     json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
#     print(f"[OK] Wrote {out_path}")


def cmd_run(db_path: str, index_dir: str, questions_path: str, out_path: str,
            topk: int, per_kw_topk: int, agg: str, no_faiss: bool):
    data = json.load(open(questions_path, "r", encoding="utf-8"))
    # read once entity name mapping, for backfilling
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM entities")
    id_name = {int(eid): nm for eid, nm in cur.fetchall()}
    conn.close()

    for q in data[5:10]:
        print(f"[INFO] Processing question {q.get('quid')} ...")
        kws = _keywords_from_item(q)
        cands = semantic_candidates_by_keywords(
            db_path, index_dir, kws,
            topk=topk, per_kw_topk=per_kw_topk, agg=agg,
            exclude_ids=[], use_faiss=(not no_faiss)
        )
        result = {
            "quid": q.get("quid"),
            "question": q.get("question"),
            "keywords_used": kws,
            "candidates": [
                {"id": int(eid), "name": id_name.get(int(eid), ""), "score": float(score)}
                for eid, _nm, score in cands if score > 0.7
            ],
            "answer_type": q.get("answer_type"),
            "time_level": q.get("time_level"),
            "qtype": q.get("qtype"),
            "qlabel": q.get("qlabel"),
            "answers": q.get("answers")
               
        }

        # --- key part: read + append + write ---
        try:
            # if file exists, read
            with open(out_path, "r", encoding="utf-8") as f:
                out = json.load(f)
        except FileNotFoundError:
            # if file does not exist, initialize
            out = []

        out.append(result)

        # write back
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"[OK] wrote question {q.get('quid')} to {out_path}")

# ============== CLI ==============
def main():
    ap = argparse.ArgumentParser(description="Semantic candidates from selected keywords (OpenAI embeddings, optional HNSW)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("build-index", help="Build entity embeddings and optional HNSW index")
    p1.add_argument("--db", required=True)
    p1.add_argument("--out_dir", required=True)
    p1.add_argument("--fp16", action="store_true")
    p1.add_argument("--no_hnsw", action="store_true")
    p1.add_argument("--M", type=int, default=64)
    p1.add_argument("--ef_construction", type=int, default=400)
    p1.add_argument("--ef_search", type=int, default=128)

    p2 = sub.add_parser("run", help="Score entities against your selected keywords (pure semantic)")
    p2.add_argument("--db", required=True)
    p2.add_argument("--index_dir", required=True)
    p2.add_argument("--questions", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--topk", type=int, default=200)
    p2.add_argument("--per_kw_topk", type=int, default=500)
    p2.add_argument("--agg", choices=["max","mean","sum","softmax"], default="max")
    p2.add_argument("--no_faiss", action="store_true")

    args = ap.parse_args()
    if args.cmd == "build-index":
        cmd_build_index(
            args.db, args.out_dir, fp16=args.fp16,
            build_hnsw=(not args.no_hnsw),
            M=args.M, ef_construction=args.ef_construction, ef_search=args.ef_search
        )
    elif args.cmd == "run":
        cmd_run(
            args.db, args.index_dir, args.questions, args.out,
            topk=args.topk, per_kw_topk=args.per_kw_topk,
            agg=args.agg, no_faiss=args.no_faiss
        )
    else:
        raise ValueError("unknown subcommand")

if __name__ == "__main__":
    main()
