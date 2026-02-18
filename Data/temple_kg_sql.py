
# file: tempkg_sqlite.py
import sqlite3, csv
from typing import Optional, Iterable, Tuple, Union, List

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    # Pragmas for better write/read performance with durability tradeoffs acceptable for single-writer
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db(db_path: str, drop: bool = False):
    conn = _connect(db_path)
    cur = conn.cursor()
    if drop:
        cur.executescript("""
        DROP TABLE IF EXISTS edges;
        DROP TABLE IF EXISTS entities;
        """)
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS entities(
        id   INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE
    );
    CREATE TABLE IF NOT EXISTS edges(
        id      INTEGER PRIMARY KEY,
        head_id INTEGER NOT NULL,
        relation TEXT NOT NULL,
        tail_id INTEGER NOT NULL,
        time    TEXT NOT NULL,
        FOREIGN KEY(head_id) REFERENCES entities(id),
        FOREIGN KEY(tail_id) REFERENCES entities(id),
        UNIQUE(head_id, relation, tail_id, time)
    );                   

    CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
    CREATE INDEX IF NOT EXISTS idx_edges_head   ON edges(head_id);
    CREATE INDEX IF NOT EXISTS idx_edges_tail   ON edges(tail_id);
    CREATE INDEX IF NOT EXISTS idx_edges_time   ON edges(time);
    """)
    conn.commit()
    conn.close()

def _get_or_create_entity_id(cur: sqlite3.Cursor, name: str) -> int:
    name = name.strip()
    cur.execute("SELECT id FROM entities WHERE name=?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO entities(name) VALUES (?)", (name,))
    return cur.lastrowid

def bulk_load_tsv(db_path: str, tsv_path: str, delimiter: str = "\t", batch: int = 50_000):
    conn = _connect(db_path); cur = conn.cursor()
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        buf: List[Tuple[int,str,int,str]] = []
        for i, row in enumerate(reader, 1):
            if not row or len(row) < 4: 
                continue
            h_name, rel, t_name, time_str = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            h_id = _get_or_create_entity_id(cur, h_name)
            t_id = _get_or_create_entity_id(cur, t_name)
            buf.append((h_id, rel, t_id, time_str))
            if len(buf) >= batch:
                cur.executemany("INSERT OR IGNORE INTO edges(head_id, relation, tail_id, time) VALUES (?,?,?,?)", buf)
                conn.commit()
                buf.clear()
        if buf:
            cur.executemany("INSERT OR IGNORE INTO edges(head_id, relation, tail_id, time) VALUES (?,?,?,?)", buf)
            conn.commit()
    conn.close()

def entity_id(db_path: str, query: Union[str,int]) -> int:
    conn = _connect(db_path); cur = conn.cursor()
    if isinstance(query, int):
        cur.execute("SELECT 1 FROM entities WHERE id=?", (query,))
        if cur.fetchone(): 
            conn.close(); return query
        conn.close(); raise KeyError(f"entity id {query} not found")
    q = str(query).strip()
    cur.execute("SELECT id FROM entities WHERE name=?", (q,))
    row = cur.fetchone(); conn.close()
    if not row: raise KeyError(f"entity name '{q}' not found")
    return row[0]

def entity_name(db_path: str, eid: int) -> str:
    conn = _connect(db_path); cur = conn.cursor()
    cur.execute("SELECT name FROM entities WHERE id=?", (eid,))
    row = cur.fetchone(); conn.close()
    if not row: raise KeyError(f"entity id {eid} not found")
    return row[0]
def export_entities(db_path: str) -> List[Tuple[int, str]]:
    """
    导出所有实体的 (id, name) 列表，按 id 升序。
    """
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM entities ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()
    return rows
def retrieve_one_hop(
    db_path: str,
    query: Union[str,int],
    direction: str = "both",     # "out" | "in" | "both"
    return_as: str = "names",    # "names" | "ids"
    limit: Optional[int] = None,
    sort_by_time: bool = False
) -> List[Tuple[Union[str,int], str, Union[str,int], str]]:
    eid = entity_id(db_path, query)
    conn = _connect(db_path); cur = conn.cursor()

    rows = []
    if direction in ("out", "both"):
        sql = "SELECT head_id, relation, tail_id, time FROM edges WHERE head_id=?"
        if sort_by_time: sql += " ORDER BY time ASC"
        if limit: sql += f" LIMIT {int(limit)}"
        cur.execute(sql, (eid,))
        rows.extend(cur.fetchall())
    if direction in ("in", "both"):
        sql = "SELECT head_id, relation, tail_id, time FROM edges WHERE tail_id=?"
        if sort_by_time: sql += " ORDER BY time ASC"
        if limit: sql += f" LIMIT {int(limit)}"
        cur.execute(sql, (eid,))
        rows.extend(cur.fetchall())

    if return_as == "ids":
        conn.close()
        return rows

    # map ids to names in one query
    ids = set()
    for h, _, t, _ in rows:
        ids.add(h); ids.add(t)
    id_list = list(ids)
    qmarks = ",".join("?" for _ in id_list) or "NULL"
    id2name = {}
    if id_list:
        cur.execute(f"SELECT id, name FROM entities WHERE id IN ({qmarks})", id_list)
        id2name = {i:n for i,n in cur.fetchall()}
    conn.close()

    out = []
    for h, r, t, ts in rows:
        out.append((id2name.get(h, h), r, id2name.get(t, t), ts))
    return out

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build a SQLite-backed TemporalKG from TSV.")
    p.add_argument("--db", required=True, help="SQLite db path, e.g., tempkg.db")
    p.add_argument("--tsv", required=True, help="TSV with columns: head<TAB>relation<TAB>tail<TAB>time")
    p.add_argument("--drop", action="store_true", help="Drop existing tables before build")
    p.add_argument("--batch", type=int, default=50000, help="Insert batch size")
    args = p.parse_args()
    init_db(args.db, drop=args.drop)
    bulk_load_tsv(args.db, args.tsv, batch=args.batch)
    # quick stats
    conn = _connect(args.db); cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM entities"); n_ent = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM edges"); n_edg = cur.fetchone()[0]
    conn.close()
    print(f"Built DB: {args.db}. Entities={n_ent}, Edges={n_edg}.")
