#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: temporal_kg_pack.py
import sqlite3, csv, re, calendar, argparse, os
from datetime import datetime, timezone
from typing import Optional, Iterable, Tuple, Union, List, Dict, Any

# ======================
# Time parsing
# ======================
ISO_DAY   = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_MONTH = re.compile(r"^\d{4}-\d{2}$")
ISO_YEAR  = re.compile(r"^\d{4}$")
ISO_DT_Z  = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

def _to_epoch(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def _month_end(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]

def smart_parse_time(time_input: Union[str, int, None], context_year: Optional[int] = None, context_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Parse various time formats intelligently, return ISO format string
    
    Supported input formats:
    - Standard ISO format: "2010-05", "2010", "2010-05-15"
    - Integer: 5 (month, needs context_year), 2010 (year)
    - Natural language: "May 2010", "2010年5month"
    - Time range: "2010-05" parse constraint
    
    Args:
        time_input: time input (string, integer or None)
        context_year: context year, used to parse separate month
        context_info: additional context info, may contain constraint info
    
    Returns:
        ISOformattimestring, like "2010-05" or None
    """
    if time_input is None:
        return None
    
    # convert to string
    time_str = str(time_input).strip()
    
    # 1. already is standard ISO format
    if ISO_DAY.match(time_str) or ISO_MONTH.match(time_str) or ISO_YEAR.match(time_str) or ISO_DT_Z.match(time_str):
        return time_str
    
    # 2. pure integer process
    if isinstance(time_input, int) or time_str.isdigit():
        num = int(time_str)
        
        # 2.1 year (1800-2100)
        if 1800 <= num <= 2100:
            return f"{num:04d}"
        
        # 2.2 month (1-12), need context year
        if 1 <= num <= 12:
            # try to extract year from context_info
            if context_year:
                return f"{context_year:04d}-{num:02d}"
            
            # try to find constraint info from context_info
            if context_info:
                constraints = context_info.get('constraints', [])
                for constraint in constraints:
                    # find constraint like "t1 = 2010-05" of constraint
                    match = re.search(r'(\d{4})-(\d{2})', constraint)
                    if match:
                        year = int(match.group(1))
                        return f"{year:04d}-{num:02d}"
                    # find separate year
                    match = re.search(r'\b(19\d{2}|20\d{2})\b', constraint)
                    if match:
                        year = int(match.group(1))
                        return f"{year:04d}-{num:02d}"
            
                print(f"⚠️ month {num} missing context year info")
            return None
        
        # 2.3 date possibility (YYYYMMDD or YYMMDD)
        if len(time_str) == 8:
            try:
                year = int(time_str[:4])
                month = int(time_str[4:6])
                day = int(time_str[6:8])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    return f"{year:04d}-{month:02d}-{day:02d}"
            except:
                pass
    
    # 3. process natural language month
    months_map = {
        'january': '01', 'jan': '01', 'february': '02', 'feb': '02',
        'march': '03', 'mar': '03', 'april': '04', 'apr': '04',
        'may': '05', 'june': '06', 'jun': '06', 'july': '07', 'jul': '07',
        'august': '08', 'aug': '08', 'september': '09', 'sep': '09', 'sept': '09',
        'october': '10', 'oct': '10', 'november': '11', 'nov': '11',
        'december': '12', 'dec': '12'
    }
    
    time_lower = time_str.lower()
    
    # 3.1 "May 2010" or "2010 May"
    for month_name, month_num in months_map.items():
        if month_name in time_lower:
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', time_str)
            if year_match:
                year = year_match.group(1)
                return f"{year}-{month_num}"
    
    # 4. process slash separated range "2010-05-01/2010-05-31"
    if '/' in time_str:
        parts = time_str.split('/')
        if len(parts) == 2:
            # use the start part of the range
            start_part = parts[0].strip()
            if ISO_DAY.match(start_part) or ISO_MONTH.match(start_part) or ISO_YEAR.match(start_part):
                return start_part
    
    # 5. try to extract any year info
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', time_str)
    if year_match:
        year = year_match.group(1)
        # try to extract month
        month_match = re.search(r'\b(0?[1-9]|1[0-2])\b', time_str)
        if month_match:
            month = int(month_match.group(1))
            return f"{year}-{month:02d}"
        return year
    
    return None


def parse_time_to_range(s: Union[str, int], context_year: Optional[int] = None, context_info: Optional[Dict[str, Any]] = None) -> Tuple[str, str, str, int, int]:
    """
    return (t_start_iso, t_end_iso, granularity, t_start_epoch, t_end_epoch)
    supported: YYYY-MM-DD, YYYY-MM, YYYY, YYYY-MM-DDTHH:MM:SSZ, and various non-standard format
    
    Args:
        s: time string or integer
        context_year: context year (used to parse separate month)
        context_info: additional context info
    """
    # first use smart parsing
    if not isinstance(s, str):
        parsed = smart_parse_time(s, context_year, context_info)
        if parsed is None:
            raise ValueError(f"Cannot parse time input: {s}")
        s = parsed
    else:
        s = s.strip()
        # try to smart parse to handle non-standard format
        if not (ISO_DAY.match(s) or ISO_MONTH.match(s) or ISO_YEAR.match(s) or ISO_DT_Z.match(s)):
            parsed = smart_parse_time(s, context_year, context_info)
            if parsed:
                s = parsed
    
    # standard format parsing
    if ISO_DAY.match(s):
        y, m, d = map(int, s.split("-"))
        start = datetime(y, m, d, 0, 0, 0, tzinfo=timezone.utc)
        end   = datetime(y, m, d, 23, 59, 59, tzinfo=timezone.utc)
        return (s, s, "day", _to_epoch(start), _to_epoch(end))
    if ISO_MONTH.match(s):
        y, m = map(int, s.split("-"))
        d_end = _month_end(y, m)
        start = datetime(y, m, 1, 0, 0, 0, tzinfo=timezone.utc)
        end   = datetime(y, m, d_end, 23, 59, 59, tzinfo=timezone.utc)
        return (f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{d_end:02d}", "month", _to_epoch(start), _to_epoch(end))
    if ISO_YEAR.match(s):
        y = int(s)
        start = datetime(y, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end   = datetime(y, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        return (f"{y:04d}-01-01", f"{y:04d}-12-31", "year", _to_epoch(start), _to_epoch(end))
    if ISO_DT_Z.match(s):
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        ep = _to_epoch(dt)
        iso = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return (iso, iso, "datetime", ep, ep)
    raise ValueError(f"Unrecognized time format: '{s}'")

# ======================
# connect with structure
# ======================
def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities(
  id   INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS edges(
  id            INTEGER PRIMARY KEY,
  head_id       INTEGER NOT NULL,
  relation      TEXT    NOT NULL,
  tail_id       INTEGER NOT NULL,
  t_start       TEXT    NOT NULL,
  t_end         TEXT    NOT NULL,
  granularity   TEXT    NOT NULL CHECK (granularity IN ('datetime','day','month','year')),
  t_start_epoch INTEGER NOT NULL,
  t_end_epoch   INTEGER NOT NULL,
  source        TEXT,
  FOREIGN KEY(head_id) REFERENCES entities(id),
  FOREIGN KEY(tail_id) REFERENCES entities(id),
  UNIQUE(head_id, relation, tail_id, t_start, t_end)
);
CREATE INDEX IF NOT EXISTS idx_entities_name     ON entities(name);
CREATE INDEX IF NOT EXISTS idx_edges_head        ON edges(head_id);
CREATE INDEX IF NOT EXISTS idx_edges_tail        ON edges(tail_id);
CREATE INDEX IF NOT EXISTS idx_edges_tstart      ON edges(t_start_epoch);
CREATE INDEX IF NOT EXISTS idx_edges_tend        ON edges(t_end_epoch);
CREATE INDEX IF NOT EXISTS idx_edges_head_tstart ON edges(head_id, t_start_epoch);
CREATE INDEX IF NOT EXISTS idx_edges_tail_tstart ON edges(tail_id, t_start_epoch);

CREATE TABLE IF NOT EXISTS dim_day (
  day_date TEXT PRIMARY KEY,
  day_start_epoch INTEGER NOT NULL,
  day_end_epoch   INTEGER NOT NULL,
  year INTEGER NOT NULL,
  month INTEGER NOT NULL,
  day INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dim_day_range ON dim_day(day_start_epoch, day_end_epoch);

CREATE TABLE IF NOT EXISTS dim_month (
  month_str TEXT PRIMARY KEY,
  month_start_epoch INTEGER NOT NULL,
  month_end_epoch   INTEGER NOT NULL,
  year INTEGER NOT NULL,
  month INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dim_month_range ON dim_month(month_start_epoch, month_end_epoch);
"""
# 放在 SCHEMA_SQL after definition
def _ensure_schema(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_SQL)
    conn.commit()

def init_new_db(db_path: str, overwrite: bool = False):
    if os.path.exists(db_path) and overwrite:
        os.remove(db_path)
    conn = _connect(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
def fill_dim_tables(db_path: str, start_year: int = 1990, end_year: int = 2035):
    conn = _connect(db_path); cur = conn.cursor()
    _ensure_schema(conn)  # new: ensure dim_month / dim_day exists

    months = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            d_end = _month_end(y, m)
            ms = datetime(y, m, 1, 0, 0, 0, tzinfo=timezone.utc)
            me = datetime(y, m, d_end, 23, 59, 59, tzinfo=timezone.utc)
            months.append((f"{y:04d}-{m:02d}", int(ms.timestamp()), int(me.timestamp()), y, m))
    cur.executemany("""
      INSERT OR IGNORE INTO dim_month(month_str, month_start_epoch, month_end_epoch, year, month)
      VALUES (?,?,?,?,?)
    """, months)

    days = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            for d in range(1, _month_end(y, m) + 1):
                ds = datetime(y, m, d, 0, 0, 0, tzinfo=timezone.utc)
                de = datetime(y, m, d, 23, 59, 59, tzinfo=timezone.utc)
                days.append((f"{y:04d}-{m:02d}-{d:02d}", int(ds.timestamp()), int(de.timestamp()), y, m, d))
    cur.executemany("""
      INSERT OR IGNORE INTO dim_day(day_date, day_start_epoch, day_end_epoch, year, month, day)
      VALUES (?,?,?,?,?,?)
    """, days)

    conn.commit(); conn.close()

# ======================
# entity tool
# ======================
def _get_or_create_entity_id(cur: sqlite3.Cursor, name: str) -> int:
    name = name.strip()
    cur.execute("SELECT id FROM entities WHERE name=?", (name,))
    row = cur.fetchone()
    if row: return row[0]
    cur.execute("INSERT INTO entities(name) VALUES (?)", (name,))
    return cur.lastrowid

def entity_id(db_path: str, query: Union[str,int]) -> int:
    """Get entity ID - Compatible with space and underscore formats"""
    conn = _connect(db_path); cur = conn.cursor()
    try:
        if isinstance(query, int):
            cur.execute("SELECT 1 FROM entities WHERE id=?", (query,))
            if cur.fetchone(): return query
            raise KeyError(f"entity id {query} not found")
        
        # try to use raw name first
        q = str(query).strip()
        cur.execute("SELECT id FROM entities WHERE name=?", (q,))
        row = cur.fetchone()
        if row: return row[0]
        
        # if not found, try another format
        if " " in q:
            q_alt = q.replace(" ", "_")
        elif "_" in q:
            q_alt = q.replace("_", " ")
        else:
            raise KeyError(f"entity name '{q}' not found")
        
        cur.execute("SELECT id FROM entities WHERE name=?", (q_alt,))
        row = cur.fetchone()
        if not row: raise KeyError(f"entity name '{q}' (also tried '{q_alt}') not found")
        return row[0]
    finally:
        conn.close()

def entity_name(db_path: str, eid: int) -> str:
    conn = _connect(db_path); cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM entities WHERE id=?", (eid,))
        row = cur.fetchone()
        if not row: raise KeyError(f"entity id {eid} not found")
        return row[0]
    finally:
        conn.close()

# ======================
# Read txt/tsv/csv
# ======================
def _detect_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        head = f.readline()
    if head.count("\t") >= head.count(","):
        return "\t"
    return ","

def load_txt_into_new_db(
    db_path: str,
    txt_path: str,
    delimiter: Optional[str] = None,
    batch: int = 50_000,
    source: Optional[str] = None
):
    if delimiter is None:
        delimiter = _detect_delimiter(txt_path)

    conn = _connect(db_path); cur = conn.cursor()
    _ensure_schema(conn)  # new: ensure table exists before loadingCreate
    with open(txt_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        buf: List[tuple] = []
        for i, row in enumerate(reader, 1):
            if not row or len(row) < 4:
                continue
            h_name, rel, t_name, time_str = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            if not h_name or not rel or not t_name or not time_str:
                continue
            h_id = _get_or_create_entity_id(cur, h_name)
            t_id = _get_or_create_entity_id(cur, t_name)
            t_start, t_end, gran, t0, t1 = parse_time_to_range(time_str)
            buf.append((h_id, rel, t_id, t_start, t_end, gran, t0, t1, source))
            if len(buf) >= batch:
                cur.executemany(
                    "INSERT OR IGNORE INTO edges(head_id, relation, tail_id, t_start, t_end, granularity, t_start_epoch, t_end_epoch, source) VALUES (?,?,?,?,?,?,?,?,?)",
                    buf
                )
                conn.commit(); buf.clear()
        if buf:
            cur.executemany(
                "INSERT OR IGNORE INTO edges(head_id, relation, tail_id, t_start, t_end, granularity, t_start_epoch, t_end_epoch, source) VALUES (?,?,?,?,?,?,?,?,?)",
                buf
            )
            conn.commit()
    conn.close()

# ======================
# retrieval: one hop (time filtering as needed)
# ======================
def retrieve_one_hop(
    db_path: str,
    query: Union[str,int],
    direction: str = "both",       # "out" | "in" | "both"
    return_as: str = "names",      # "names" | "ids"
    limit: Optional[int] = None,
    sort_by_time: bool = False,
    before: Optional[str] = None,  # 截止到某时：return t_end <= before(日末/该粒度末)
    after: Optional[str]  = None,  # 自某时起：return t_start >= after(日始/该粒度始)
    between: Optional[Tuple[str,str]] = None,  # 区间重叠
    same_day: Optional[str] = None,    # 'YYYY-MM-DD'
    same_month: Optional[str] = None,  # 'YYYY-MM'
    same_year: Optional[str] = None    # 'YYYY'
) -> List[Tuple[Union[str,int], str, Union[str,int], str, str]]:
    eid = entity_id(db_path, query)
    conn = _connect(db_path); cur = conn.cursor()

    # parse query range [q0,q1]（epoch）
    q0 = q1 = None
    if same_day:
        _, _, _, q0, q1 = parse_time_to_range(same_day)
    elif same_month:
        _, _, _, q0, q1 = parse_time_to_range(same_month)
    elif same_year:
        _, _, _, q0, q1 = parse_time_to_range(same_year)
    elif between:
        a, b = between
        a0 = parse_time_to_range(a)[3]
        b1 = parse_time_to_range(b)[4]
        q0, q1 = a0, b1
    elif before:
        q1 = parse_time_to_range(before)[4]
    elif after:
        q0 = parse_time_to_range(after)[3]

    # direction
    where_dir = []
    params: List[Union[int,str]] = []
    if direction in ("out", "both"):
        where_dir.append("head_id = ?"); params.append(eid)
    if direction in ("in", "both"):
        where_dir.append("tail_id = ?"); params.append(eid)
    where_dir_sql = " OR ".join(where_dir)
    if len(where_dir) > 1:
        where_dir_sql = f"({where_dir_sql})"

    # time
    where_time_sql = ""
    if (q0 is not None) and (q1 is not None):
        where_time_sql = "AND NOT (t_end_epoch < ? OR t_start_epoch > ?)"
        params.extend([q0, q1])
    elif q1 is not None:
        where_time_sql = "AND t_end_epoch <= ?"
        params.append(q1)
    elif q0 is not None:
        where_time_sql = "AND t_start_epoch >= ?"
        params.append(q0)

    order_sql = " ORDER BY t_start_epoch ASC, t_end_epoch ASC" if sort_by_time else " ORDER BY t_start_epoch DESC, t_end_epoch DESC"
    limit_sql = f" LIMIT {int(limit)}" if limit else ""

    sql = f"""
      SELECT head_id, relation, tail_id, t_start, t_end, granularity
      FROM edges
      WHERE {where_dir_sql} {where_time_sql}
      {order_sql}
      {limit_sql}
    """.strip()
    cur.execute(sql, tuple(params))
    rows = cur.fetchall()

    if return_as == "ids":
        conn.close()
        return [(h, r, t, f"{ts}~{te}", g) for (h, r, t, ts, te, g) in rows]

    # id -> name
    ids = set()
    for h, _, t, _, _, _ in rows:
        ids.add(h); ids.add(t)
    id_map: Dict[int, str] = {}
    if ids:
        qmarks = ",".join("?" for _ in ids)
        cur.execute(f"SELECT id, name FROM entities WHERE id IN ({qmarks})", tuple(ids))
        id_map = {i: n for i, n in cur.fetchall()}
    conn.close()

    out = []
    for h, r, t, ts, te, g in rows:
        out.append((id_map.get(h, h), r, id_map.get(t, t), f"{ts}~{te}", g))
    return out

# ======================
# full graph query: by day / by month
# ======================
def events_on_day(db_path: str, day: str, limit: Optional[int] = None, sort_by_time: bool = True):
    conn = _connect(db_path); cur = conn.cursor()
    _ensure_schema(conn) 
    cur.execute("SELECT day_start_epoch, day_end_epoch FROM dim_day WHERE day_date=?", (day,))
    row = cur.fetchone()
    if not row:
        conn.close(); raise KeyError(f"dim_day missing for {day}. Run build-dims first.")
    q0, q1 = row
    order_sql = " ORDER BY e.t_start_epoch ASC, e.t_end_epoch ASC" if sort_by_time else ""
    limit_sql = f" LIMIT {int(limit)}" if limit else ""
    sql = f"""
      SELECT h.name, e.relation, t.name, e.t_start, e.t_end
      FROM edges e
      JOIN entities h ON h.id = e.head_id
      JOIN entities t ON t.id = e.tail_id
      WHERE NOT (e.t_end_epoch < ? OR e.t_start_epoch > ?)
      {order_sql}
      {limit_sql}
    """
    cur.execute(sql, (q0, q1))
    rows = cur.fetchall()
    conn.close()
    return rows

def events_in_month(db_path: str, month_str: str, limit: Optional[int] = None, sort_by_time: bool = True):
    conn = _connect(db_path); cur = conn.cursor()
    cur.execute("SELECT month_start_epoch, month_end_epoch FROM dim_month WHERE month_str=?", (month_str,))
    row = cur.fetchone()
    if not row:
        conn.close(); raise KeyError(f"dim_month missing for {month_str}. Run build-dims first.")
    q0, q1 = row
    order_sql = " ORDER BY e.t_start_epoch ASC, e.t_end_epoch ASC" if sort_by_time else ""
    limit_sql = f" LIMIT {int(limit)}" if limit else ""
    sql = f"""
      SELECT h.name, e.relation, t.name, e.t_start, e.t_end
      FROM edges e
      JOIN entities h ON h.id = e.head_id
      JOIN entities t ON t.id = e.tail_id
      WHERE NOT (e.t_end_epoch < ? OR e.t_start_epoch > ?)
      {order_sql}
      {limit_sql}
    """
    cur.execute(sql, (q0, q1))
    rows = cur.fetchall()
    conn.close()
    return rows

# ======================
# CLI (Command Line Interface)
# ======================
def main():
    ap = argparse.ArgumentParser(description="Build a NEW temporal-KG DB from txt/tsv/csv and query with time filters.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_init = sub.add_parser("init", help="Create a NEW db file (schema only).")
    sp_init.add_argument("--db", required=True)
    sp_init.add_argument("--overwrite", action="store_true")

    sp_dims = sub.add_parser("build-dims", help="Fill dim_day and dim_month tables.")
    sp_dims.add_argument("--db", required=True)
    sp_dims.add_argument("--start-year", type=int, default=1990)
    sp_dims.add_argument("--end-year", type=int, default=2035)

    sp_load = sub.add_parser("load", help="Load 4-col txt/tsv/csv: head, relation, tail, time")
    sp_load.add_argument("--db", required=True)
    sp_load.add_argument("--in", dest="in_path", required=True)
    sp_load.add_argument("--delimiter", choices=[",", "\\t"], default=None)
    sp_load.add_argument("--batch", type=int, default=50000)
    sp_load.add_argument("--source", type=str, default=None)

    sp_day = sub.add_parser("events-day", help="List all events on a day (whole graph).")
    sp_day.add_argument("--db", required=True)
    sp_day.add_argument("--day", required=True)      # YYYY-MM-DD
    sp_day.add_argument("--limit", type=int, default=None)
    sp_day.add_argument("--no-sort", action="store_true")

    sp_month = sub.add_parser("events-month", help="List all events in a month (whole graph).")
    sp_month.add_argument("--db", required=True)
    sp_month.add_argument("--month", required=True)  # YYYY-MM
    sp_month.add_argument("--limit", type=int, default=None)
    sp_month.add_argument("--no-sort", action="store_true")

    sp_onehop = sub.add_parser("one-hop", help="Retrieve one-hop edges for an entity with time filters.")
    sp_onehop.add_argument("--db", required=True)
    sp_onehop.add_argument("--entity", required=True)  # name or numeric id
    sp_onehop.add_argument("--direction", choices=["out","in","both"], default="both")
    sp_onehop.add_argument("--return-as", choices=["names","ids"], default="names")
    sp_onehop.add_argument("--limit", type=int, default=None)
    sp_onehop.add_argument("--sort", action="store_true")
    # time filters
    sp_onehop.add_argument("--before", type=str, default=None)
    sp_onehop.add_argument("--after", type=str, default=None)
    sp_onehop.add_argument("--between", type=str, nargs=2, default=None)
    sp_onehop.add_argument("--same-day", type=str, default=None)
    sp_onehop.add_argument("--same-month", type=str, default=None)
    sp_onehop.add_argument("--same-year", type=str, default=None)

    args = ap.parse_args()

    if args.cmd == "init":
        init_new_db(args.db, overwrite=args.overwrite)
        print("OK: schema ready.")

    elif args.cmd == "build-dims":
        fill_dim_tables(args.db, start_year=args.start_year, end_year=args.end_year)
        print(f"OK: dim tables filled for {args.start_year}-{args.end_year}.")

    elif args.cmd == "load":
        delim = None if args.delimiter is None else ("\t" if args.delimiter == "\\t" else args.delimiter)
        load_txt_into_new_db(args.db, args.in_path, delimiter=delim, batch=args.batch, source=args.source)
        conn = _connect(args.db); cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM entities"); n_ent = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM edges"); n_edg = cur.fetchone()[0]
        conn.close()
        print(f"Loaded. Entities={n_ent}, Edges={n_edg}.")

    elif args.cmd == "events-day":
        rows = events_on_day(args.db, args.day, limit=args.limit, sort_by_time=(not args.no_sort))
        for h, r, t, ts, te in rows:
            print(f"{h}\t{r}\t{t}\t{ts}~{te}")

    elif args.cmd == "events-month":
        rows = events_in_month(args.db, args.month, limit=args.limit, sort_by_time=(not args.no_sort))
        for h, r, t, ts, te in rows:
            print(f"{h}\t{r}\t{t}\t{ts}~{te}")

    elif args.cmd == "one-hop":
        ent: Union[str,int]
        try:
            ent = int(args.entity)
        except:
            ent = args.entity
        rows = retrieve_one_hop(
            db_path=args.db,
            query=ent,
            direction=args.direction,
            return_as=args.return_as,
            limit=args.limit,
            sort_by_time=args.sort,
            before=args.before,
            after=args.after,
            between=tuple(args.between) if args.between else None,
            same_day=args.same_day,
            same_month=args.same_month,
            same_year=args.same_year
        )
        for row in rows:
            if args.return_as == "ids":
                h, r, t, ts_te, g = row
                print(f"{h}\t{r}\t{t}\t{ts_te}\t{g}")
            else:
                h, r, t, ts_te, g = row
                print(f"{h}\t{r}\t{t}\t{ts_te}\t{g}")

if __name__ == "__main__":
    main()
    # from tempkg_sqlite import retrieve_one_hop

