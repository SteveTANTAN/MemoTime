#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: temporal_kg_toolkit.py
"""
Temporal Knowledge Graph Toolkit
Integrate the tool functions in temporal_kg_pack.py and temporal_kg_demo.py 
Provide flexible KG query functions for each subquestion
"""

import sqlite3
import re
import calendar
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass

# ======================
# Time parsing tools
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
    Smartly parse various time formats, return ISO format string
    
    Supported input formats:        
    - Standard ISO format: "2010-05", "2010", "2010-05-15"
    - Integer: 5 (month, need context_year), 2010 (year)
    - Natural language: "May 2010", "2010 May"
    - Time range: "2010-05" parse constraint
    
    Args:   
        time_input: time input (string, integer or None)
        context_year: context year, for parsing separate month
        context_info: additional context information, may contain constraint information
    
    Returns:
        ISO format time string, like "2010-05" or None
    """
    if time_input is None:
        return None
    
    # convert to string
    time_str = str(time_input).strip()
    
    # 1. already standard ISO format
    if ISO_DAY.match(time_str) or ISO_MONTH.match(time_str) or ISO_YEAR.match(time_str) or ISO_DT_Z.match(time_str):
        return time_str
    
    # 2. pure integer processing
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
            
            # try to find constraint information from context_info
            if context_info:
                constraints = context_info.get('constraints', [])
                for constraint in constraints:
                    # find constraint like "t1 = 2010-05"
                    match = re.search(r'(\d{4})-(\d{2})', constraint)
                    if match:
                        year = int(match.group(1))
                        return f"{year:04d}-{num:02d}"
                    # find separate year
                    match = re.search(r'\b(19\d{2}|20\d{2})\b', constraint)
                    if match:
                        year = int(match.group(1))
                        return f"{year:04d}-{num:02d}"
            
            print(f"⚠️ month {num} missing context year information")
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
    
    # 5. try to extract any year information
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
# Database connection tool  
# ======================
def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

# ======================
# Entity tool   
# ======================
def entity_id(db_path: str, query: Union[str, int]) -> int:
    """Get entity ID - compatible with space and underscore format"""
    conn = _connect(db_path)
    cur = conn.cursor()
    try:
        if isinstance(query, int):
            cur.execute("SELECT 1 FROM entities WHERE id=?", (query,))
            if cur.fetchone():
                return query
            raise KeyError(f"entity id {query} not found")
        
        # try to extract original name
        q = str(query).strip()
        cur.execute("SELECT id FROM entities WHERE name=?", (q,))
        row = cur.fetchone()
        if row:
            return row[0]
        
        # if not found, try another format
        # if contains space, try to replace with underscore
        if " " in q:
            q_alt = q.replace(" ", "_")
        # if contains underscore, try to replace with space
        elif "_" in q:
            q_alt = q.replace("_", " ")
        else:
            raise KeyError(f"entity name '{q}' not found")
        
        cur.execute("SELECT id FROM entities WHERE name=?", (q_alt,))
        row = cur.fetchone()
        if not row:
            raise KeyError(f"entity name '{q}' (also tried '{q_alt}') not found")
        return row[0]
    finally:
        conn.close()

def entity_name(db_path: str, eid: int) -> str:
    """Get entity name"""
    conn = _connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM entities WHERE id=?", (eid,))
        row = cur.fetchone()
        if not row:
            raise KeyError(f"entity id {eid} not found")
        return row[0]
    finally:
        conn.close()

# ======================
# Query result data structure
# ======================
@dataclass
class TemporalEdge:
    """Time edge data structure"""
    head: str
    relation: str
    tail: str
    time_start: str
    time_end: str
    granularity: str
    head_id: int
    tail_id: int
    time_start_epoch: int
    time_end_epoch: int
    source: Optional[str] = None

@dataclass
class QueryResult:
    """Query result data structure"""
    def __init__(self, edges: List[TemporalEdge] = None, total_count: int = 0, query_params: Dict[str, Any] = None, metadata: Dict[str, Any] = None):
        self.edges = edges or []
        self.total_count = total_count
        self.query_params = query_params or {}
        self.metadata = metadata or {}

# ======================
# Core query tool
# ======================
class TemporalKGQuery:
    """Time knowledge graph query tool"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def retrieve_one_hop(
        self,
        query: Union[str, int],
        direction: str = "both",       # "out" | "in" | "both"
        return_as: str = "names",      # "names" | "ids"
        limit: Optional[int] = None,
        sort_by_time: bool = False,
        before: Optional[str] = None,  # before to some time
        after: Optional[str] = None,   # after to some time
        between: Optional[Tuple[str, str]] = None,  # time range overlap
        same_day: Optional[str] = None,    # 'YYYY-MM-DD'
        same_month: Optional[str] = None,  # 'YYYY-MM'
        same_year: Optional[str] = None    # 'YYYY'
    ) -> QueryResult:
        """One hop query"""
        eid = entity_id(self.db_path, query)
        conn = _connect(self.db_path)
        cur = conn.cursor()

        # parse query time range [q0,q1] (epoch)
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
        params: List[Union[int, str]] = []
        if direction in ("out", "both"):
            where_dir.append("head_id = ?")
            params.append(eid)
        if direction in ("in", "both"):
            where_dir.append("tail_id = ?")
            params.append(eid)
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
        # if limit is not specified, return all data; if limit is specified, use the specified limit
        limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""

        sql = f"""
          SELECT head_id, relation, tail_id, t_start, t_end, granularity, t_start_epoch, t_end_epoch
          FROM edges
          WHERE {where_dir_sql} {where_time_sql}
          {order_sql}
          {limit_sql}
        """.strip()
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()

        if return_as == "ids":
            conn.close()
            edges = [TemporalEdge(
                head=str(h), relation=r, tail=str(t),
                time_start=ts, time_end=te, granularity=g,
                head_id=h, tail_id=t, time_start_epoch=t0, time_end_epoch=t1,
                
            ) for (h, r, t, ts, te, g, t0, t1) in rows]
            return QueryResult(edges=edges, total_count=len(edges), query_params={
                "query": query, "direction": direction, "return_as": return_as,
                "limit": limit, "sort_by_time": sort_by_time, "before": before,
                "after": after, "between": between, "same_day": same_day,
                "same_month": same_month, "same_year": same_year
            })

        # id -> name
        ids = set()
        for h, _, t, _, _, _, _, _ in rows:
            ids.add(h)
            ids.add(t)
        id_map: Dict[int, str] = {}
        if ids:
            qmarks = ",".join("?" for _ in ids)
            cur.execute(f"SELECT id, name FROM entities WHERE id IN ({qmarks})", tuple(ids))
            id_map = {i: n for i, n in cur.fetchall()}
        conn.close()

        edges = []
        for h, r, t, ts, te, g, t0, t1 in rows:
            edges.append(TemporalEdge(
                head=id_map.get(h, str(h)), relation=r, tail=id_map.get(t, str(t)),
                time_start=ts, time_end=te, granularity=g,
                head_id=h, tail_id=t, time_start_epoch=t0, time_end_epoch=t1
            ))
        
        return QueryResult(edges=edges, total_count=len(edges), query_params={
            "query": query, "direction": direction, "return_as": return_as,
            "limit": limit, "sort_by_time": sort_by_time, "before": before,
            "after": after, "between": between, "same_day": same_day,
            "same_month": same_month, "same_year": same_year
        })

    def events_on_day(self, day: str, limit: Optional[int] = None, sort_by_time: bool = True) -> QueryResult:
        """Query all events on a day"""
        conn = _connect(self.db_path)
        cur = conn.cursor()
        
        # use dim_day table to get time range
        cur.execute("SELECT day_start_epoch, day_end_epoch FROM dim_day WHERE day_date=?", (day,))
        row = cur.fetchone()
        if not row:
            conn.close()
            raise KeyError(f"dim_day missing for {day}. Run build-dims first.")
        q0, q1 = row
        
        order_sql = " ORDER BY e.t_start_epoch ASC, e.t_end_epoch ASC" if sort_by_time else ""
        limit_sql = f" LIMIT {int(limit)}" if limit else ""
        sql = f"""
          SELECT h.name, e.relation, t.name, e.t_start, e.t_end, e.granularity, 
                 e.head_id, e.tail_id, e.t_start_epoch, e.t_end_epoch, e.source
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
        
        edges = [TemporalEdge(
            head=h, relation=r, tail=t, time_start=ts, time_end=te, granularity=g,
            head_id=h_id, tail_id=t_id, time_start_epoch=t0, time_end_epoch=t1
        ) for (h, r, t, ts, te, g, h_id, t_id, t0, t1) in rows]
        
        return QueryResult(edges=edges, total_count=len(edges), query_params={
            "day": day, "limit": limit, "sort_by_time": sort_by_time
        })

    def events_in_month(self, month_str: str, limit: Optional[int] = None, sort_by_time: bool = True) -> QueryResult:
        """Query all events in a month"""
        conn = _connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute("SELECT month_start_epoch, month_end_epoch FROM dim_month WHERE month_str=?", (month_str,))
        row = cur.fetchone()
        if not row:
            conn.close()
            raise KeyError(f"dim_month missing for {month_str}. Run build-dims first.")
        q0, q1 = row
        
        order_sql = " ORDER BY e.t_start_epoch ASC, e.t_end_epoch ASC" if sort_by_time else ""
        limit_sql = f" LIMIT {int(limit)}" if limit else ""
        sql = f"""
          SELECT h.name, e.relation, t.name, e.t_start, e.t_end, e.granularity,
                 e.head_id, e.tail_id, e.t_start_epoch, e.t_end_epoch, e.source
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
        
        edges = [TemporalEdge(
            head=h, relation=r, tail=t, time_start=ts, time_end=te, granularity=g,
            head_id=h_id, tail_id=t_id, time_start_epoch=t0, time_end_epoch=t1
        ) for (h, r, t, ts, te, g, h_id, t_id, t0, t1) in rows]
        
        return QueryResult(edges=edges, total_count=len(edges), query_params={
            "month": month_str, "limit": limit, "sort_by_time": sort_by_time
        })

    def events_in_year(self, year_str: str, limit: Optional[int] = None, sort_by_time: bool = True) -> QueryResult:
        """Query all events in a year"""
        conn = _connect(self.db_path)
        cur = conn.cursor()
        
        # parse year string, generate year range
        try:
            if isinstance(year_str, int):
                year = year_str
            else:
                year = int(year_str.strip())
            
            # calculate the start and end epoch of the year
            start_dt = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            end_dt = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            q0 = _to_epoch(start_dt)
            q1 = _to_epoch(end_dt)
        except (ValueError, AttributeError) as e:
            conn.close()
            raise ValueError(f"Invalid year format: {year_str}. Expected format: YYYY")
        
        order_sql = " ORDER BY e.t_start_epoch ASC, e.t_end_epoch ASC" if sort_by_time else ""
        limit_sql = f" LIMIT {int(limit)}" if limit else ""
        sql = f"""
          SELECT h.name, e.relation, t.name, e.t_start, e.t_end, e.granularity,
                 e.head_id, e.tail_id, e.t_start_epoch, e.t_end_epoch, e.source
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
        
        edges = [TemporalEdge(
            head=h, relation=r, tail=t, time_start=ts, time_end=te, granularity=g,
            head_id=h_id, tail_id=t_id, time_start_epoch=t0, time_end_epoch=t1
        ) for (h, r, t, ts, te, g, h_id, t_id, t0, t1) in rows]
        
        return QueryResult(edges=edges, total_count=len(edges), query_params={
            "year": year_str, "limit": limit, "sort_by_time": sort_by_time
        })

    def find_entities_by_name_pattern(self, pattern: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find entities by name pattern"""
        conn = _connect(self.db_path)
        cur = conn.cursor()
        
        limit_sql = f" LIMIT {int(limit)}" if limit else ""
        sql = f"""
          SELECT id, name FROM entities 
          WHERE name LIKE ? 
          ORDER BY name
          {limit_sql}
        """
        cur.execute(sql, (f"%{pattern}%",))
        rows = cur.fetchall()
        conn.close()
        
        return [{"id": row[0], "name": row[1]} for row in rows]

    def get_entity_statistics(self, entity_id: int) -> Dict[str, Any]:
        """Get entity statistics"""
        conn = _connect(self.db_path)
        cur = conn.cursor()
        
        # get entity name
        cur.execute("SELECT name FROM entities WHERE id=?", (entity_id,))
        name_row = cur.fetchone()
        if not name_row:
            conn.close()
            raise KeyError(f"Entity {entity_id} not found")
        entity_name = name_row[0]
        
        # count out edges and in edges
        cur.execute("SELECT COUNT(*) FROM edges WHERE head_id=?", (entity_id,))
        out_edges = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM edges WHERE tail_id=?", (entity_id,))
        in_edges = cur.fetchone()[0]
        
        # get time range
        cur.execute("""
          SELECT MIN(t_start_epoch), MAX(t_end_epoch) 
          FROM edges 
          WHERE head_id=? OR tail_id=?
        """, (entity_id, entity_id))
        time_row = cur.fetchone()
        min_time = time_row[0] if time_row[0] else None
        max_time = time_row[1] if time_row[1] else None
        
        conn.close()
        
        return {
            "id": entity_id,
            "name": entity_name,
            "out_edges": out_edges,
            "in_edges": in_edges,
            "total_edges": out_edges + in_edges,
            "min_time": min_time,
            "max_time": max_time
        }
    
    def find_direct_connection(
        self,
        entity1: Union[str, int],
        entity2: Union[str, int],
        relation_types: Optional[List[str]] = None,
        direction: str = "both",  # "forward" | "backward" | "both"
        limit: Optional[int] = None,
        sort_by_time: bool = False,
        before: Optional[str] = None,
        after: Optional[str] = None,
        between: Optional[Tuple[str, str]] = None,
        same_day: Optional[str] = None,
        same_month: Optional[str] = None,
        same_year: Optional[str] = None
    ) -> QueryResult:
        """Search direct connection between two entities"""
        eid1 = entity_id(self.db_path, entity1)
        eid2 = entity_id(self.db_path, entity2)
        
        if eid1 is None or eid2 is None:
            return QueryResult(edges=[], metadata={"error": "Entity not found"})
        
        conn = _connect(self.db_path)
        cur = conn.cursor()
        
        # parse query time range [q0,q1] (epoch)
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
        
        # build direction conditions
        direction_conditions = []
        params = []
        
        if direction in ("forward", "both"):
            # entity1 -> entity2
            direction_conditions.append("(e.head_id = ? AND e.tail_id = ?)")
            params.extend([eid1, eid2])
        
        if direction in ("backward", "both"):
            # entity2 -> entity1
            direction_conditions.append("(e.head_id = ? AND e.tail_id = ?)")
            params.extend([eid2, eid1])
        
        if not direction_conditions:
            return QueryResult(edges=[], metadata={"error": "Invalid direction"})
        
        # build SQL query
        where_conditions = [f"({' OR '.join(direction_conditions)})"]
        
        # add relation type filter
        if relation_types:
            relation_placeholders = ','.join(['?' for _ in relation_types])
            where_conditions.append(f"e.relation IN ({relation_placeholders})")
            params.extend(relation_types)
        
        # add time filter
        if q0 is not None:
            where_conditions.append("e.t_start_epoch >= ?")
            params.append(q0)
        if q1 is not None:
            where_conditions.append("e.t_end_epoch <= ?")
            params.append(q1)
        
        where_clause = " AND ".join(where_conditions)
        
        # build order
        order_clause = ""
        if sort_by_time:
            order_clause = "ORDER BY e.t_start_epoch ASC"
        
        # build limit
        limit_clause = ""
        if limit:
            limit_clause = f"LIMIT {limit}"
        
        sql = f"""
        SELECT DISTINCT
          e.head_id, e.relation, e.tail_id, e.t_start, e.t_end, e.granularity,
          e.t_start_epoch, e.t_end_epoch, e.source
        FROM edges e
        WHERE {where_clause}
        {order_clause}
        {limit_clause}
        """
        
        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
            
            # get entity name mapping
            entity_ids = set()
            for row in rows:
                entity_ids.add(row[0])  # head_id
                entity_ids.add(row[2])  # tail_id
            
            id_map = {}
            if entity_ids:
                id_placeholders = ','.join(['?' for _ in entity_ids])
                cur.execute(f"SELECT id, name FROM entities WHERE id IN ({id_placeholders})", list(entity_ids))
                id_map = dict(cur.fetchall())
            
            # build result
            edges = []
            for row in rows:
                h, r, t, ts, te, g, t0, t1, s = row
                edge = TemporalEdge(
                    head=id_map.get(h, str(h)),
                    relation=r,
                    tail=id_map.get(t, str(t)),
                    time_start=ts,
                    time_end=te,
                    granularity=g,
                    head_id=h,
                    tail_id=t,
                    time_start_epoch=t0,
                    time_end_epoch=t1,
                    
                )
                edges.append(edge)
            
            conn.close()
            return QueryResult(edges=edges, metadata={
                "total_edges": len(edges),
                "entity1": entity1,
                "entity2": entity2,
                "direction": direction,
                "relation_types": relation_types
            })
            
        except Exception as e:
            conn.close()
            return QueryResult(edges=[], metadata={"error": str(e)})

# ======================
# Advanced temporal operation tool
# ======================
class AdvancedTemporalQuery:
    """Advanced temporal query tool, support complex temporal relationship operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.base_query = TemporalKGQuery(db_path)
    
    def find_temporal_sequence(self, entity: Union[str, int], relation: str, 
                             start_time: str, end_time: str) -> QueryResult:
        """Find temporal sequence"""
        return self.base_query.retrieve_one_hop(
            query=entity,
            direction="out",
            between=(start_time, end_time),
            sort_by_time=True
        )
    
    def find_entities_after_time(self, time_point: str, limit: Optional[int] = None) -> QueryResult:
        """Find all entities after a time point"""
        conn = _connect(self.db_path)
        cur = conn.cursor()
        
        _, _, _, q0, _ = parse_time_to_range(time_point)
        
        limit_sql = f" LIMIT {int(limit)}" if limit else ""
        sql = f"""
          SELECT h.name, e.relation, t.name, e.t_start, e.t_end, e.granularity,
                 e.head_id, e.tail_id, e.t_start_epoch, e.t_end_epoch, e.source
          FROM edges e
          JOIN entities h ON h.id = e.head_id
          JOIN entities t ON t.id = e.tail_id
          WHERE e.t_start_epoch >= ?
          ORDER BY e.t_start_epoch ASC
          {limit_sql}
        """
        cur.execute(sql, (q0,))
        rows = cur.fetchall()
        conn.close()
        
        edges = [TemporalEdge(
            head=h, relation=r, tail=t, time_start=ts, time_end=te, granularity=g,
            head_id=h_id, tail_id=t_id, time_start_epoch=t0, time_end_epoch=t1
        ) for (h, r, t, ts, te, g, h_id, t_id, t0, t1) in rows]
        
        return QueryResult(edges=edges, total_count=len(edges), query_params={
            "time_point": time_point, "limit": limit
        })
    
    def find_entities_before_time(self, time_point: str, limit: Optional[int] = None) -> QueryResult:
        """Find all entities before a time point"""
        conn = _connect(self.db_path)
        cur = conn.cursor()
        
        _, _, _, _, q1 = parse_time_to_range(time_point)
        
        limit_sql = f" LIMIT {int(limit)}" if limit else ""
        sql = f"""
          SELECT h.name, e.relation, t.name, e.t_start, e.t_end, e.granularity,
                 e.head_id, e.tail_id, e.t_start_epoch, e.t_end_epoch, e.source
          FROM edges e
          JOIN entities h ON h.id = e.head_id
          JOIN entities t ON t.id = e.tail_id
          WHERE e.t_end_epoch <= ?
          ORDER BY e.t_end_epoch DESC
          {limit_sql}
        """
        cur.execute(sql, (q1,))
        rows = cur.fetchall()
        conn.close()
        
        edges = [TemporalEdge(
            head=h, relation=r, tail=t, time_start=ts, time_end=te, granularity=g,
            head_id=h_id, tail_id=t_id, time_start_epoch=t0, time_end_epoch=t1
        ) for (h, r, t, ts, te, g, h_id, t_id, t0, t1) in rows]
        
        return QueryResult(edges=edges, total_count=len(edges), query_params={
            "time_point": time_point, "limit": limit
        })

    def find_before_last(self, entity: Union[str, int], reference_time: str, 
                        limit: Optional[int] = None) -> QueryResult:
        """Find the last event before the reference time"""
        conn = _connect(self.db_path)
        cur = conn.cursor()
        
        eid = entity_id(self.db_path, entity)
        _, _, _, _, ref_epoch = parse_time_to_range(reference_time)
        
        limit_sql = f" LIMIT {int(limit)}" if limit else " LIMIT 50"
        sql = f"""
          SELECT h.name, e.relation, t.name, e.t_start, e.t_end, e.granularity,
                 e.head_id, e.tail_id, e.t_start_epoch, e.t_end_epoch, e.source
          FROM edges e
          JOIN entities h ON h.id = e.head_id
          JOIN entities t ON t.id = e.tail_id
          WHERE (e.head_id = ? OR e.tail_id = ?) AND e.t_end_epoch < ?
          ORDER BY e.t_end_epoch DESC
          {limit_sql}
        """
        cur.execute(sql, (eid, eid, ref_epoch))
        rows = cur.fetchall()
        conn.close()
        
        edges = [TemporalEdge(
            head=h, relation=r, tail=t, time_start=ts, time_end=te, granularity=g,
            head_id=h_id, tail_id=t_id, time_start_epoch=t0, time_end_epoch=t1
        ) for (h, r, t, ts, te, g, h_id, t_id, t0, t1) in rows]
        
        return QueryResult(edges=edges, total_count=len(edges), query_params={
            "entity": entity, "reference_time": reference_time, "operation": "before_last", "limit": limit
        })

    def find_after_first(self, entity: Union[str, int], reference_time: str, 
                        limit: Optional[int] = None) -> QueryResult:
        """Find the first event after the reference time"""
        conn = _connect(self.db_path)
        cur = conn.cursor()
        
        eid = entity_id(self.db_path, entity)
        _, _, _, ref_epoch, _ = parse_time_to_range(reference_time)
        
        # optimize time filter: before use > instead of >=, increase return number
        limit_sql = f" LIMIT {int(limit)}" if limit else " LIMIT 100"
        sql = f"""
          SELECT h.name, e.relation, t.name, e.t_start, e.t_end, e.granularity,
                 e.head_id, e.tail_id, e.t_start_epoch, e.t_end_epoch, e.source
          FROM edges e
          JOIN entities h ON h.id = e.head_id
          JOIN entities t ON t.id = e.tail_id
          WHERE (e.head_id = ? OR e.tail_id = ?) AND e.t_start_epoch > ?
          ORDER BY e.t_start_epoch ASC
          {limit_sql}
        """
        cur.execute(sql, (eid, eid, ref_epoch))
        rows = cur.fetchall()
        conn.close()
        
        edges = [TemporalEdge(
            head=h, relation=r, tail=t, time_start=ts, time_end=te, granularity=g,
            head_id=h_id, tail_id=t_id, time_start_epoch=t0, time_end_epoch=t1
        ) for (h, r, t, ts, te, g, h_id, t_id, t0, t1) in rows]
        
        return QueryResult(edges=edges, total_count=len(edges), query_params={
            "entity": entity, "reference_time": reference_time, "operation": "after_first", "limit": limit
        })

    def find_between_times(self, entity: Union[str, int], start_time: str, end_time: str,
                          limit: Optional[int] = None) -> QueryResult:
        """Find events between two time points"""
        return self.base_query.retrieve_one_hop(
            query=entity,
            direction="both",
            between=(start_time, end_time),
            sort_by_time=True,
            limit=limit
        )

    def find_temporal_neighbors(self, entity: Union[str, int], time_operation: str,
                               reference_time: str, limit: Optional[int] = None) -> QueryResult:
        """Find temporal neighbors (support multiple time operations)"""
        if time_operation == "before":
            return self.find_entities_before_time(reference_time, limit)
        elif time_operation == "after":
            return self.find_entities_after_time(reference_time, limit)
        elif time_operation == "before_last":
            return self.find_before_last(entity, reference_time, limit)
        elif time_operation == "after_first":
            return self.find_after_first(entity, reference_time, limit)
        elif time_operation == "between":
            # need two time points, here assume reference_time contains two time points
            times = reference_time.split(",")
            if len(times) == 2:
                return self.find_between_times(entity, times[0].strip(), times[1].strip(), limit)
            else:
                raise ValueError("between operation requires two time points separated by comma")
        else:
            raise ValueError(f"Unsupported time operation: {time_operation}")

    def find_chronological_sequence(self, entities: List[Union[str, int]], 
                                   start_time: str, end_time: str,
                                   limit: Optional[int] = None) -> QueryResult:
        """Find temporal sequence of multiple entities"""
        conn = _connect(self.db_path)
        cur = conn.cursor()
        
        # get entity ID
        entity_ids = []
        for entity in entities:
            try:
                eid = entity_id(self.db_path, entity)
                entity_ids.append(eid)
            except KeyError:
                continue
        
        if not entity_ids:
            return QueryResult(edges=[], total_count=0, query_params={
                "entities": entities, "start_time": start_time, "end_time": end_time
            })
        
        _, _, _, start_epoch, _ = parse_time_to_range(start_time)
        _, _, _, _, end_epoch = parse_time_to_range(end_time)
        
        # build query conditions
        entity_conditions = " OR ".join(["(e.head_id = ? OR e.tail_id = ?)" for _ in entity_ids])
        params = []
        for eid in entity_ids:
            params.extend([eid, eid])
        params.extend([start_epoch, end_epoch])
        
        limit_sql = f" LIMIT {int(limit)}" if limit else ""
        sql = f"""
          SELECT h.name, e.relation, t.name, e.t_start, e.t_end, e.granularity,
                 e.head_id, e.tail_id, e.t_start_epoch, e.t_end_epoch, e.source
          FROM edges e
          JOIN entities h ON h.id = e.head_id
          JOIN entities t ON t.id = e.tail_id
          WHERE ({entity_conditions}) AND NOT (e.t_end_epoch < ? OR e.t_start_epoch > ?)
          ORDER BY e.t_start_epoch ASC, e.t_end_epoch ASC
          {limit_sql}
        """
        cur.execute(sql, params)
        rows = cur.fetchall()
        conn.close()
        
        edges = [TemporalEdge(
            head=h, relation=r, tail=t, time_start=ts, time_end=te, granularity=g,
            head_id=h_id, tail_id=t_id, time_start_epoch=t0, time_end_epoch=t1
        ) for (h, r, t, ts, te, g, h_id, t_id, t0, t1) in rows]
        
        return QueryResult(edges=edges, total_count=len(edges), query_params={
            "entities": entities, "start_time": start_time, "end_time": end_time, "limit": limit
        })

    def find_time_gaps(self, entity: Union[str, int], min_gap_days: int = 30,
                      limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find time gaps in entity activities"""
        conn = _connect(self.db_path)
        cur = conn.cursor()
        
        eid = entity_id(self.db_path, entity)
        
        sql = """
          SELECT h.name, e.relation, t.name, e.t_start, e.t_end, e.granularity,
                 e.head_id, e.tail_id, e.t_start_epoch, e.t_end_epoch, e.source
          FROM edges e
          JOIN entities h ON h.id = e.head_id
          JOIN entities t ON t.id = e.tail_id
          WHERE (e.head_id = ? OR e.tail_id = ?)
          ORDER BY e.t_start_epoch ASC
        """
        cur.execute(sql, (eid, eid))
        rows = cur.fetchall()
        conn.close()
        
        if len(rows) < 2:
            return []
        
        gaps = []
        for i in range(len(rows) - 1):
            current_end = rows[i][8]  # t_end_epoch
            next_start = rows[i + 1][7]  # t_start_epoch
            gap_days = (next_start - current_end) / (24 * 3600)
            
            if gap_days >= min_gap_days:
                gaps.append({
                    "gap_days": gap_days,
                    "after_event": {
                        "head": rows[i][0], "relation": rows[i][1], "tail": rows[i][2],
                        "time": f"{rows[i][3]}~{rows[i][4]}"
                    },
                    "before_event": {
                        "head": rows[i+1][0], "relation": rows[i+1][1], "tail": rows[i+1][2],
                        "time": f"{rows[i+1][3]}~{rows[i+1][4]}"
                    }
                })
        
        if limit:
            gaps = gaps[:limit]
        
        return gaps

# ======================
# Toolkit main class
# ======================
class TemporalKGToolkit:
    """Time knowledge graph toolkit main class"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.query = TemporalKGQuery(db_path)
        self.advanced = AdvancedTemporalQuery(db_path)
    
    def get_toolkit_info(self) -> Dict[str, Any]:
        """Get toolkit information"""
        return {
            "db_path": self.db_path,
            "available_tools": [
                "retrieve_one_hop",
                "find_direct_connection",
                "events_on_day", 
                "events_in_month",
                "events_in_year",
                "find_entities_by_name_pattern",
                "get_entity_statistics",
                "find_temporal_sequence",
                "find_entities_after_time",
                "find_entities_before_time",
                "find_before_last",
                "find_after_first",
                "find_between_times",
                "find_temporal_neighbors",
                "find_chronological_sequence",
                "find_time_gaps"
            ],
            "time_operations": [
                "before", "after", "before_last", "after_first", 
                "between", "same_day", "same_month", "same_year"
            ],
            "time_formats": ["YYYY-MM-DD", "YYYY-MM", "YYYY", "YYYY-MM-DDTHH:MM:SSZ"]
        }
    
    def __getattr__(self, name):
        """Delegate to the corresponding query object"""
        if hasattr(self.query, name):
            return getattr(self.query, name)
        elif hasattr(self.advanced, name):
            return getattr(self.advanced, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

# ======================
# Convenient function
# ======================
def create_toolkit(db_path: str) -> TemporalKGToolkit:
    """Create toolkit instance"""
    return TemporalKGToolkit(db_path)

def quick_query(db_path: str, entity: Union[str, int], **kwargs) -> QueryResult:
    """Quick query function"""
    toolkit = create_toolkit(db_path)
    return toolkit.retrieve_one_hop(entity, **kwargs)

def quick_events_on_day(db_path: str, day: str, **kwargs) -> QueryResult:
    """Quick query events on a day"""
    toolkit = create_toolkit(db_path)
    return toolkit.events_on_day(day, **kwargs)

def quick_events_in_month(db_path: str, month: str, **kwargs) -> QueryResult:
    """Quick query events in a month"""
    toolkit = create_toolkit(db_path)
    return toolkit.events_in_month(month, **kwargs)
