"""
sql_fetcher.py — Maps question intent to SQL queries against CommandesDB.
Returns structured context dicts ready to be injected into the Mistral prompt.

Design principle: no ORM, no heavy abstraction. Plain SQL + dict results.
Each fetch_* function accepts an optional planning_id and returns a
human-readable text block (not raw SQL column names).

FIX: _cached_query() had a stray 'from db import query as _q' on the cache-miss
path.  'db' is not a resolvable top-level module — the correct import is
'from chatbot.db import …', which is already done at the top of this file.
The redundant local import is removed; the top-level 'query' is used directly.

FIX v3 (fetch_fragmentation): Removed HAVING COUNT(*) > 1 filter.
    BEFORE: The HAVING clause silently excluded any (Machine, Commande, Operation)
            group that had only 1 lot — e.g. CMD5 Poudre on Brongo 2 and CMD5
            Javellisation on Brongo 5. Mistral never saw these rows and either
            omitted CMD5 entirely or hallucinated lot counts for it by blending
            data from other commandes.
    AFTER:  All groups are returned regardless of NbLots. Single-lot entries
            (NbLots=1) are legitimate planning facts that Mistral must report
            accurately. The FRAGMENTATION prefix + system prompt rules already
            prevent Mistral from inventing counts; having complete data is what
            prevents omissions.
"""

from typing import Optional, List, Dict, Any
from chatbot.db import query, query_one, scalar
from chatbot.redis_cache import get_sql_cache, set_sql_cache


# ── Cache wrapper ─────────────────────────────────────────────────────────────

def _cached_query(
    planning_id: Optional[int],
    sql: str,
    params: tuple = (),
    cache_key_extra: str = "",
) -> List[Dict]:
    """
    Execute sql with params, caching the result in Redis.

    cache_key_extra: an optional string appended to the cache key to
    disambiguate queries that share the same SQL template but differ by
    a runtime value (e.g. machine name, commande number) already captured
    in params.  Must NOT be appended to sql — doing so corrupts the query
    string sent to SQL Server.

    FIX: previous call sites passed `sql + machine_name` or `sql + commande_num`
    as the sql argument.  SQL Server received a query ending with e.g.
    "ORDER BY ...\nBrongo 1" and raised error 42000 / syntax error near 'Brongo'.
    The correct pattern is:
        _cached_query(pid, sql, (pid, val), cache_key_extra=val)
    """
    cache_key = sql + str(params) + cache_key_extra
    cached = get_sql_cache(planning_id, cache_key)
    if cached is not None:
        return cached
    result = query(sql, params)
    set_sql_cache(planning_id, cache_key, result)
    return result


# ── Formatters ────────────────────────────────────────────────────────────────

def _fmt_minutes(minutes) -> str:
    """Convert minutes to human-readable string."""
    if minutes is None:
        return "N/A"
    m = int(minutes)
    if m < 60:
        return f"{m} min"
    h, rem = divmod(m, 60)
    return f"{h}h{rem:02d}" if rem else f"{h}h"


def _fmt_rows_as_text(rows: List[Dict], field_map: Dict[str, str]) -> str:
    """
    Convert DB rows to a human-readable block.
    field_map: {db_column: friendly_label}
    """
    lines = []
    for row in rows:
        parts = []
        for col, label in field_map.items():
            val = row.get(col)
            if val is not None:
                parts.append(f"{label}: {val}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


# ── Planning overview ─────────────────────────────────────────────────────────

def fetch_planning_summary(planning_id: int) -> str:
    sql = """
        SELECT DateGeneration, DateDebut, Statut,
               MakespanPM, MakespanDays, NombreCommandes, NombreLignes
        FROM Plannings WHERE Id = ?
    """
    row = _cached_query(planning_id, sql, (planning_id,))
    if not row:
        return f"Planning #{planning_id} introuvable."
    r = row[0]
    makespan_h = _fmt_minutes(r.get("MakespanPM"))
    return (
        f"Planning #{planning_id} — statut: {r.get('Statut')} | "
        f"généré le: {str(r.get('DateGeneration','?'))[:10]} | "
        f"début: {str(r.get('DateDebut','?'))[:10]} | "
        f"durée totale: {makespan_h} ({r.get('MakespanDays','?')} jours) | "
        f"commandes: {r.get('NombreCommandes','?')} | "
        f"lignes: {r.get('NombreLignes','?')}"
    )


# ── Machine load / utilisation ────────────────────────────────────────────────

def fetch_machine_load(planning_id: int) -> str:
    sql = """
        SELECT MachineName,
               SUM(DureeTotale) AS TotalMinutes,
               COUNT(DISTINCT NumeroCommande) AS NbCommandes,
               COUNT(*) AS NbLignes
        FROM PlanningRows
        WHERE PlanningId = ?
        GROUP BY MachineName
        ORDER BY TotalMinutes DESC
    """
    rows = _cached_query(planning_id, sql, (planning_id,))
    if not rows:
        return "Aucune donnée de charge machine."
    lines = ["Charge par machine:"]
    for r in rows:
        # FIX v2.6: label changed from "lot(s)" to "ligne(s) planning".
        # COUNT(*) grouped by MachineName counts scheduling rows, not production
        # lots. Labelling it "lots" caused Mistral to report e.g. "12 lots sur
        # Brongo 1" in résumé answers, which is factually wrong — those are
        # planning rows spanning multiple commandes and operations.
        lines.append(
            f"  {r['MachineName']}: {_fmt_minutes(r['TotalMinutes'])} planifiées | "
            f"{r['NbCommandes']} commande(s) | {r['NbLignes']} ligne(s) planning"
        )
    return "\n".join(lines)


# ── Commandes late / at risk ──────────────────────────────────────────────────

def fetch_late_orders(planning_id: int) -> str:
    sql = """
        SELECT NumeroCommande, DateExport, MAX(DateEnd) AS FinPlannifiee,
               Urgence, Quantite
        FROM PlanningRows
        WHERE PlanningId = ?
        GROUP BY NumeroCommande, DateExport, Urgence, Quantite
        HAVING MAX(DateEnd) > DateExport
        ORDER BY DATEDIFF(day, DateExport, MAX(DateEnd)) DESC
    """
    rows = _cached_query(planning_id, sql, (planning_id,))
    if not rows:
        return "Aucune commande en retard détectée."
    lines = ["Commandes en retard:"]
    for r in rows:
        fin = str(r.get('FinPlannifiee', '?'))[:10]
        exp = str(r.get('DateExport',    '?'))[:10]
        lines.append(
            f"  {r['NumeroCommande']}: fin prévue {fin} > export {exp} | "
            f"urgence {r['Urgence']} | quantité {r['Quantite']}"
        )
    return "\n".join(lines)


# ── Commandes overview ────────────────────────────────────────────────────────

def fetch_orders_for_planning(planning_id: int) -> str:
    sql = """
        SELECT DISTINCT pr.NumeroCommande, pr.Quantite, pr.Urgence,
                        pr.DateExport, r.NomRecette,
                        MIN(pr.DateStart) AS DebutReel,
                        MAX(pr.DateEnd)   AS FinReelle
        FROM PlanningRows pr
        LEFT JOIN Recettes r ON r.Id = pr.RecetteId
        WHERE pr.PlanningId = ?
        GROUP BY pr.NumeroCommande, pr.Quantite, pr.Urgence, pr.DateExport, r.NomRecette
        ORDER BY pr.Urgence, pr.DateExport
    """
    rows = _cached_query(planning_id, sql, (planning_id,))
    if not rows:
        return "Aucune commande dans ce planning."
    lines = ["Commandes planifiées:"]
    for r in rows:
        lines.append(
            f"  {r['NumeroCommande']}: recette={r.get('NomRecette','?')} | "
            f"qté={r['Quantite']} | urgence={r['Urgence']} | "
            f"export={str(r['DateExport'])[:10]} | "
            f"du {str(r.get('DebutReel','?'))[:10]} au {str(r.get('FinReelle','?'))[:10]}"
        )
    return "\n".join(lines)


# ── Operations sequence for a commande ───────────────────────────────────────

def fetch_operations_for_order(planning_id: int, commande_num: str) -> str:
    sql = """
        SELECT NomOperation, MachineName, LotIdx, NbLots, LotSize,
               DureeTotale, DateStart, DateEnd
        FROM PlanningRows
        WHERE PlanningId = ? AND NumeroCommande = ?
        ORDER BY NomOperation, LotIdx
    """
    rows = _cached_query(planning_id, sql, (planning_id, commande_num), cache_key_extra=commande_num)
    if not rows:
        return f"Aucune opération trouvée pour {commande_num}."
    lines = [f"Opérations de {commande_num}:"]
    for r in rows:
        lines.append(
            f"  {r['NomOperation']} lot {r['LotIdx']+1}/{r['NbLots']} "
            f"sur {r['MachineName']}: {_fmt_minutes(r['DureeTotale'])} | "
            f"du {str(r['DateStart'])[:10]} au {str(r['DateEnd'])[:10]}"
        )
    return "\n".join(lines)


# ── Fragmentation (lots) ──────────────────────────────────────────────────────

def fetch_fragmentation(planning_id: int) -> str:
    """
    Returns the number of lots per (Machine, Commande, Operation) group
    for the given planning. Every group is returned — including single-lot
    entries (NbLots=1).

    FIX v3: Removed HAVING COUNT(*) > 1.
        The old filter silently dropped any group with only 1 lot, e.g.:
          - CMD5  Poudre       on Brongo 2  (1 lot)
          - CMD5  Javellisation on Brongo 5 (1 lot)
        Mistral never received these rows and either omitted CMD5 entirely
        or invented lot counts by blending data from CMD1/CMD2. The fix
        ensures the SQL block is complete so Mistral can report accurately.

    FIX v2 (preserved): Removed TotalPieces and DureeMoyenneLot from output.
        Those columns caused Mistral to invent "2 lots de 200 pièces" by
        mixing LotSize values across Poudre (100 pièces/lot) and Javellisation
        (200 pièces/lot) operations. The only reliable, unambiguous fact is
        NbLots — that is all Mistral should report for fragmentation.

    Each output line is a single, unambiguous statement:
      [i/n] Machine=X Commande=Y Operation=Z NbLots=N
    The numbered prefix lets Mistral verify it has listed every row.
    """
    sql = """
        SELECT MachineName, NumeroCommande, NomOperation,
               COUNT(*) AS NbLots
        FROM PlanningRows
        WHERE PlanningId = ?
        GROUP BY MachineName, NumeroCommande, NomOperation
        ORDER BY MachineName, NumeroCommande, NomOperation
    """
    # FIX v3: HAVING COUNT(*) > 1 removed — all groups returned, including NbLots=1.
    rows = _cached_query(planning_id, sql, (planning_id,))
    if not rows:
        return "Pas de données de fragmentation pour ce planning."

    n = len(rows)
    lines = [
        f"FRAGMENTATION PLANNING #{planning_id} — EXACTEMENT {n} ligne(s) ci-dessous:",
        f"INSTRUCTION CRITIQUE: Ta réponse doit lister exactement {n} entrées — ni plus, ni moins.",
        "INSTRUCTION CRITIQUE: Copie NbLots tel quel depuis chaque ligne. Ne pas additionner, regrouper, ou inventer.",
        "",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"[{i}/{n}] Machine={r['MachineName']} "
            f"Commande={r['NumeroCommande']} "
            f"Operation={r['NomOperation']} "
            f"NbLots={r['NbLots']}"
        )
    return "\n".join(lines)


# ── Active alerts ─────────────────────────────────────────────────────────────

def fetch_alerts(planning_id: Optional[int] = None) -> str:
    if planning_id:
        sql = """
            SELECT a.Type, a.Severity, a.Message, a.NumeroCommande,
                   a.MachineName, a.GeneratedAt
            FROM Alerts a
            WHERE a.IsDismissed = 0
              AND (a.NumeroCommande IN (
                    SELECT DISTINCT NumeroCommande FROM PlanningRows WHERE PlanningId = ?
                  ) OR a.MachineId IN (
                    SELECT DISTINCT MachineId FROM PlanningRows WHERE PlanningId = ?
                  ))
            ORDER BY a.Severity DESC, a.GeneratedAt DESC
        """
        rows = _cached_query(planning_id, sql, (planning_id, planning_id))
    else:
        sql = """
            SELECT TOP 20 Type, Severity, Message, NumeroCommande,
                          MachineName, GeneratedAt
            FROM Alerts
            WHERE IsDismissed = 0
            ORDER BY Severity DESC, GeneratedAt DESC
        """
        rows = _cached_query(None, sql)
    if not rows:
        return "Aucune alerte active."
    lines = ["Alertes actives:"]
    for r in rows:
        lines.append(
            f"  [{r['Severity'].upper()}] {r['Type']}: {r['Message']}"
        )
    return "\n".join(lines)


# ── Machines overview ─────────────────────────────────────────────────────────

def fetch_machines() -> str:
    # Only return functional machines so Mistral never counts or names
    # non-functional ones (e.g. Brongo 4, Tupesa 2) in its answers.
    sql = """
        SELECT NomMachine, CapaciteMax, Operations
        FROM Machines
        WHERE Statut = 'Fonctionnel'
        ORDER BY NomMachine
    """
    rows = _cached_query(None, sql)
    if not rows:
        return "Aucune machine fonctionnelle trouvée."
    lines = [f"Machines fonctionnelles ({len(rows)}):"]
    for r in rows:
        lines.append(
            f"  {r['NomMachine']}: capacité={r['CapaciteMax']} | opérations={r['Operations']}"
        )
    return "\n".join(lines)


# ── Commandes actives (hors planning) ────────────────────────────────────────

def fetch_active_orders() -> str:
    sql = """
        SELECT NumeroCommande, Statut, DateExport, Urgence, Quantite,
               r.NomRecette
        FROM Commandes c
        LEFT JOIN Recettes r ON r.Id = c.RecetteId
        WHERE c.Statut NOT IN ('Terminé', 'Annulé')
        ORDER BY c.Urgence, c.DateExport
    """
    rows = _cached_query(None, sql)
    if not rows:
        return "Aucune commande active."
    lines = ["Commandes actives:"]
    for r in rows:
        lines.append(
            f"  {r['NumeroCommande']}: {r['Statut']} | recette={r.get('NomRecette','?')} | "
            f"qté={r['Quantite']} | urgence={r['Urgence']} | export={str(r['DateExport'])[:10]}"
        )
    return "\n".join(lines)


# ── Recette detail (operations sequence) ─────────────────────────────────────

def fetch_recette_for_commande(planning_id: int, commande_num: str) -> str:
    """
    Fetch the full recette (name + ordered operation sequence with all timing
    and lot parameters) for a specific commande in a planning.

    Joins: PlanningRows → Recettes → OperationsRecette
    Returns every operation in order with:
      - Ordre, NomOperation
      - DureeMinutes (cycle time per lot)
      - QuantiteLot (max pieces per lot → drives NbLots = ceil(Quantite/QuantiteLot))
      - TempsChargementMinutes, TempsDecharementMinutes
    This is the source of truth for "how long does operation X take" or
    "why is commande Y split into N lots".
    """
    sql = """
        SELECT DISTINCT r.NomRecette,
               op.Ordre, op.NomOperation,
               op.DureeMinutes, op.QuantiteLot,
               op.TempsChargementMinutes, op.TempsDecharementMinutes
        FROM PlanningRows pr
        JOIN Recettes r       ON r.Id  = pr.RecetteId
        JOIN OperationsRecette op ON op.RecetteId = r.Id
        WHERE pr.PlanningId = ? AND pr.NumeroCommande = ?
        ORDER BY op.Ordre
    """
    rows = _cached_query(planning_id, sql, (planning_id, commande_num), cache_key_extra=commande_num)
    if not rows:
        return f"Aucune recette trouvée pour {commande_num} dans le planning #{planning_id}."
    recette_name = rows[0].get("NomRecette", "?")
    lines = [
        f"Recette de {commande_num} : {recette_name}",
        f"Nombre d'opérations : {len(rows)}",
        "",
    ]
    for op in rows:
        charg  = op["TempsChargementMinutes"]
        decharg = op["TempsDecharementMinutes"]
        cycle  = op["DureeMinutes"]
        total_op = charg + cycle + decharg
        lines.append(
            f"  Étape {op['Ordre']} — {op['NomOperation']}: "
            f"chargement={charg}min | cycle={cycle}min | déchargement={decharg}min | "
            f"total/lot={total_op}min | taille_lot={op['QuantiteLot']} pièces"
        )
    return "\n".join(lines)


def fetch_all_recettes(planning_id: Optional[int] = None) -> str:
    """
    Fetch all recettes visible in a planning (or all recettes in DB if no
    planning_id).  Returns name + operation count + operation names in order.
    Used when the user asks a general "what recipes exist" / "what operations
    does recipe X have" question without a specific commande number.
    """
    if planning_id:
        sql = """
            SELECT DISTINCT r.Id, r.NomRecette,
                   op.Ordre, op.NomOperation,
                   op.DureeMinutes, op.QuantiteLot,
                   op.TempsChargementMinutes, op.TempsDecharementMinutes
            FROM PlanningRows pr
            JOIN Recettes r        ON r.Id  = pr.RecetteId
            JOIN OperationsRecette op ON op.RecetteId = r.Id
            WHERE pr.PlanningId = ?
            ORDER BY r.NomRecette, op.Ordre
        """
        rows = _cached_query(planning_id, sql, (planning_id,))
    else:
        sql = """
            SELECT r.Id, r.NomRecette,
                   op.Ordre, op.NomOperation,
                   op.DureeMinutes, op.QuantiteLot,
                   op.TempsChargementMinutes, op.TempsDecharementMinutes
            FROM Recettes r
            JOIN OperationsRecette op ON op.RecetteId = r.Id
            ORDER BY r.NomRecette, op.Ordre
        """
        rows = _cached_query(None, sql)

    if not rows:
        return "Aucune recette trouvée."

    # Group by recette
    from collections import defaultdict
    recettes: dict = defaultdict(list)
    for row in rows:
        recettes[row["NomRecette"]].append(row)

    lines = [f"Recettes disponibles ({len(recettes)}):"]
    for nom, ops in recettes.items():
        lines.append(f"\n  Recette : {nom} ({len(ops)} opération(s))")
        for op in ops:
            charg   = op["TempsChargementMinutes"]
            decharg = op["TempsDecharementMinutes"]
            cycle   = op["DureeMinutes"]
            total_op = charg + cycle + decharg
            lines.append(
                f"    Étape {op['Ordre']} — {op['NomOperation']}: "
                f"cycle={cycle}min | charg={charg}min | décharg={decharg}min | "
                f"total/lot={total_op}min | taille_lot={op['QuantiteLot']} pièces"
            )
    return "\n".join(lines)


# ── Valid transfer targets (pre-computed in Python, never by LLM) ─────────────

def fetch_valid_transfers(planning_id: int) -> str:
    """
    Pre-compute every valid lot-transfer opportunity for the amélioration prompt.

    A transfer (commande C, operation O, from machine A → to machine B) is valid
    if and only if machine B ALREADY handles (C, O) in this planning — i.e. it
    appears as a source machine for the same (commande, operation) pair.
    This is determined purely from PlanningRows: no schema knowledge, no LLM
    inference, no machine-capability table required.

    WHY THIS EXISTS:
    Mistral 7B cannot be trusted to verify machine-operation compatibility.
    Even with fragmentation rows in context it invents transfers like
    "réaffecter CMD2/Poudre vers Brongo 5" when Brongo 5 only handles
    Javellisation in this planning. Computing validity in Python and injecting
    a pre-screened list eliminates that class of hallucination entirely.

    ALGORITHM:
      1. Query all (MachineName, NumeroCommande, NomOperation, NbLots) groups
         from PlanningRows — same query as fetch_fragmentation.
      2. Build a dict: (commande, operation) → {machine: nb_lots}.
      3. Load machine total load (TotalMinutes) for ranking.
      4. For each (commande, operation) that appears on 2+ machines, every
         machine pair (source → target) where source has more lots than target
         is a valid transfer candidate. Rank candidates by load delta
         (most-loaded source → least-loaded target first).
      5. Format as unambiguous lines Mistral can copy verbatim.

    RETURNS:
      A text block with header + one line per valid transfer, e.g.:
        TRANSFERT VALIDE [1/3]: CMD1 / Poudre — de Brongo 1 (10 lots) → vers Brongo 2 (1 lot) | delta charge: 240 min
      Or a "no transfers possible" message if every (commande, operation) maps
      to exactly one machine (no parallelism exists in this planning).
    """
    # Step 1 — fragmentation groups
    frag_sql = """
        SELECT MachineName, NumeroCommande, NomOperation,
               COUNT(*) AS NbLots
        FROM PlanningRows
        WHERE PlanningId = ?
        GROUP BY MachineName, NumeroCommande, NomOperation
        ORDER BY MachineName, NumeroCommande, NomOperation
    """
    frag_rows = _cached_query(planning_id, frag_sql, (planning_id,))

    # Step 2 — machine total load
    load_sql = """
        SELECT MachineName, SUM(DureeTotale) AS TotalMinutes
        FROM PlanningRows
        WHERE PlanningId = ?
        GROUP BY MachineName
    """
    load_rows = _cached_query(planning_id, load_sql, (planning_id,))
    load_map: Dict[str, int] = {r["MachineName"]: int(r["TotalMinutes"] or 0) for r in load_rows}

    # Step 3 — build (commande, operation) → {machine: nb_lots}
    from collections import defaultdict
    groups: Dict[tuple, Dict[str, int]] = defaultdict(dict)
    for r in frag_rows:
        key = (r["NumeroCommande"], r["NomOperation"])
        groups[key][r["MachineName"]] = int(r["NbLots"])

    # Step 4 — find valid transfers
    transfers = []
    for (commande, operation), machine_lots in groups.items():
        if len(machine_lots) < 2:
            continue  # only one machine handles this pair — no transfer possible

        # Sort machines by load descending so we propose overloaded → underloaded
        machines_by_load = sorted(
            machine_lots.keys(),
            key=lambda m: load_map.get(m, 0),
            reverse=True,
        )
        for i, src in enumerate(machines_by_load):
            for tgt in machines_by_load[i + 1:]:
                src_lots   = machine_lots[src]
                tgt_lots   = machine_lots[tgt]
                src_load   = load_map.get(src, 0)
                tgt_load   = load_map.get(tgt, 0)
                delta      = src_load - tgt_load
                if delta <= 0:
                    continue  # source is not heavier — not a useful transfer
                transfers.append({
                    "commande":  commande,
                    "operation": operation,
                    "src":       src,
                    "tgt":       tgt,
                    "src_lots":  src_lots,
                    "tgt_lots":  tgt_lots,
                    "delta":     delta,
                })

    # Sort globally: largest load delta first
    transfers.sort(key=lambda t: t["delta"], reverse=True)

    if not transfers:
        return (
            "TRANSFERTS VALIDES: Aucun transfert possible — "
            "chaque (commande, opération) n'est traitée que par une seule machine dans ce planning."
        )

    n = len(transfers)
    lines = [
        f"TRANSFERTS VALIDES PRÉ-CALCULÉS — {n} transfert(s) possible(s):",
        "RÈGLE ABSOLUE: Tu ne peux suggérer QUE des transferts de cette liste.",
        "Ne jamais inventer une machine source ou cible absente de cette liste.",
        "",
    ]
    for i, t in enumerate(transfers, 1):
        lines.append(
            f"TRANSFERT VALIDE [{i}/{n}]: "
            f"{t['commande']} / {t['operation']} — "
            f"de {t['src']} ({t['src_lots']} lot(s)) "
            f"→ vers {t['tgt']} ({t['tgt_lots']} lot(s)) "
            f"| delta charge: {_fmt_minutes(t['delta'])}"
        )
    return "\n".join(lines)


# ── Operation sequencing / ordering verification ──────────────────────────────

def fetch_operation_sequence(planning_id: int) -> str:
    """
    For "Est-ce que Javellisation commence après Poudre ?" questions.

    Returns, per commande, the MIN(DateStart) and MAX(DateEnd) of every
    operation across all machines in this planning, ordered by operation
    start time. Python then computes the gap (in minutes) between each
    consecutive operation pair and explicitly labels each transition as:
      - SÉQUENTIEL (gap ≥ 0 min): next operation starts after previous ends
      - CHEVAUCHEMENT (gap < 0 min): next operation starts before previous ends

    WHY THIS EXISTS:
      Without this fetcher, "séquence/ordre des opérations" questions triggered
      the 'operation' intent with no commande number, falling through to
      fetch_orders_for_planning (order-level totals, no timestamps). Mistral
      had no sequencing data and hallucinated overlaps or strict ordering based
      on general LLM priors. The VERDICT prefix below lets Mistral copy the
      answer rather than reason about it.

    FORMAT (per commande):
      Commande CMD1:
        VERDICT Poudre → Javellisation: SÉQUENTIEL (gap=+0 min) | Poudre fin=2026-05-17 01h45 | Javellisation début=2026-05-17 02h45
        ...
    """
    sql = """
        SELECT NumeroCommande, NomOperation,
               MIN(DateStart) AS OpDebut,
               MAX(DateEnd)   AS OpFin
        FROM PlanningRows
        WHERE PlanningId = ?
        GROUP BY NumeroCommande, NomOperation
        ORDER BY NumeroCommande, MIN(DateStart)
    """
    rows = _cached_query(planning_id, sql, (planning_id,))
    if not rows:
        return "Aucune donnée de séquençage pour ce planning."

    # Group by commande
    from collections import defaultdict
    from datetime import datetime as _dt

    def _parse_dt(val):
        if val is None:
            return None
        s = str(val)[:16]  # "YYYY-MM-DD HH:MM"
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
            try:
                return _dt.strptime(s, fmt)
            except ValueError:
                pass
        return None

    commandes: dict = defaultdict(list)
    for r in rows:
        commandes[r["NumeroCommande"]].append(r)

    lines = [
        f"SÉQUENÇAGE DES OPÉRATIONS — Planning #{planning_id}:",
        "INSTRUCTION: Reporte les VERDICT exactement. Ne pas déduire d'ordre hors de ces données.",
        "",
    ]

    for cmd, ops in sorted(commandes.items()):
        lines.append(f"Commande {cmd}:")
        for i, op in enumerate(ops):
            debut_str = str(op.get("OpDebut", "?"))[:16].replace("T", " ")
            fin_str   = str(op.get("OpFin",   "?"))[:16].replace("T", " ")
            lines.append(
                f"  {op['NomOperation']}: début={debut_str} | fin={fin_str}"
            )
            if i > 0:
                prev = ops[i - 1]
                prev_fin = _parse_dt(prev.get("OpFin"))
                cur_debut = _parse_dt(op.get("OpDebut"))
                if prev_fin and cur_debut:
                    gap_min = int((cur_debut - prev_fin).total_seconds() / 60)
                    label = "SÉQUENTIEL" if gap_min >= 0 else "CHEVAUCHEMENT"
                    lines.append(
                        f"  VERDICT {prev['NomOperation']} → {op['NomOperation']}: "
                        f"{label} (gap={'+' if gap_min >= 0 else ''}{gap_min} min)"
                    )
        lines.append("")

    return "\n".join(lines).rstrip()


# ── Machine impact analysis (panne / breakdown hypotheticals) ─────────────────


def fetch_machine_impact(planning_id: int, machine_name: str) -> str:
    """
    For "Si <machine> tombe en panne, quelles commandes sont impactées ?" questions.

    Two queries, assembled into one unambiguous block:

    Query A — What is directly assigned to this machine in this planning?
      Returns (NumeroCommande, NomOperation, NbLots, TotalPieces) per
      (commande, operation) pair on the named machine.
      This is the ground truth: exactly what is lost if the machine goes down.

    Query B — For each (commande, operation) found in A, which OTHER machines
      in this planning also handle that same pair?
      These are the fallback machines — partial mitigation if the machine fails.
      If a (commande, operation) has NO other machine, it is fully blocked.

    WHY THIS EXISTS:
      Without this fetcher, "panne" questions only triggered the 'commande'
      intent, which called fetch_orders_for_planning. That gives order-level
      totals (quantity, date) but no machine assignment data. Mistral then had
      to infer which operations were on which machine from FAISS chunks —
      hallucinating lot counts, conflating Poudre and Javellisation lots, and
      incorrectly claiming "aucune autre machine" when alternatives existed.

    FORMAT:
      Each line is prefixed GROUPE: with an inline ALTERNATIVES: field so
      "Les [DONNÉES SQL] sont la source de vérité absolue" is applied correctly.
      Mistral is instructed via the panne_preamble to copy these lines verbatim.
    """
    # Query A — operations directly on the named machine
    impact_sql = """
        SELECT NumeroCommande, NomOperation,
               COUNT(*)        AS NbLots,
               SUM(LotSize)    AS TotalPieces,
               MIN(DateStart)  AS DebutPrevu,
               MAX(DateEnd)    AS FinPrevue,
               MAX(Urgence)    AS Urgence
        FROM PlanningRows
        WHERE PlanningId = ? AND MachineName = ?
        GROUP BY NumeroCommande, NomOperation
        ORDER BY MAX(Urgence), NumeroCommande, NomOperation
    """
    impact_rows = _cached_query(
        planning_id, impact_sql, (planning_id, machine_name), cache_key_extra=machine_name
    )

    if not impact_rows:
        return (
            f"IMPACT ANALYSE — {machine_name} (Planning #{planning_id}):\n"
            f"Aucune opération assignée à {machine_name} dans ce planning."
        )

    # Query B — alternative machines for each (commande, operation) pair.
    # FIX: SQL Server does not support row-value constructor syntax:
    #   (col1, col2) IN (SELECT col1, col2 FROM ...)
    # Replaced with a correlated EXISTS subquery — ANSI-compatible and
    # supported by all SQL Server versions including SQL Server 2012+.
    fallback_sql = """
        SELECT pr.NumeroCommande, pr.NomOperation, pr.MachineName,
               COUNT(*) AS NbLots
        FROM PlanningRows pr
        WHERE pr.PlanningId = ?
          AND pr.MachineName <> ?
          AND EXISTS (
              SELECT 1
              FROM PlanningRows sub
              WHERE sub.PlanningId    = ?
                AND sub.MachineName   = ?
                AND sub.NumeroCommande = pr.NumeroCommande
                AND sub.NomOperation  = pr.NomOperation
          )
        GROUP BY pr.NumeroCommande, pr.NomOperation, pr.MachineName
        ORDER BY pr.NumeroCommande, pr.NomOperation, pr.MachineName
    """
    fallback_rows = _cached_query(
        planning_id, fallback_sql,
        (planning_id, machine_name, planning_id, machine_name),
        cache_key_extra="fb_" + machine_name,
    )

    # Build (commande, operation) → [alternative machine strings]
    from collections import defaultdict
    fallbacks: Dict[tuple, list] = defaultdict(list)
    for r in fallback_rows:
        key = (r["NumeroCommande"], r["NomOperation"])
        fallbacks[key].append(f"{r['MachineName']} ({r['NbLots']} lot(s))")

    n = len(impact_rows)
    lines = [
        f"IMPACT ANALYSE — {machine_name} tombe en panne (Planning #{planning_id}):",
        f"EXACTEMENT {n} groupe(s) opération assigné(s) à {machine_name}:",
        "FORMAT: chaque ligne = GROUPE: ... | ALTERNATIVES: ...",
        "INSTRUCTION: Pour chaque ligne GROUPE, reporte EXACTEMENT NbLots et TotalPieces tels qu'écrits.",
        "INSTRUCTION: Pour chaque ligne GROUPE, reporte EXACTEMENT le champ ALTERNATIVES — ne pas l'ignorer.",
        "",
    ]

    for r in impact_rows:
        key = (r["NumeroCommande"], r["NomOperation"])
        debut = str(r.get("DebutPrevu", "?"))[:16].replace("T", " ")
        fin   = str(r.get("FinPrevue",  "?"))[:16].replace("T", " ")
        alts = fallbacks.get(key)
        alt_str = (" | ".join(alts)) if alts else "BLOQUÉ — aucune autre machine"
        # FIX: IMPACT and FALLBACK merged onto one line so Mistral cannot
        # lose the pairing between a group and its alternatives. When they
        # were on separate lines, Mistral read IMPACT: but skipped FALLBACK:,
        # causing it to fabricate "aucune alternative" for CMD1/ORD-A001
        # even when Brongo 2, 3, 5 alternatives were listed in FALLBACK:.
        lines.append(
            f"GROUPE: Commande={r['NumeroCommande']} "
            f"Operation={r['NomOperation']} "
            f"NbLots={r['NbLots']} "
            f"TotalPieces={r.get('TotalPieces', '?')} "
            f"Urgence={r.get('Urgence', '?')} "
            f"Du={debut} Au={fin} "
            f"| ALTERNATIVES: {alt_str}"
        )

    return "\n".join(lines)


# ── Generic SQL for free-form questions ──────────────────────────────────────

def fetch_generic(sql: str, planning_id: Optional[int] = None) -> List[Dict]:
    """Execute arbitrary read-only SQL, with cache."""
    return _cached_query(planning_id, sql)