"""
rag_engine.py — RAG engine v2.14 — SQL + FAISS semantic retrieval + Mistral.

Pipeline per question:
  1. Intent detection           → which SQL fetchers to call
  2. SQL fetch (sql_fetcher.py) → structured, human-readable context
  3. FAISS retrieve             → semantically similar DB chunks
  4. Prompt assembly            → system + Redis memory + SQL + FAISS + question
  5. Mistral call (streaming)   → answer

FIXES v2.14:
  - FIX (panne hallucinations): Strengthened panne_preamble with two targeted rules
      that directly counter the two observed failure modes:
        1. "Il y a EXACTEMENT N lignes IMPACT:" — Mistral now knows the exact count
           and must list every group. This stops silent omission of commandes (CMD2
           was being dropped because Mistral collapsed the enumeration prematurely).
        2. "INTERDIT : fusionner ou résumer plusieurs lignes IMPACT: d'une même
           commande en une seule phrase" — directly prohibits the "toutes ses
           opérations sont sur Brongo 1" fabrication pattern for CMD1.
      The n_impact count is computed from sql_context.count("\\nGROUPE:") so it
      reflects the actual data without any extra SQL query.
  - FIX (panne context pollution): All non-panne intent branches in
      _build_sql_context are now gated with 'panne not in intents'.
      'si Brongo 1 tombe en panne, quelles commandes...' matches both panne
      AND commande intents, causing fetch_orders_for_planning to inject
      recette/quantity data that Mistral answered from instead of from the
      GROUPE: lines. Planning summary is also skipped for panne (same reason).
      When panne is detected, only fetch_machine_impact is injected.

FIXES v2.13 (preserved):
  - FIX (amélioration truncation): num_predict 400 → 600.
      "comment réduire le makespan" matches both amélioration AND makespan intents.
      With num_predict=400 Mistral ran out of tokens mid-sentence when writing
      3-4 concrete transfer suggestions (each requires a full descriptive sentence).
      The reply was cut off and _is_complete() in chat_router.py rejected it from
      Redis memory, so the turn was lost. 600 tokens gives sufficient headroom
      for the preamble + 3-4 suggestions with one sentence each, while staying
      well below the threshold that caused the 300s timeout (which was driven by
      num_ctx=2048, not num_predict).

FIXES v2.12 (preserved):
  - FIX (300s timeout on amélioration): Three coordinated changes:

    1. num_ctx 2048 → 1536 for amélioration.
       fetch_valid_transfers replaced fetch_fragmentation; the new block is
       ~10 lines (one per valid transfer) vs ~20 raw fragmentation rows.
       2048 KV cache on CPU was the primary cause of the 300s timeout.

    2. num_predict 950 → 400 for amélioration.
       Pre-validated transfers mean Mistral only needs to copy 2-3 lines
       and write one sentence each — not reason through 20 fragmentation rows.
       950 was sized for the old fragmentation-reasoning task.
       NOTE: 400 proved too small when the answer contained 3-4 suggestions
       with full sentences — raised to 600 in v2.13.

    3. "amélioration" added to SQL_ONLY_INTENTS (faiss_top_k = 0).
       fetch_valid_transfers already contains all actionable data. FAISS chunks
       add ~300 chars of cross-planning noise and push KV cache use up.
       The v2.7 rationale ("FAISS general scheduling knowledge") no longer
       applies once valid transfers are pre-computed in Python.

FIXES v2.11 (preserved):
  - fetch_valid_transfers replaces fetch_fragmentation for amélioration.
  - amelioration_preamble simplified: instructs copy-from-list, not reasoning.
  - AMÉLIORATION system prompt rule updated to reference pre-validated list.

FIXES v2.10 (preserved):
  - accent-insensitive amélioration regex (amelior, reduire).
  - amelioration_preamble injected inside SQL data block for higher salience.

FIXES v2.9 and earlier preserved unchanged.
"""

import re
import time
import httpx
from typing import Optional, List

import chatbot.sql_fetcher as sf
from chatbot.rag.faiss_index import retrieve as faiss_retrieve

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "mistral"

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant expert en planification industrielle de production denim.
Tu as un accès complet à la base de données CommandesDB qui gère :
les commandes de production, les recettes de fabrication, les machines, les opérations, et les plannings générés.

Règles absolues :
- Réponds UNIQUEMENT à partir des données fournies (SQL + contexte sémantique).
- Les [DONNÉES SQL] sont la source de vérité absolue. En cas de contradiction avec le [CONTEXTE SÉMANTIQUE], les données SQL ont toujours priorité.
- Ne jamais inventer de machines, commandes, durées, nombres de lots ou opérations.
- Si une information est absente des données fournies, dis-le clairement en une phrase.
- LANGUE : Réponds TOUJOURS dans la langue de la [QUESTION] courante, peu importe la langue des échanges précédents. Si la question est en anglais, réponds en anglais. Si elle est en français, réponds en français.
- Réponses concises et directes, comme un chef d'atelier — pas de jargon SQL.
- Ne jamais recommander de modifier les durées de recette, les temps de chargement/déchargement, ou la capacité physique des machines (ce sont des constantes industrielles).
- FRAGMENTATION : Quand les données SQL contiennent des lignes préfixées par "FRAGMENTATION:", reporte EXACTEMENT le NbLots de chaque ligne séparément. Ne JAMAIS additionner, regrouper, ou mélanger les NbLots de lignes différentes. Chaque ligne "FRAGMENTATION: Machine=X Commande=Y Operation=Z NbLots=N" est un fait indépendant — liste-les tous sans exception.
- Tu peux proposer des améliorations de scheduling : réaffecter des lots, paralléliser, changer l'ordre des commandes, prioriser par urgence.
- PRIORITÉ URGENCE : urgence=1 est la plus haute priorité (à traiter en premier), urgence=2 est moins prioritaire. Ne jamais dire qu'une urgence=2 est plus prioritaire qu'une urgence=1.
- Toute suggestion d'amélioration doit être basée uniquement sur les machines et commandes présentes dans les données SQL fournies. Ne jamais affecter une opération à une machine qui ne la traite pas dans les données.
- RÉSUMÉ : Pour un résumé de planning, liste uniquement les faits littéralement présents dans les données SQL : statut, makespan, commandes (numéro, quantité, urgence, dates), charge par machine (temps planifié, nombre de commandes, nombre de lignes planning). Ne jamais déduire ni mentionner une "priorité par machine" — l'urgence est une propriété de la commande, pas de la machine. Ne jamais introduire de détail d'opération (ex. Javellisation, Poudre) sauf s'il figure explicitement dans les données fournies.
- AMÉLIORATION : Les données SQL contiennent une liste "TRANSFERTS VALIDES PRÉ-CALCULÉS". Chaque entrée garantit que la machine cible traite déjà cette (commande, opération) dans ce planning — validé en Python, pas à vérifier. Tu dois UNIQUEMENT suggérer des transferts présents dans cette liste. Pour chaque suggestion : copie exactement la commande, l'opération, la machine source, la machine cible et le NbLots depuis la ligne TRANSFERT VALIDE. Priorise les transferts avec le delta de charge le plus élevé (meilleur gain de makespan). INTERDIT : inventer une machine source ou cible, proposer un transfert absent de la liste, donner des conseils génériques ("réaffecter sur une machine plus rapide", "ajouter des ressources", "paralléliser si possible").
- DÉFINITION : Si la question utilise "c'est quoi", "qu'est-ce que", "what is", "définition", "explique", "signifie" ou "veut dire" à propos d'un terme technique (makespan, recette, urgence, lot, fragmentation, etc.), commence TOUJOURS par une définition claire en une phrase, puis complète avec la valeur SQL si disponible. Ne jamais répondre uniquement par une valeur chiffrée à une question de définition. Ne jamais inventer d'exemples chiffrés (quantités, nombres de lots, durées, noms de machines) pour illustrer une définition — si aucun exemple ne figure explicitement dans les données SQL fournies, donne uniquement la définition conceptuelle sans aucun chiffre inventé.
- PANNE / IMPACT MACHINE : Les données SQL contiennent des lignes préfixées GROUPE:. Chaque ligne GROUPE: contient sur une seule ligne : Commande, Operation, NbLots, TotalPieces, Urgence, et un champ ALTERNATIVES: listant les autres machines disponibles (ou 'BLOQUÉ' si aucune). Reporte EXACTEMENT NbLots et TotalPieces depuis chaque ligne GROUPE — ne pas additionner, regrouper, ou inventer. Pour chaque ligne GROUPE:, lis le champ ALTERNATIVES: sur la même ligne — ne jamais l'ignorer. Ne jamais affirmer qu'une opération est bloquée si son champ ALTERNATIVES: liste des machines. Ne jamais fusionner plusieurs lignes GROUPE: d'une même commande en une seule affirmation. Chaque ligne GROUPE: est un fait indépendant à lister séparément.
- SÉQUENCE / ORDRE DES OPÉRATIONS : Les données SQL contiennent des lignes préfixées VERDICT. Chaque VERDICT indique si la transition entre deux opérations est SÉQUENTIEL (gap ≥ 0 min) ou CHEVAUCHEMENT (gap < 0 min), calculé en Python depuis les timestamps réels. Reporte ces VERDICT exactement — ne jamais déduire l'ordre à partir de la logique industrielle générale ou d'exemples hors du planning fourni. Si toutes les lignes VERDICT d'une commande disent SÉQUENTIEL, la réponse est "oui, toujours séquentiel pour cette commande". Ne jamais inventer un chevauchement absent des données.
"""

# ── Question routing ──────────────────────────────────────────────────────────

_PATTERNS = {
    "retard":        r"retard|en retard|late|delay|dépasse|dépassé|export",
    "charge":        r"charge|utilisa|load|surchargé|sous.utilis|bottleneck|goulot",
    # "lot" removed from fragmentation — it ambiguously matches per-lot detail
    # questions (e.g. "montre les lots de CMD1") which should route to commande.
    "fragmentation": r"fragment|split|morcel|découp|fragmentation des lots",
    # "lot" removed from commande — "c'est quoi un lot" matched here and caused
    # fetch_orders_for_planning (verbose, slow) to fire for a definition query.
    # Definition questions about "lot" are caught by the définition intent.
    "commande":      r"commande|order|CMD|numéro",
    "operation":     r"opération|operation|étape|step",
    "machine":       r"machine|équipement|appareil",
    "makespan":      r"makespan|durée totale|total duration|combien de temps|temps total",
    "alerte":        r"alerte|alert|warning|problème|probleme",
    "résumé":        r"résumé|résume|summary|overview|bilan|rapport",
    # FIX v2.10: added unaccented variants "amelior" and "ameliorer".
    # "amélior" (with accent) was matching fine, but "ameliorer" or "comment
    # ameliorer" (typed without accent) silently missed the intent, causing
    # Mistral to fall back to fetch_planning_summary only and give generic advice.
    # Also added "reduire" (no accent) as a standalone term in case the user
    # types it without the makespan context.
    "amélioration":  r"amélior|amelior|optim|improve|better|r[eé]duire.*makespan|makespan.*r[eé]duire|comment.*planif|make.*optimal|how.*improv|comment.*am[eé]lior|comment.*r[eé]duire|comment.*reduire",
    # panne intent: machine breakdown / failure impact hypotheticals.
    # Must be detected BEFORE 'machine' and 'commande' so the dedicated
    # fetch_machine_impact fetcher fires instead of fetch_machine_load or
    # fetch_orders_for_planning (which have no machine-assignment data).
    "panne":         r"panne|tombe.*en panne|en panne|breakdown|fail|hors.service|arrêt.*machine|machine.*arrêt|si.*tombe|impact.*machine|machine.*impact|affecté.*machine",
    # séquence intent: temporal ordering of operations within a commande.
    # Covers "après", "avant", "ordre", "toujours", "d'abord", "ensuite".
    # Must NOT rely on FAISS (cross-planning chunks can contain contradictory
    # ordering patterns). Fetches MIN/MAX timestamps from PlanningRows directly.
    "séquence":      r"apr[eè]s|avant|toujours.*apr[eè]s|apr[eè]s.*toujours|ordre.*op[eé]ration|op[eé]ration.*ordre|d'abord|ensuite|avant.*de|commence.*apr[eè]s|fin.*avant|séquence|sequen|chevauchement|overlap|en m[eê]me temps|simultan",
    # définition intent: user is asking what a term means, not requesting data.
    "définition":    r"c'est quoi|qu'est.ce que|what is|what's|définition|expliqu|signif|veut dire|mean",
    # recette intent: questions about recipe steps, operation durations, lot sizes
    "recette":       r"recette|recipe|opération.*durée|durée.*opération|taille.*lot|lot.*taille|QuantiteLot|étape.*recette|combien.*lot|temps.*charg|chargement|déchargement",
}


def _detect_intent(question: str) -> set:
    q = question.lower()
    intents = set()
    for intent, pattern in _PATTERNS.items():
        if re.search(pattern, q):
            intents.add(intent)
    return intents or {"résumé"}


def _extract_commande_num(question: str) -> Optional[str]:
    """
    Extract a commande identifier like CMD1, CMD2, ORD-A001 from the question.

    FIX: The original pattern r'\b([A-Z]{2,}[\-0-9]{2,20})\b' was silently
    broken — it required 2+ digits/dashes after the letters, so 'CMD1' (only
    1 digit) never matched. This caused the 'operation'/'commande' intent branch
    to always fall through to fetch_orders_for_planning instead of the more
    precise fetch_operations_for_order.

    New pattern: 2+ uppercase letters followed by at least one digit (anywhere
    in the suffix, possibly mixed with letters and dashes). Excludes common
    French/English uppercase acronyms that are not commande IDs.
    """
    _GENERIC = {
        "LES", "DES", "EST", "PAR", "SUR", "POUR", "QUE", "QUI", "OU", "ET",
        "UNE", "DU", "AU", "EN", "SE", "LA", "LE", "UN", "DE", "CE", "IL",
        "SQL", "RAG", "API", "OK",
    }
    # Must start with 2+ letters, contain at least one digit in the suffix
    pattern = r'\b([A-Z]{2,}[\-A-Z0-9]*[0-9][A-Z0-9\-]*)\b'
    candidates = [m for m in re.findall(pattern, question.upper()) if m not in _GENERIC]
    return candidates[0] if candidates else None


def _extract_machine_name(question: str) -> Optional[str]:
    """
    Extract a machine name like 'Brongo 1', 'Brongo 3', 'Tupesa 2' from the question.
    Returns the canonical 'Name N' form, or None if not found.
    """
    match = re.search(r'\b(Brongo|Tupesa)\s*(\d+)\b', question, re.IGNORECASE)
    if match:
        return f"{match.group(1).capitalize()} {match.group(2)}"
    return None


# ── SQL context builder ───────────────────────────────────────────────────────

def _build_sql_context(question: str, planning_id: Optional[int]) -> str:
    intents = _detect_intent(question)
    blocks = []

    # FIX v2.9: skip planning summary for pure definition questions.
    # fetch_planning_summary gives Mistral numeric context that it uses to
    # invent plausible-sounding examples when answering definitions.
    # FIX v2.14: also skip for panne intent — the summary block (makespan,
    # commande count, order quantities) gives Mistral easier-to-read numeric
    # data that it answers from instead of from the GROUPE: impact lines.
    # fetch_machine_impact already contains everything needed for panne answers.
    if planning_id and "définition" not in intents and "panne" not in intents:
        blocks.append(sf.fetch_planning_summary(planning_id))

    if "résumé" in intents and "panne" not in intents:
        if planning_id:
            blocks.append(sf.fetch_orders_for_planning(planning_id))
            blocks.append(sf.fetch_machine_load(planning_id))

    if "panne" in intents:
        if planning_id:
            machine_name = _extract_machine_name(question)
            if machine_name:
                # fetch_machine_impact is the sole data source for panne questions.
                # It provides exact (commande, operation, NbLots, TotalPieces) rows
                # assigned to the named machine, plus FALLBACK machines per pair.
                #
                # FIX v2.14: fetch_machine_load removed from this branch.
                # When both blocks were injected, Mistral consistently read the
                # load summary ('3 lignes de planning', '12h planifiées') and
                # answered from that instead of from the IMPACT:/FALLBACK: lines.
                # The load block is noise for a panne question — the IMPACT block
                # already contains NbLots, TotalPieces, urgence, and alternatives.
                blocks.append(sf.fetch_machine_impact(planning_id, machine_name))
            else:
                # Machine name not parseable — fall back to full load overview
                blocks.append(sf.fetch_machine_load(planning_id))

    if "amélioration" in intents:
        if planning_id:
            # fetch_machine_load: current load per machine (which is overloaded,
            #   which is underused).
            # fetch_late_orders: which commandes to prioritise first.
            # fetch_valid_transfers: pre-computed in Python — every valid
            #   (commande, operation, source → target) transfer that exists in
            #   this planning. Replaces fetch_fragmentation so Mistral NEVER
            #   has to infer machine-operation compatibility itself.
            blocks.append(sf.fetch_machine_load(planning_id))
            blocks.append(sf.fetch_late_orders(planning_id))
            blocks.append(sf.fetch_valid_transfers(planning_id))

    if "retard" in intents and "panne" not in intents:
        if planning_id:
            blocks.append(sf.fetch_late_orders(planning_id))
        else:
            blocks.append(sf.fetch_active_orders())

    if ("charge" in intents or "machine" in intents) and "panne" not in intents:
        if planning_id:
            blocks.append(sf.fetch_machine_load(planning_id))
        else:
            blocks.append(sf.fetch_machines())

    if "fragmentation" in intents:
        if planning_id:
            blocks.append(sf.fetch_fragmentation(planning_id))

    if "séquence" in intents:
        if planning_id:
            # fetch_operation_sequence returns MIN(DateStart)/MAX(DateEnd) per
            # (commande, operation) with Python-computed VERDICT lines.
            # Mistral copies the VERDICT rather than reasoning about timestamps.
            blocks.append(sf.fetch_operation_sequence(planning_id))

    if ("commande" in intents or "operation" in intents) and "panne" not in intents:
        if planning_id and "séquence" not in intents:
            # Skip when séquence is present: fetch_operation_sequence already
            # provides per-commande operation data with timestamps. Adding
            # fetch_orders_for_planning would inject order-level totals with no
            # timestamps — pure noise for a sequencing question.
            cmd_num = _extract_commande_num(question)
            if cmd_num:
                blocks.append(sf.fetch_operations_for_order(planning_id, cmd_num))
            else:
                blocks.append(sf.fetch_orders_for_planning(planning_id))

    if "recette" in intents:
        cmd_num = _extract_commande_num(question)
        if planning_id and cmd_num:
            blocks.append(sf.fetch_recette_for_commande(planning_id, cmd_num))
        else:
            blocks.append(sf.fetch_all_recettes(planning_id))

    if "makespan" in intents:
        if planning_id:
            blocks.append(sf.fetch_planning_summary(planning_id))

    if "alerte" in intents and "panne" not in intents:
        blocks.append(sf.fetch_alerts(planning_id))

    if not planning_id and not blocks:
        blocks.append(sf.fetch_active_orders())
        blocks.append(sf.fetch_machines())
        blocks.append(sf.fetch_alerts())

    seen, unique = set(), []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            unique.append(b)

    return "\n\n".join(unique)


# ── num_ctx selection ─────────────────────────────────────────────────────────

def _num_ctx(intents: set) -> int:
    """
    Choose context window size based on expected prompt size.

    - amélioration:
        1536 — fetch_valid_transfers replaced fetch_fragmentation; the new
        block is far more compact (one line per valid transfer). 2048 caused
        consistent 300s CPU timeouts. 1536 fits the full payload with headroom.
    - commande / operation / fragmentation / recette:
        2048 — larger SQL payloads still need the full window.
    - résumé:
        1024 — fetch_orders + fetch_machine_load ~200 tokens. 2048 caused
        ~300s inference on CPU for what is a small payload.
    - définition (without heavy data intents):
        1024 — definition answers are short; 2048 caused timeouts for simple
        definition queries like "c'est quoi un lot".
    - alerte / retard / charge / machine / makespan:
        Small, focused SQL block. 1536 is plenty and maximises speed on CPU.
    - default: 1536.

    NOTE: 4096 was the original default but caused consistent timeouts on CPU
    inference (no GPU). Lowering num_ctx is the single biggest speed win for
    local Mistral — it directly controls KV cache allocation.
    """
    if intents & {"définition"} and not intents & {"amélioration", "fragmentation"}:
        return 1024
    if intents & {"panne"}:
        # fetch_machine_impact: IMPACT:/FALLBACK: lines only — compact block.
        # 1536 is sufficient; fetch_machine_load no longer injected (v2.14).
        return 1536
    if intents & {"séquence"}:
        # fetch_operation_sequence: one row per (commande, operation) + VERDICT lines.
        # 4 commandes × ~3 operations = ~12 rows. 1024 is ample.
        return 1024
    if intents & {"amélioration"}:
        # FIX v2.12: lowered 2048 → 1536.
        # fetch_valid_transfers replaced fetch_fragmentation. The new block is
        # ~10 lines (one per valid transfer) vs ~20 raw fragmentation rows.
        # 2048 KV cache on CPU caused consistent 300s timeouts. 1536 fits
        # machine_load + late_orders + valid_transfers comfortably.
        return 1536
    if intents & {"commande", "operation"}:
        return 2048
    if intents & {"résumé"}:
        return 1024
    if intents & {"fragmentation", "recette"}:
        return 2048
    # alerte, retard, charge, machine, makespan → small context
    return 1536


def _temperature(intents: set) -> float:
    """Lower temperature for factual lookups; slightly higher for suggestions."""
    if intents & {"amélioration"}:
        return 0.4
    return 0.2


# ── Mistral streaming call ────────────────────────────────────────────────────

async def _call_mistral(
    messages: list,
    num_predict: int = 300,
    num_ctx: int = 2048,
    temperature: float = 0.2,
) -> str:
    import json as _json
    payload = {
        "model":      OLLAMA_MODEL,
        "messages":   messages,
        "stream":     True,
        "keep_alive": "10m",
        "options": {
            "num_predict": num_predict,
            "num_ctx":     num_ctx,
            "temperature": temperature,
            "top_p":       0.9,
            # num_thread saturates all available CPU cores for faster inference.
            # Adjust to match your server's actual physical core count.
            "num_thread":  8,
        },
    }
    tokens = []
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = _json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        tokens.append(content)
                    if chunk.get("done"):
                        break
                except Exception:
                    pass
    return "".join(tokens).strip()


def _token_budget(intents: set) -> int:
    """
    Choose num_predict (max output tokens) based on expected answer length.
    """
    if intents & {"définition"} and not intents & {"amélioration", "fragmentation"}:
        return 150
    if intents & {"panne"}:
        # Impact analysis: list affected (commande, operation) groups + fallback
        # machines + a risk summary. 400 tokens is enough for ≤10 impact rows.
        return 400
    if intents & {"séquence"}:
        # VERDICT lines are short; Mistral only needs to confirm/summarise them.
        # 250 tokens is enough for a 4-commande planning with 2 operations each.
        return 250
    if intents & {"amélioration"}:
        # FIX v2.13: raised 400 → 600.
        # With num_predict=400, Mistral ran out of tokens mid-sentence when
        # writing 3-4 concrete transfer suggestions (each requires a descriptive
        # sentence). The reply was rejected by _is_complete() in chat_router.py
        # and not saved to Redis memory. "comment réduire le makespan" matches
        # both amélioration AND makespan — the combined answer (preamble + top
        # transfers by delta + one sentence per suggestion) consistently needs
        # 450-550 tokens. 600 provides headroom without approaching the timeout
        # threshold (which was caused by num_ctx=2048, not num_predict).
        return 600
    if intents & {"résumé"}:
        # 200 tokens: a résumé of 4 commandes + 4 machines fits comfortably.
        # 600 (old value) forced Mistral to pad its answer and contributed to
        # ~311s inference time on CPU.
        return 200
    if intents & {"fragmentation"}:
        # After removing HAVING COUNT(*) > 1, all groups are returned including
        # single-lot rows. Planning #117 produces ~18 rows; at ~15 tokens/row
        # that is ~270 tokens of raw data. 900 gives Mistral enough headroom
        # to list every row without truncation.
        return 900
    if intents & {"recette"}:
        # Recette detail includes every operation step with 5 timing fields each.
        # 500 tokens covers a recette with ~10 operations comfortably.
        return 500
    if intents & {"alerte", "commande", "operation"}:
        return 400
    if intents & {"charge", "retard", "machine"}:
        return 300
    return 200


# ── Main analyze function ─────────────────────────────────────────────────────

def _build_amelioration_answer(planning_id: int, question: str) -> Optional[str]:
    """
    FIX v2.15 — Hallucination-proof amélioration answers.

    ROOT CAUSE of all previous hallucinations:
        Mistral 7B cannot reliably "copy exactly" from a list. Even with
        INTERDIT instructions and inline preambles, it paraphrases, invents
        delta values, swaps machine names, and fabricates load numbers.
        The prompt-engineering approach (v2.10–v2.14) has hit a hard ceiling.

    SOLUTION — remove Mistral from the factual parts entirely:
        Python reads the pre-validated transfer list from fetch_valid_transfers,
        picks the top MAX_SUGGESTIONS by delta, and formats the bullet points
        verbatim. Mistral is only asked to write a short intro sentence
        (≤ 40 words, no numbers, no machine names) — the one task it does well.

    If planning_id is absent or fetch_valid_transfers returns no transfers,
    returns None so the caller falls through to the normal Mistral pipeline.
    """
    MAX_SUGGESTIONS = 3

    raw = sf.fetch_valid_transfers(planning_id)
    if not raw or "Aucun transfert possible" in raw:
        return None

    # Parse lines: "TRANSFERT VALIDE [i/n]: CMD / Op — de Src (X lot(s)) → vers Tgt (Y lot(s)) | delta charge: Z"
    import re as _re
    pattern = _re.compile(
        r"TRANSFERT VALIDE \[(\d+)/(\d+)\]: "
        r"(.+?) / (.+?) — "
        r"de (.+?) \((\d+) lot\(s\)\) → vers (.+?) \((\d+) lot\(s\)\) "
        r"\| delta charge: (.+)"
    )
    transfers = []
    for line in raw.splitlines():
        m = pattern.match(line.strip())
        if m:
            transfers.append({
                "idx":       int(m.group(1)),
                "total":     int(m.group(2)),
                "commande":  m.group(3),
                "operation": m.group(4),
                "src":       m.group(5),
                "src_lots":  int(m.group(6)),
                "tgt":       m.group(7),
                "tgt_lots":  int(m.group(8)),
                "delta":     m.group(9),
            })

    if not transfers:
        return None

    top = transfers[:MAX_SUGGESTIONS]
    total = transfers[0]["total"]

    # Build the factual bullet lines in Python — Mistral never touches these
    bullets = []
    for t in top:
        bullets.append(
            f"• {t['commande']} / {t['operation']} : "
            f"déplacer {t['src_lots']} lot(s) de **{t['src']}** → **{t['tgt']}** "
            f"(gain de charge : {t['delta']})"
        )

    bullet_block = "\n".join(bullets)
    remaining = total - len(top)
    footer = (
        f"\n\n_{remaining} autre(s) transfert(s) valide(s) disponible(s) — "
        f"demandez des détails sur une commande spécifique._"
        if remaining > 0 else ""
    )

    # Ask Mistral ONLY for a short intro — no numbers, no machine names
    intro_prompt = (
        f"En une seule phrase courte (max 30 mots, sans chiffres, sans noms de machines), "
        f"explique pourquoi rééquilibrer la charge entre machines améliore un planning de production denim."
    )
    return (intro_prompt, bullet_block, footer, top)


async def analyze(
    question: str,
    planning_id: Optional[int],
    memory: list,
) -> str:
    """
    Full RAG pipeline:
      1. Intent detection
      2. SQL context  — structured, intent-routed
      3. For amélioration: Python assembles the factual answer, Mistral writes only intro (v2.15)
      4. For all other intents: FAISS + Mistral full pipeline
    """
    intents = _detect_intent(question)
    print(f"[RAG] Intents detected: {intents}")

    # ── FIX v2.15: amélioration short-circuit ────────────────────────────────
    # For amélioration questions with a known planning, Python assembles the
    # entire factual answer. Mistral only generates a short intro sentence
    # (≤ 30 words, no numbers, no machine names). This eliminates hallucination
    # on transfer details — Mistral was consistently paraphrasing / inventing
    # values despite "copie exactement" and INTERDIT instructions (v2.10–v2.14).
    if "amélioration" in intents and planning_id:
        t0 = time.time()
        amelio_result = _build_amelioration_answer(planning_id, question)
        print(f"[TIMING] SQL fetch (amélioration): {time.time() - t0:.2f}s")

        if amelio_result is not None:
            intro_prompt, bullet_block, footer, top = amelio_result

            # Ask Mistral for a single intro sentence only — no data in the prompt
            intro_messages = [
                {"role": "system", "content": (
                    "Tu es un assistant expert en planification industrielle. "
                    "Réponds UNIQUEMENT avec une seule phrase courte (max 30 mots). "
                    "Sans chiffres. Sans noms de machines. Sans listes."
                )},
                {"role": "user", "content": intro_prompt},
            ]
            t2 = time.time()
            try:
                intro = await _call_mistral(
                    intro_messages,
                    num_predict=60,
                    num_ctx=512,
                    temperature=0.3,
                )
                print(f"[TIMING] Mistral intro: {time.time() - t2:.2f}s ({len(intro)} chars)")
            except Exception:
                intro = "Voici les transferts de lots recommandés pour améliorer ce planning :"

            # Compose final answer entirely in Python
            reply = f"{intro}\n\n{bullet_block}{footer}"
            print(f"[RAG] amélioration answer built in Python ({len(reply)} chars, Mistral intro only)")
            return reply

    # Layer 1 — SQL (structured)
    t0 = time.time()
    sql_context = _build_sql_context(question, planning_id)
    print(f"[TIMING] SQL fetch: {time.time() - t0:.2f}s ({len(sql_context)} chars)")

    # Layer 2 — FAISS (semantic)
    # Intents that are purely numerical/structural must come from SQL only.
    # FAISS chunks are built from historical planning summaries and can contain
    # lot counts, machine assignments, and durations from *other* plannings.
    # Setting top_k=0 skips FAISS entirely for these intents.
    #
    # FIX v2.7: amélioration checked BEFORE SQL_ONLY_INTENTS.
    # "comment réduire le makespan" matches both amélioration AND makespan.
    # The old order set faiss_top_k=0 (makespan wins) which blocked FAISS for
    # amélioration questions. amélioration benefits from FAISS general scheduling
    # knowledge, so it must take precedence even when makespan is also detected.
    # FIX v2.12: "amélioration" added to SQL_ONLY_INTENTS.
    # fetch_valid_transfers already contains all actionable data pre-validated
    # in Python. FAISS chunks add cross-planning noise (~300 chars) and push
    # KV cache use up, contributing to the 300s timeout. There is no scheduling
    # knowledge in FAISS that isn't already captured by the transfer list.
    SQL_ONLY_INTENTS = {"fragmentation", "retard", "makespan", "recette", "résumé", "amélioration", "panne", "séquence"}

    if intents & SQL_ONLY_INTENTS:
        faiss_top_k = 0   # purely numerical: SQL is authoritative, FAISS pollutes
    elif intents & {"charge", "machine"}:
        faiss_top_k = 2   # numeric data, FAISS is supplementary only
    else:
        faiss_top_k = 4   # alerte, commande, operation, general — FAISS helps

    t1 = time.time()
    faiss_chunks: List[str] = await faiss_retrieve(
        question=question,
        planning_id=planning_id,
        top_k=faiss_top_k,
    )
    print(f"[TIMING] FAISS retrieve: {time.time() - t1:.2f}s ({len(faiss_chunks)} chunks)")

    faiss_context = (
        "\n".join(f"- {c}" for c in faiss_chunks)
        if faiss_chunks else ""
    )

    # Layer 3 — Prompt assembly
    # SQL labelled as ground truth; FAISS as indicative only.
    mode_label = f"Planning #{planning_id}" if planning_id else "mode général"

    # FIX v2.10: For amélioration intent, inject a hard grounding instruction
    # directly inside the SQL data block, immediately before the fragmentation
    # rows. System-prompt-only instructions are lower-salience for Mistral 7B
    # on CPU — placing the instruction adjacent to the data forces the model to
    # enumerate FRAGMENTATION lines first and anchor every suggestion to a
    # specific (commande, opération, machine source, machine cible) tuple drawn
    # from those rows. This reliably suppresses generic advice like "réaffecter
    # sur une machine plus rapide" in favour of concrete, data-backed suggestions.
    if "amélioration" in intents:
        amelioration_preamble = (
            "INSTRUCTION OBLIGATOIRE — AMÉLIORATION DU PLANNING:\n"
            "Les transferts ci-dessous ont été pré-validés en Python. "
            "Chaque ligne TRANSFERT VALIDE garantit que la machine cible "
            "traite déjà cette (commande, opération) dans ce planning.\n"
            "Étape 1 — Identifie les 2-3 transferts avec le plus grand delta de charge "
            "(delta charge le plus élevé = meilleur gain de makespan).\n"
            "Étape 2 — Pour chaque suggestion, copie EXACTEMENT : "
            "commande, opération, machine source, machine cible, NbLots depuis la ligne TRANSFERT VALIDE. "
            "Ne paraphrase pas, ne recalcule pas, ne remplace pas par d'autres machines.\n"
            "INTERDIT : tout transfert non présent dans la liste TRANSFERTS VALIDES ci-dessous.\n"
            "INTERDIT : conseils génériques (\"réaffecter\", \"paralléliser\", \"ajouter des ressources\").\n"
        )
        sql_block = (
            f"[DONNÉES SQL — SOURCE DE VÉRITÉ — {mode_label}]\n"
            f"{amelioration_preamble}\n"
            f"{sql_context}"
        )
    elif "panne" in intents:
        # Count IMPACT: lines so Mistral knows exactly how many groups to enumerate.
        n_impact = sql_context.count("\nGROUPE:")
        panne_preamble = (
            "INSTRUCTION OBLIGATOIRE — ANALYSE D'IMPACT MACHINE:\n"
            "Les données ci-dessous proviennent de PlanningRows — elles sont exhaustives et exactes.\n"
            f"Il y a EXACTEMENT {n_impact} ligne(s) GROUPE: ci-dessous. "
            f"Ta réponse doit lister EXACTEMENT {n_impact} groupe(s) — ni plus, ni moins.\n"
            "Chaque ligne GROUPE: contient sur une seule ligne : Commande, Operation, NbLots, TotalPieces, Urgence, "
            "et le champ ALTERNATIVES: qui liste les autres machines disponibles (ou BLOQUÉ si aucune).\n"
            "Étape 1 — Pour chaque ligne GROUPE:, reporte EXACTEMENT Commande, Operation, NbLots et TotalPieces.\n"
            "Étape 2 — Pour chaque ligne GROUPE:, lis le champ ALTERNATIVES: sur la même ligne. "
            "Si ALTERNATIVES contient des machines, indique ces machines comme alternatives. "
            "Si ALTERNATIVES contient 'BLOQUÉ', indique 'aucune alternative — opération bloquée'.\n"
            "Étape 3 — Conclus avec les commandes les plus impactées (urgence la plus haute, aucune alternative).\n"
            "INTERDIT : inventer des lots, des pièces, des machines, ou des alternatives non listées ci-dessous.\n"
            "INTERDIT : ignorer ou omettre le champ ALTERNATIVES: d'une ligne GROUPE:.\n"
            "INTERDIT : affirmer qu'une opération est bloquée si son champ ALTERNATIVES: liste des machines.\n"
            "INTERDIT : fusionner plusieurs lignes GROUPE: d'une même commande en une seule affirmation.\n"
        )
        sql_block = (
            f"[DONNÉES SQL — SOURCE DE VÉRITÉ — {mode_label}]\n"
            f"{panne_preamble}\n"
            f"{sql_context}"
        )
    elif "séquence" in intents:
        sequence_preamble = (
            "INSTRUCTION OBLIGATOIRE — SÉQUENÇAGE DES OPÉRATIONS:\n"
            "Les données ci-dessous contiennent des lignes VERDICT calculées en Python depuis les timestamps réels.\n"
            "Chaque VERDICT indique SÉQUENTIEL (gap ≥ 0) ou CHEVAUCHEMENT (gap < 0) entre deux opérations.\n"
            "Étape 1 — Lis chaque ligne VERDICT pour chaque commande.\n"
            "Étape 2 — Si toutes les lignes VERDICT d'une commande disent SÉQUENTIEL, la réponse pour cette commande est 'oui'.\n"
            "Étape 3 — Conclus globalement: si toutes les commandes sont SÉQUENTIEL, réponds 'Oui, toujours'. "
            "Si au moins une est CHEVAUCHEMENT, cite-la explicitement avec le gap.\n"
            "INTERDIT : déduire l'ordre à partir de la logique industrielle générale.\n"
            "INTERDIT : inventer un chevauchement absent des lignes VERDICT.\n"
        )
        sql_block = (
            f"[DONNÉES SQL — SOURCE DE VÉRITÉ — {mode_label}]\n"
            f"{sequence_preamble}\n"
            f"{sql_context}"
        )
    else:
        sql_block = f"[DONNÉES SQL — SOURCE DE VÉRITÉ — {mode_label}]\n{sql_context}"

    parts = [sql_block]

    if faiss_context:
        parts.append(
            f"[CONTEXTE SÉMANTIQUE — indicatif uniquement, "
            f"céder aux données SQL en cas de conflit]\n{faiss_context}"
        )

    lang_hint = (
        "IMPORTANT: Reply in the same language as this question "
        "(ignore the language of previous turns)."
    )
    parts.append(f"[QUESTION — {lang_hint}]\n{question}")
    user_content = "\n\n".join(parts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in memory:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_content})

    num_predict = _token_budget(intents)
    ctx_size    = _num_ctx(intents)
    temp        = _temperature(intents)

    print(f"[RAG] Calling Mistral: num_ctx={ctx_size}, num_predict={num_predict}, temp={temp}")

    # Layer 4 — Mistral
    t2 = time.time()
    try:
        reply = await _call_mistral(
            messages,
            num_predict=num_predict,
            num_ctx=ctx_size,
            temperature=temp,
        )
        print(f"[TIMING] Mistral: {time.time() - t2:.2f}s ({len(reply)} chars)")
    except httpx.TimeoutException:
        print(f"[TIMING] Mistral TIMEOUT after {time.time() - t2:.2f}s")
        return (
            "Mistral a mis trop de temps à répondre. "
            "Essayez une question plus courte ou vérifiez qu'Ollama tourne (ollama serve)."
        )
    except Exception as e:
        return f"Erreur moteur IA : {e}"

    return reply or "Aucune donnée disponible pour répondre à cette question."