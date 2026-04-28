"""
rag/rag_engine.py  —  RAG Engine for Denim Washing Production Planner
======================================================================
FIXES vs previous version:

  BUG FIX — MakespanPM unit confusion (root cause of wrong makespan answers):
  • MakespanPM is a CP-SAT slot offset, NOT real minutes.
    1 PM = 1 real minute (PPD = 1440).
  • _fmt_makespan() now converts PM → real minutes → hours correctly.
  • _build_hard_facts() injects MAKESPAN_REAL as a ground-truth token so
    Mistral always sees the correct human-readable duration (e.g. "9h00").
  • Query E is used as the authoritative makespan source: we compute
    max(EndPM) - min(StartPM) from the actual rows, which is always correct
    regardless of what MakespanPM/MakespanDays say in Query A.
  • The fallback chain is:  Query E derived → Query A PM converted → Query A days.
  • MakespanDays=0 is now explicitly labelled as "< 1 full day" with the real
    hours shown, so Mistral never guesses "4 jours" from a 0-day value.

  PRESERVED from previous version:
  • _build_hard_facts() STATUS / LATE_ORDERS / UNUSED_MACHINES hard tokens
  • _assert_db_rows_safe() security guard
  • _build_objective_section() CP-SAT formula injection
  • All SQL formatters (A–F)
  • FAISS index, embed, FaissIndex class
  • Language detection, gibberish guard, factual/sequence routing

TIMEOUT CHAIN (keep consistent across all layers):
  rag_engine   httpx timeout : 10 min  (this file)
  .NET HttpClient timeout    : 12 min  (Program.cs)
"""

import json
import os
import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL        = "http://localhost:11434"
EMBED_MODEL       = "nomic-embed-text"
LLM_MODEL         = "mistral"
FAISS_INDEX_PATH  = Path(__file__).parent / "faiss_index"
FAISS_META_PATH   = Path(__file__).parent / "faiss_meta.pkl"
CHUNKS_JSON_PATH  = Path(__file__).parent / "chunks.json"
TOP_K             = 3
EMBED_DIM         = 768

# ── Production time model ───────────────────────────────────────────────────
# PPD = Production Minutes Per Day.
# In this project PM values ARE real minutes — confirmed by two sources:
#   1. Angular frontend: export const PPD = 1440, xPx = r.startPM * pxPerMin
#   2. PlanningService.cs: baseDate.AddMinutes(pm)  (not AddMinutes(pm * factor))
# Therefore: 1 PM = 1 real minute, PPD = 1440.
# A 9-hour planning has MakespanPM=540, MakespanDays=0 (540 // 1440 = 0). ✅
PPD = 1440
MINUTES_PER_PM = 1   # 1 PM = 1 real minute

# ── LLM generation options ──────────────────────────────────────────────────
# Three LLM option tiers — chosen per question type in analyze():
#   LOOKUP  : "which machines did X?" → answer is 1-2 lines, no FAISS needed
#   SHORT   : follow-up / rephrasing  → moderate context, medium output
#   FULL    : analysis / explanation  → full context, full output
LLM_OPTIONS_LOOKUP = {
    "num_predict": 80,    # machine list = ~20 tokens; 80 is generous
    "num_ctx":     2048,  # prompt is tiny for lookup (no FAISS, small E summary)
    "temperature": 0.0,   # fully deterministic — just read the list
    "top_p":       1.0,
}
LLM_OPTIONS_SHORT = {
    "num_predict": 200,
    "num_ctx":     4096,
    "temperature": 0.1,
    "top_p":       0.9,
}
LLM_OPTIONS_FULL = {
    "num_predict": 400,
    "num_ctx":     4096,
    "temperature": 0.1,
    "top_p":       0.9,
}
LLM_OPTIONS = LLM_OPTIONS_FULL  # backward-compat alias
OLLAMA_TIMEOUT = 600              # 10 minutes

# Dangerous SQL keywords — strip any db_rows string containing these
_SQL_DANGER = re.compile(
    r'\b(DROP|DELETE|UPDATE|INSERT|TRUNCATE|ALTER|EXEC|EXECUTE|xp_|sp_|UNION)\b',
    re.IGNORECASE,
)

# ── Question type constants ─────────────────────────────────────────────────
#
# LOOKUP questions: "which machine did op X?" / "quelles machines ont fait X?"
#   → answered entirely from the pre-computed summary in _fmt_row_E (tiny).
#     No FAISS retrieval, no full E rows, num_ctx=2048, num_predict=80.
#
# SEQUENCE questions: lot order, timing, "when did X start", "show me the schedule"
#   → need full Query E rows (up to 30), full context, full output.
#
# MAKESPAN questions: "how long", "combien de temps"
#   → need Query E only to compute MAKESPAN_REAL; rows NOT sent to prompt.
#     The summary + HARD FACTS + MAKESPAN section carry the answer.

# Patterns that identify a pure "which machine did operation X" lookup.
# These are answered from the RÉSUMÉ MACHINES section alone — no full E rows.
LOOKUP_PATTERNS = re.compile(
    r"(quell?es?\s+machines?\s+(ont|a|on[t]?|did|have|has|effectu|fait|r[eé]alis|utilis)"
    r"|which\s+machines?\s+(did|performed|ran|processed|used|handled)"
    r"|(ont|a)\s+fait\s+l.op[eé]ration"
    r"|machines?\s+(pour|for|sur|on|doing|used\s+for)\s+(l.op[eé]ration|the\s+op)"
    r"|op[eé]ration\s+\w+\s+(machines?|sur\s+quell?es?))",
    re.IGNORECASE,
)

# Keywords that trigger full Query E rows (sequencing / lot / timing detail).
SEQUENCE_KEYWORDS = [
    "séquence", "sequence", "séquencement",
    "ordre des", "order of",
    "startpm", "endpm", "start_pm", "end_pm",
    "lotidx", "lot idx",
    "quand", "when",
    "planifié", "scheduled",
    "fragmentation",
    "show me the", "montre moi le",
    "détail", "detail",
    "timeline", "chronologie",
]

# Keywords that need Query E for makespan computation but NOT for prompt rows.
MAKESPAN_KEYWORDS = [
    "makespan", "durée", "duree", "combien de temps", "how long",
    "combien d'heure", "combien d'heures", "how many hour",
    "temps total", "total time", "how long does", "combien dure",
    "planning dure", "planning took",
]

# Keywords that need Query E summary (machines per op) — no full rows.
SUMMARY_KEYWORDS = [
    "machine", "fait", "effectué", "réalisé", "utilisé",
    "quel", "quelle", "quels", "quelles",
    "who did", "which machine", "what machine",
    "opération", "operation",
    "poudre", "javellisation", "stonage", "lavage", "rinçage",
    "essorage", "séchage", "sechage", "finition", "trempage",
    "cmd",
]

# Pure factual lookups answerable from Query A alone — FAISS bypassed.
FACTUAL_KEYWORDS = [
    "combien de commande", "nombre de commande", "how many order",
    "combien de ligne", "nombre de ligne", "how many line",
    "date de début", "date debut", "date de generation", "généré le",
    "start date", "generated on",
    "résumé du planning", "summary of the planning",
]

# If any of these appear alongside a factual keyword → compound, not factual
COMPOUND_SIGNALS = [
    "pourquoi", "why", "comment", "how", "réduire", "reduce",
    "améliorer", "improve", "expliquer", "explain", "analyse",
    "recommande", "suggest", "conseille",
    "optimal", "makespan", "statut",
    "combien de temps", "durée", "duree",
]

ROW_CAPS: dict[str, int] = {
    "A": 10,
    "B": 20,
    "C": 20,
    "D": 20,
    "E": 60,
    "F": 15,
}


# ---------------------------------------------------------------------------
# Security guard
# ---------------------------------------------------------------------------

def _assert_db_rows_safe(db_rows: dict) -> dict:
    """
    Validate and sanitise db_rows before any use:
    1. Each key must map to a list.
    2. Cap rows to ROW_CAPS to prevent token-flooding.
    3. Strip string values that contain dangerous SQL keywords.
    Returns a clean copy of db_rows.
    """
    safe = {}
    for key, rows in db_rows.items():
        if not isinstance(rows, list):
            print(f"[SECURITY] db_rows[{key!r}] is not a list — skipping")
            safe[key] = []
            continue

        cap = ROW_CAPS.get(key, 50)
        capped = rows[:cap]

        cleaned = []
        for row in capped:
            if not isinstance(row, dict):
                continue
            clean_row = {}
            for col, val in row.items():
                if isinstance(val, str) and _SQL_DANGER.search(val):
                    print(f"[SECURITY] Suspicious value in db_rows[{key!r}][{col!r}] — redacted")
                    clean_row[col] = "[REDACTED]"
                else:
                    clean_row[col] = val
            cleaned.append(clean_row)

        safe[key] = cleaned

    return safe


# ---------------------------------------------------------------------------
# Chunk loader
# ---------------------------------------------------------------------------

def _load_domain_chunks() -> List[str]:
    if CHUNKS_JSON_PATH.exists():
        try:
            raw = json.loads(CHUNKS_JSON_PATH.read_text(encoding="utf-8"))
            texts = [c["text"] for c in raw if isinstance(c, dict) and c.get("text")]
            if texts:
                print(f"[RAG] Loaded {len(texts)} domain chunks from {CHUNKS_JSON_PATH.name}")
                return texts
        except Exception as e:
            print(f"[RAG] Warning: could not load chunks.json: {e}")

    print("[RAG] Warning: chunks.json not found — using minimal fallback chunks")
    return [
        f"""MODÈLE DE TEMPS — Minutes productives (PPD={PPD})
L'atelier tourne 24h/24, 7j/7. PPD={PPD} minutes par journée réelle.
StartPM et EndPM sont des offsets en minutes réelles depuis le jour 0 du planning.
1 PM = 1 minute réelle. Un planning de 540 PM = 9 heures.
MakespanDays=0 signifie que le planning se termine en moins d'une journée complète — ce n'est PAS une anomalie.
La durée réelle en heures est toujours indiquée dans MAKESPAN_REAL dans les FAITS CERTIFIÉS.""",

        """RÈGLE DES LOTS — Contrainte hard du solveur
lot_size_effectif = min(op.QuantiteLot, machine.CapaciteMax, cmd.Quantite)
NbLots = ceil(Quantite / lot_size_effectif)
DureeTotale opération = DureeTotale_un_lot × NbLots
Une machine avec une petite CapaciteMax génère plus de lots et une durée plus longue.
Si NbLots > 5, chercher une machine avec une plus grande capacité.""",

        """PRIORITÉ ET FONCTION OBJECTIF CP-SAT
Minimize(100_000 × Σ(urgency_weight × tardiness) + makespan)
urgency_weight: urgence 1 → 10, urgence 5 → 2, urgence 10 → 1.
Zéro retard sur une commande urgente est TOUJOURS préféré à un makespan plus court.
OPTIMAL = preuve mathématique d'optimalité. Ne pas relancer si OPTIMAL.
FEASIBLE = solution valide mais non prouvée optimale (time limit atteinte).""",

        """PROBLÈMES COURANTS ET RECOMMANDATIONS
Commande en retard : vérifier NbLots, chercher machine avec plus grande capacité ou machine non utilisée.
Machine NON UTILISÉE avec opérations compatibles : activer mode multi-machines.
Statut FEASIBLE : augmenter MAX_SOLVE_SECONDS à 120-150s.
Ne JAMAIS qualifier une machine de surchargée sur le seul TauxCharge relatif au makespan.
Ne JAMAIS recommander une machine non utilisée sans vérifier ses opérations supportées.""",
    ]


DOMAIN_CHUNKS: List[str] = _load_domain_chunks()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

EXPERT_SYSTEM_PROMPT = """You are an industrial planning expert for the Micwic denim washing workshop.

═══════════════════════════════════════════════════
LANGUAGE RULE — HIGHEST PRIORITY, APPLY FIRST:
Detect the language of the user's question and reply in that EXACT language.
- Question in French → answer entirely in French.
- Question in English → answer entirely in English.
- Mixed → use the dominant language.
- NEVER switch languages mid-answer.
- This rule overrides everything else.
═══════════════════════════════════════════════════

RULES — no exceptions:

1. ANSWER ONLY THE QUESTION ASKED. 2 to 6 sentences max, unless the question explicitly lists multiple points.

2. ZERO FILLER: no greetings, no "Sure!", no "Of course", no rephrasing of the question, no "Based on the data provided…", no "According to the information…". Start directly with the answer.

3. REAL NUMBERS ONLY — only values that appear verbatim in the SQL data sections or HARD FACTS block below. NEVER invent, infer, or approximate a number. If a number is not in the data, do not use it.

4. MISSING DATA: if a specific piece of information is absent from the data, say exactly one sentence: "Cette information n'est pas disponible pour ce planning." (French) or "This information is not available for this planning." (English). Do not guess.

5. FORBIDDEN in the response: SQL column names, internal bracket labels ([RETARDS], [MACHINES]…), phrases like "SQL data", "database", "the data shows", "according to the data". Answer as if you know the facts directly.

6. RECOMMENDATIONS: only mention machines and commands that appear in the provided data. Never reference entities not present in the data.

7. URGENCY: never suggest modifying a command's urgency to improve makespan. Urgency is a fixed business input, not a solver lever.

8. STATUS OPTIMAL — ABSOLUTE RULE: The [HARD FACTS] block contains STATUS=OPTIMAL as a ground-truth token. If STATUS=OPTIMAL, the planning IS mathematically optimal. You MUST NOT say the planning is not optimal, suggest it could be improved, or invent problems. This is a mathematical proof — not an opinion.

9. STATUS FEASIBLE: a valid solution whose optimality is not proven. It may be improvable by re-running with a longer solve time.

10. LATE ORDERS — ABSOLUTE RULE: The [HARD FACTS] block contains either LATE_ORDERS=NONE or LATE_ORDERS=<list>. This is the ONLY source of truth for late orders.
    - If LATE_ORDERS=NONE → there are zero late orders. Do NOT mention any order as late. Do NOT invent CMD numbers, delays, or dates.
    - If LATE_ORDERS=<list> → only those orders are late. Do not mention any order not in that list.
    - Any invented late order is a critical hallucination error.

11. MAKESPAN — ABSOLUTE RULE:
    - The [HARD FACTS] block contains MAKESPAN_REAL=<value> (e.g. "9h00"). This is the ONLY correct makespan to quote.
    - MakespanPM is in real minutes (1 PM = 1 minute). MakespanDays = MakespanPM ÷ 1440.
    - MakespanDays=0 means less than one full day — it is NOT zero hours.
    - Always use the MAKESPAN_REAL value from [HARD FACTS] when answering duration questions.

12. UNUSED MACHINES: if a machine is unused in an OPTIMAL planning, its supported operations do not match the operations required by the planned commands. Never recommend an unused machine without first verifying its operations appear in the planning's operations.

13. MACHINE LOAD: never label a machine overloaded or underused based solely on planned minutes. Only use the explicit Etat field value ("SURCHARGE" or "SOUS-UTILISEE") from the data.

14. MAKESPAN REDUCTION: the only valid suggestions are: enable multi-machine mode, assign lots to an unused machine that supports the same operations, or re-run with longer solve time (only if FEASIBLE, NEVER if OPTIMAL).

15. DATA PRIORITY: SQL data sections and HARD FACTS contain the only allowed numeric values. Domain rule chunks provide context only.

16. QUESTION CLARITY: if the question is incomprehensible, respond with: "Je n'ai pas compris la question. Pouvez-vous la reformuler ?" (French) or "I did not understand the question. Could you please rephrase it?" (English)."""


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

async def embed(texts: List[str]) -> np.ndarray:
    embeddings = []
    async with httpx.AsyncClient(timeout=60) as client:
        for text in texts:
            r = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
            )
            r.raise_for_status()
            embeddings.append(r.json()["embedding"])
    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# FAISS index wrapper
# ---------------------------------------------------------------------------

class FaissIndex:
    def __init__(self):
        self.index  = None
        self.texts  = []
        self.metas  = []

    def add(self, vectors: np.ndarray, texts: List[str], metas: List[dict]):
        try:
            import faiss
        except ImportError:
            print("[FAISS] faiss-cpu not installed — vector search disabled")
            return
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        self.texts.extend(texts)
        self.metas.extend(metas)

    def search(self, query_vec: np.ndarray, k: int = 5):
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        D, I = self.index.search(query_vec.reshape(1, -1), k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], float(dist), self.metas[idx]))
        return results

    def save(self):
        try:
            import faiss
            if self.index:
                faiss.write_index(self.index, str(FAISS_INDEX_PATH))
                with open(FAISS_META_PATH, "wb") as f:
                    pickle.dump({"texts": self.texts, "metas": self.metas}, f)
        except Exception as e:
            print(f"[FAISS] Save error: {e}")

    def load(self) -> bool:
        try:
            import faiss
            if FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists():
                if CHUNKS_JSON_PATH.exists():
                    idx_mtime    = FAISS_INDEX_PATH.stat().st_mtime
                    chunks_mtime = CHUNKS_JSON_PATH.stat().st_mtime
                    if chunks_mtime > idx_mtime:
                        print("[FAISS] chunks.json is newer than FAISS index — rebuilding")
                        return False
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                with open(FAISS_META_PATH, "rb") as f:
                    meta = pickle.load(f)
                self.texts = meta["texts"]
                self.metas = meta["metas"]
                return True
        except Exception as e:
            print(f"[FAISS] Load error: {e}")
        return False


_faiss_index = FaissIndex()


# ---------------------------------------------------------------------------
# Startup: build FAISS index
# ---------------------------------------------------------------------------

async def ensure_domain_knowledge_indexed():
    if _faiss_index.load():
        print(f"[RAG] Loaded FAISS index: {_faiss_index.index.ntotal} vectors")
        return

    print(f"[RAG] Building FAISS index from {len(DOMAIN_CHUNKS)} domain chunks...")
    try:
        vectors = await embed(DOMAIN_CHUNKS)
        metas   = [{"source": "domain", "idx": i} for i in range(len(DOMAIN_CHUNKS))]
        _faiss_index.add(vectors, DOMAIN_CHUNKS, metas)
        _faiss_index.save()
        print(f"[RAG] Indexed {len(DOMAIN_CHUNKS)} domain chunks into FAISS")
    except Exception as e:
        print(f"[RAG] Failed to build index: {e}")


async def index_planning_rows(planning_id: int, planning_text: str):
    chunks = [planning_text[i:i+500] for i in range(0, len(planning_text), 500)]
    if not chunks:
        return
    vectors = await embed(chunks)
    metas   = [{"source": "planning", "planningId": planning_id} for _ in chunks]
    _faiss_index.add(vectors, chunks, metas)
    _faiss_index.save()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Question type enum values
_QTYPE_LOOKUP       = "lookup"       # "which machine did op X" → summary only, tiny prompt
_QTYPE_MACHINE_LOAD = "machine_load" # "surcharge / sous-utilisée / charge" → Query C only, no FAISS
_QTYPE_MAKESPAN     = "makespan"     # "how long" → E for computation, no rows in prompt
_QTYPE_SUMMARY      = "summary"      # "machine/op" → E summary only, no full rows
_QTYPE_SEQUENCE     = "sequence"     # lot order / timing → full E rows
_QTYPE_ANALYSIS     = "analysis"     # optimal? late orders? general → no E needed

# Patterns for machine load / utilisation questions.
# These are answered directly from Query C (Etat field) — no FAISS, no Query E.
MACHINE_LOAD_PATTERNS = re.compile(
    r"(surcharg|surcharges|overload|over.load"
    r"|sous.utilis|under.utilis|underus"
    r"|taux.charg|taux de charge|utilisation.rate|charge.rate"
    r"|quelle.*machine.*charg|which.*machine.*load"
    r"|machine.*surcharg|machine.*sous.utilis"
    r"|quel.*taux|what.*utilisation|what.*utilization"
    r"|machines?\s+(les\s+)?(plus\s+)?(charg|activ|occup|busy|load)"
    r"|how\s+(loaded|busy|utilized)\s+are)",
    re.IGNORECASE,
)


def _classify_question(question: str) -> str:
    """
    Classify the question into one of five types to control:
      - whether Query E rows are sent to the LLM prompt (and how many)
      - which LLM options tier to use (LOOKUP / SHORT / FULL)
      - whether FAISS retrieval is needed

    Priority order (first match wins):
      LOOKUP   > MAKESPAN > SEQUENCE > SUMMARY > ANALYSIS
    """
    q = question.strip()

    # LOOKUP: "quelles machines ont fait l'opération Poudre?"
    if LOOKUP_PATTERNS.search(q):
        return _QTYPE_LOOKUP

    q_lower = q.lower()

    # MACHINE_LOAD: "quelles machines sont surchargées / sous-utilisées / quel taux de charge?"
    # Answered from Query C (Etat field) only — no FAISS, no Query E, LLM_OPTIONS_LOOKUP tier.
    if MACHINE_LOAD_PATTERNS.search(q):
        return _QTYPE_MACHINE_LOAD

    # MAKESPAN: "how long is the makespan?"
    if any(kw in q_lower for kw in MAKESPAN_KEYWORDS):
        return _QTYPE_MAKESPAN

    # SEQUENCE: "show me the lot order", "when did CMD1 start?"
    if any(kw in q_lower for kw in SEQUENCE_KEYWORDS):
        return _QTYPE_SEQUENCE

    # SUMMARY: "which machines used Poudre?" (looser than LOOKUP)
    if any(kw in q_lower for kw in SUMMARY_KEYWORDS):
        return _QTYPE_SUMMARY

    return _QTYPE_ANALYSIS


def _needs_sequence_data(question: str) -> bool:
    """Legacy helper — kept for any external callers."""
    qtype = _classify_question(question)
    return qtype in (_QTYPE_SEQUENCE, _QTYPE_SUMMARY, _QTYPE_LOOKUP, _QTYPE_MAKESPAN)


def _is_factual_lookup(question: str) -> bool:
    q_lower = question.lower().strip()
    if any(sig in q_lower for sig in COMPOUND_SIGNALS):
        return False
    return any(kw in q_lower for kw in FACTUAL_KEYWORDS)


_SHORT_QUESTION_PATTERNS = re.compile(
    r"^\s*(pourquoi|why|comment|how|really|vraiment|et alors|so what"
    r"|explique|explain|détaille|detail|précise|clarif"
    r"|ok|oui|non|yes|no|sure|d'accord|agreed)\b",
    re.IGNORECASE,
)

def _is_short_followup(question: str) -> bool:
    q = question.strip()
    if len(q) < 40:
        return True
    if _SHORT_QUESTION_PATTERNS.match(q):
        return True
    return False


def _is_gibberish(question: str) -> bool:
    q = question.strip()
    if not q:
        return True
    letters = re.sub(r'[^a-zA-ZàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]', '', q)
    if len(letters) < 3:
        return True
    non_space = re.sub(r'\s', '', q)
    if non_space and len(letters) / len(non_space) < 0.4:
        return True
    consonant_runs = re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{5,}', q)
    if consonant_runs:
        return True
    return False


_FRENCH_MARKERS = re.compile(
    r'\b(le|la|les|un|une|des|est|sont|pas|ne|ce|cette|ces|mon|ton|son|'
    r'que|qui|quoi|quel|quelle|quels|quelles|et|ou|mais|donc|or|ni|car|'
    r'est-ce|y a-t-il|combien|comment|pourquoi|quand|votre|notre|leur|leurs|'
    r'avec|sans|pour|sur|dans|par|de|du|au|aux|en|entre|vers|chez|'
    r'retard|commande|planning|machine|optimal|makespan|lot)\b',
    re.IGNORECASE,
)

_ENGLISH_MARKERS = re.compile(
    r'\b(the|a|an|is|are|was|were|have|has|do|does|did|will|would|could|'
    r'should|may|might|can|what|which|who|how|when|where|why|'
    r'any|late|order|machine|planning|optimal|makespan|lot|operation|'
    r'used|unused|running|show|tell|give|find|list|explain)\b',
    re.IGNORECASE,
)

def _detect_language(question: str) -> str:
    fr_count = len(_FRENCH_MARKERS.findall(question))
    en_count = len(_ENGLISH_MARKERS.findall(question))
    return 'en' if en_count > fr_count else 'fr'


_KNOWN_OPERATIONS = [
    "poudre", "javellisation", "stonage", "lavage", "rinçage",
    "essorage", "séchage", "sechage", "finition", "trempage",
]

def _extract_operation_filter(question: str) -> Optional[str]:
    q_lower = question.lower()
    for op in _KNOWN_OPERATIONS:
        if op in q_lower:
            return op
    return None

def _extract_command_filter(question: str) -> Optional[str]:
    m = re.search(r'\bcmd\s*(\w+)', question.lower())
    if m:
        return "cmd" + m.group(1)
    return None


# ---------------------------------------------------------------------------
# Makespan conversion — THE FIX
# ---------------------------------------------------------------------------

def _pm_to_real_minutes(pm_value) -> Optional[int]:
    """
    Convert a CP-SAT PM slot value to real minutes.
    1 PM = MINUTES_PER_PM real minutes (= 1, since PPD=1440).
    Returns None if the value cannot be interpreted.
    """
    try:
        return int(pm_value) * MINUTES_PER_PM
    except (TypeError, ValueError):
        return None


def _minutes_to_hhmm(total_minutes: int) -> str:
    """Format an integer number of minutes as 'Xh00' or 'Xh30' etc."""
    h   = total_minutes // 60
    rem = total_minutes % 60
    if rem:
        return f"{h}h{rem:02d}"
    return f"{h}h00"


def _derive_makespan_from_query_e(rows_e: list) -> Optional[int]:
    """
    Derive the true makespan in real minutes directly from Query E rows.
    This is the most reliable source: max(EndPM) - min(StartPM), converted
    to real minutes.  Returns None if rows_e is empty or has no valid values.
    """
    if not rows_e:
        return None
    try:
        start_pms = [int(r["StartPM"]) for r in rows_e if "StartPM" in r]
        end_pms   = [int(r["EndPM"])   for r in rows_e if "EndPM"   in r]
        if not start_pms or not end_pms:
            return None
        span_pm = max(end_pms) - min(start_pms)
        return span_pm * MINUTES_PER_PM
    except (TypeError, ValueError, KeyError):
        return None


def _compute_makespan_real(db_rows: dict) -> str:
    """
    Compute the real makespan string using the best available source.

    Priority chain:
      1. Query E rows  → max(EndPM) - min(StartPM) × MINUTES_PER_PM  (most accurate)
      2. Query A MakespanPM × MINUTES_PER_PM                          (stored value)
      3. Query A MakespanDays label                                    (coarse fallback)

    MakespanPM is in real minutes (1 PM = 1 real minute, PPD=1440).
    MakespanDays=0 is normal for short plannings — it does NOT mean zero hours.

    Returns a human-readable string like "9h00 (moins d'une journée complète)".
    """

    def _format(real_minutes: int, source: str) -> str:
        hhmm = _minutes_to_hhmm(real_minutes)
        days = real_minutes // (24 * 60)
        suffix = f" (moins d'une journée complète — source: {source})" if days == 0 \
                 else f" ({days} jour(s) — source: {source})"
        print(f"[RAG] MAKESPAN_REAL={hhmm} via {source}")
        return hhmm + suffix

    # ── Source 1: derive from Query E actual rows ─────────────────────────
    rows_e = db_rows.get("E", [])
    if rows_e:
        real_minutes = _derive_makespan_from_query_e(rows_e)
        if real_minutes is not None and real_minutes > 0:
            return _format(real_minutes, "Query E rows")
        print(f"[RAG] _derive_makespan_from_query_e returned {real_minutes} — trying Query A")
    else:
        print("[RAG] Query E is empty — falling back to Query A for makespan")

    # ── Source 2: MakespanPM from Query A × MINUTES_PER_PM ───────────────
    rows_a = db_rows.get("A", [])
    if rows_a:
        pm_val = rows_a[0].get("MakespanPM")
        print(f"[RAG] Query A MakespanPM raw value = {pm_val!r}")
        real_minutes = _pm_to_real_minutes(pm_val)
        if real_minutes is not None and real_minutes > 0:
            return _format(real_minutes, f"Query A MakespanPM={pm_val}×{MINUTES_PER_PM}min")

        # ── Source 3: MakespanDays coarse fallback ────────────────────────
        days_val = rows_a[0].get("MakespanDays")
        print(f"[RAG] Query A MakespanDays raw value = {days_val!r}")
        if days_val is not None:
            try:
                d = int(days_val)
                if d == 0:
                    print("[RAG] MakespanDays=0 — planning is sub-day, exact hours unavailable from Query A")
                    return "moins d'une journée complète (durée précise non disponible — vérifier MakespanPM en base)"
                return f"{d} jour(s) complet(s) (source: Query A MakespanDays)"
            except (TypeError, ValueError):
                pass
    else:
        print("[RAG] Query A is also empty — no makespan source available")

    return "non disponible (aucune donnée reçue)"


# ---------------------------------------------------------------------------
# HARD FACTS builder — with makespan fix
# ---------------------------------------------------------------------------

def _build_hard_facts(db_rows: dict, lang: str) -> str:
    """
    Build ground-truth token lines from db_rows BEFORE the LLM call.
    Injected at the absolute top of user_prompt.
    Mistral sees these as facts it CANNOT contradict.

    FIX: Now includes MAKESPAN_REAL computed correctly from PM slots.
    """
    facts = []

    # ── Makespan (FIX — always computed from PM, not days) ──────────────────
    makespan_real = _compute_makespan_real(db_rows)
    if lang == "en":
        facts.append(
            f"MAKESPAN_REAL={makespan_real}  ← the ONLY correct duration to quote. "
            "Never use MakespanDays or raw MakespanPM to answer duration questions. "
            "MakespanDays=0 means less than one full day — NOT zero hours."
        )
    else:
        facts.append(
            f"MAKESPAN_REAL={makespan_real}  ← la SEULE durée correcte à citer. "
            "Ne jamais utiliser MakespanDays brut ni MakespanPM brut pour répondre à la durée. "
            "MakespanDays=0 signifie moins d'une journée complète — PAS zéro heure."
        )

    # ── Optimality status (from Query A) ────────────────────────────────────
    rows_a = db_rows.get("A", [])
    if rows_a:
        status = str(rows_a[0].get("Statut", "")).lower().strip()
        if status == "optimal":
            if lang == "en":
                facts.append(
                    "STATUS=OPTIMAL  ← mathematical proof of global optimality. "
                    "The planning IS optimal. Never say it is not optimal. "
                    "Never suggest re-running. Never invent problems."
                )
            else:
                facts.append(
                    "STATUS=OPTIMAL  ← preuve mathématique d'optimalité globale. "
                    "Le planning EST optimal. Ne jamais dire qu'il n'est pas optimal. "
                    "Ne jamais suggérer de le relancer. Ne jamais inventer de problème."
                )
        elif status in ("feasible", "feasible_lns_only"):
            if lang == "en":
                facts.append(
                    "STATUS=FEASIBLE  ← valid solution, optimality not proven. "
                    "May be improvable by re-running with longer solve time."
                )
            else:
                facts.append(
                    "STATUS=FEASIBLE  ← solution valide, optimalité non prouvée. "
                    "Peut être améliorée en relançant avec un temps de résolution plus long."
                )

    # ── Late orders (from Query B) ───────────────────────────────────────────
    rows_b = db_rows.get("B", [])
    if not rows_b:
        if lang == "en":
            facts.append(
                "LATE_ORDERS=NONE  ← the database confirmed zero late orders. "
                "Do NOT mention any order as late. "
                "Do NOT invent any CMD number, delay, or deadline. "
                "Any invented late order is a hallucination error."
            )
        else:
            facts.append(
                "LATE_ORDERS=NONE  ← la base de données confirme zéro commande en retard. "
                "NE PAS mentionner de commande en retard. "
                "NE PAS inventer de numéro CMD, de retard ou de date limite. "
                "Toute commande inventée est une erreur critique."
            )
    else:
        late_cmds = [str(r.get("NumeroCommande", "?")) for r in rows_b]
        late_list = ", ".join(late_cmds)
        if lang == "en":
            facts.append(
                f"LATE_ORDERS={late_list}  ← only these orders are late. "
                "Do NOT mention any other order as late."
            )
        else:
            facts.append(
                f"LATE_ORDERS={late_list}  ← seules ces commandes sont en retard. "
                "NE PAS mentionner d'autre commande comme étant en retard."
            )

    # ── Unused machines (from Query F) ──────────────────────────────────────
    rows_f = db_rows.get("F", [])
    if not rows_f:
        if lang == "en":
            facts.append("UNUSED_MACHINES=NONE  ← all functional machines are used.")
        else:
            facts.append("UNUSED_MACHINES=NONE  ← toutes les machines fonctionnelles sont utilisées.")
    else:
        unused_names = [str(r.get("NomMachine", "?")) for r in rows_f]
        if lang == "en":
            facts.append(f"UNUSED_MACHINES={', '.join(unused_names)}")
        else:
            facts.append(f"MACHINES_NON_UTILISÉES={', '.join(unused_names)}")

    header = (
        "━━━ HARD FACTS — GROUND TRUTH FROM DATABASE ━━━\n"
        "These facts are mathematically verified. You MUST NOT contradict them.\n"
    ) if lang == "en" else (
        "━━━ FAITS CERTIFIÉS — VÉRITÉ TERRAIN DEPUIS LA BASE DE DONNÉES ━━━\n"
        "Ces faits sont vérifiés mathématiquement. Vous NE POUVEZ PAS les contredire.\n"
    )

    return header + "\n".join(f"  {f}" for f in facts) + "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"


def _build_compact_reminder(db_rows: dict, lang: str) -> str:
    """
    Ultra-compact restatement of the critical facts — appended at the VERY END
    of user_prompt as a safety net. If Mistral's context window truncates from
    the top (old prompt too long), it still sees these constraints last.
    """
    rows_a = db_rows.get("A", [])
    rows_b = db_rows.get("B", [])
    status = str(rows_a[0].get("Statut", "?")).upper() if rows_a else "?"
    makespan = _compute_makespan_real(db_rows)
    late = "NONE" if not rows_b else ", ".join(r.get("NumeroCommande","?") for r in rows_b)

    if lang == "en":
        return (
            f"\n⚡ REMINDER (highest priority):\n"
            f"  STATUS={status} | MAKESPAN={makespan} | LATE_ORDERS={late}\n"
            f"  If STATUS=OPTIMAL → say the planning IS optimal. Do not invent problems.\n"
            f"  If LATE_ORDERS=NONE → say zero late orders. Do not invent any CMD.\n"
        )
    return (
        f"\n⚡ RAPPEL PRIORITAIRE (priorité maximale) :\n"
        f"  STATUT={status} | MAKESPAN={makespan} | COMMANDES_EN_RETARD={late}\n"
        f"  Si STATUT=OPTIMAL → dire que le planning EST optimal. Ne pas inventer de problème.\n"
        f"  Si COMMANDES_EN_RETARD=NONE → dire zéro retard. Ne pas inventer de numéro CMD.\n"
    )



# ---------------------------------------------------------------------------
# CP-SAT objective section
# ---------------------------------------------------------------------------

def _build_objective_section(lang: str) -> str:
    """
    Always inject the CP-SAT objective formula so the LLM understands
    what 'optimal' means and cannot invent alternative criteria.
    """
    if lang == "en":
        return (
            "\n[CP-SAT OBJECTIVE — read-only context]\n"
            "  Minimize(100_000 × Σ(urgency_weight × tardiness) + makespan)\n"
            "  urgency_weight: urgency 1 → 10, urgency 5 → 2, urgency 10 → 1.\n"
            "  Zero tardiness on urgent orders is ALWAYS preferred over a shorter makespan.\n"
            "  An OPTIMAL result means no better objective value exists — proven by the solver.\n"
        )
    return (
        "\n[FONCTION OBJECTIF CP-SAT — contexte en lecture seule]\n"
        "  Minimize(100_000 × Σ(urgency_weight × tardiness) + makespan)\n"
        "  urgency_weight : urgence 1 → 10, urgence 5 → 2, urgence 10 → 1.\n"
        "  Zéro retard sur commande urgente est TOUJOURS préféré à un makespan plus court.\n"
        "  OPTIMAL = aucune meilleure valeur objective n'existe — prouvé par le solveur.\n"
    )


# ---------------------------------------------------------------------------
# SQL formatters
# ---------------------------------------------------------------------------

def _fmt_row_A(rows: list) -> str:
    if not rows:
        return "  (aucune donnée)\n"
    out = ""
    for r in rows:
        pm_val       = r.get("MakespanPM")
        real_minutes = _pm_to_real_minutes(pm_val)
        real_str     = _minutes_to_hhmm(real_minutes) if real_minutes is not None else "?"
        days_val     = r.get("MakespanDays", "?")

        out += f"  • Planning ID           : {r.get('Id', '?')}\n"
        out += f"  • Statut                : {r.get('Statut', '?')}\n"
        out += f"  • Généré le             : {r.get('DateGeneration', '?')}\n"
        out += f"  • Date de début         : {r.get('DateDebut', '?')}\n"
        out += f"  • MakespanPM (slots)    : {pm_val}  ⚠️ UNITÉ INTERNE — ne pas citer directement\n"
        out += f"  • MakespanDays          : {days_val}  ⚠️ 0 = moins d'une journée, PAS zéro heure\n"
        out += f"  • Durée réelle calculée : {real_str}  ✅ UTILISER CETTE VALEUR pour répondre\n"
        out += f"  • Commandes             : {r.get('NombreCommandes', '?')}\n"
        out += f"  • Lignes planifiées     : {r.get('NombreLignes', '?')}\n"
    return out


def _fmt_row_B(rows: list, cap: int) -> str:
    if not rows:
        return (
            "  LATE_ORDERS=NONE\n"
            "  ✅ Zéro commande en retard confirmé par la base de données.\n"
            "  ⛔ NE PAS mentionner de commande en retard. NE PAS inventer de CMD.\n"
        )
    out = ""
    for r in rows[:cap]:
        cmd      = r.get("NumeroCommande", "?")
        urg      = r.get("Urgence", "?")
        deadline = r.get("Deadline", "?")
        fin      = r.get("FinPlanifiee", "?")
        retard   = r.get("JoursRetard", "?")
        recette  = r.get("NomRecette", "?")
        out += f"  🔴 Commande {cmd} (urgence {urg}, recette {recette})\n"
        out += f"     Date limite: {deadline} | Fin planifiée: {fin} | Retard: {retard} jour(s)\n"
    if len(rows) > cap:
        out += f"  ... ({len(rows) - cap} autres commandes en retard)\n"
    return out


def _fmt_row_C(rows: list, cap: int) -> str:
    if not rows:
        return "  (aucune donnée)\n"
    out = ""
    for r in rows[:cap]:
        nom     = r.get("NomMachine", "?")
        capa    = r.get("CapaciteMax", "?")
        ops_raw = r.get("OperationsList") or r.get("Operations", "?")
        ops     = ", ".join(ops_raw) if isinstance(ops_raw, list) else str(ops_raw)
        nb_cmd  = r.get("NbCommandes", "?")
        mins    = r.get("MinutesPlanifiees", "?")
        etat    = r.get("Etat", "")
        if mins in (0, "0", None, "?"):
            icon  = "⚪"
            label = "NON UTILISÉE"
        else:
            icon  = "🟢"
            label = f"EN SERVICE — état: {etat}" if etat else "EN SERVICE"
        out += f"  {icon} {nom} (capacité {capa} pièces/lot | opérations: {ops}) — {label}\n"
        out += f"     Commandes affectées: {nb_cmd} | Minutes planifiées: {mins}\n"
    if len(rows) > cap:
        out += f"  ... ({len(rows) - cap} autres machines)\n"
    return out


def _fmt_row_D(rows: list, cap: int) -> str:
    if not rows:
        return "  ✅ Aucune fragmentation excessive détectée (tous NbLots ≤ 3).\n"
    out = ""
    for r in rows[:cap]:
        cmd     = r.get("NumeroCommande", "?")
        op      = r.get("NomOperation", "?")
        machine = r.get("MachineName", "?")
        nb_lots = r.get("NbLots", "?")
        lot_eff = r.get("LotSizeEffectif", "?")
        cap_m   = r.get("CapaciteMaxMachine", "?")
        lot_rec = r.get("LotSizeRecette", "?")
        qte_cmd = r.get("QuantiteCommande", "?")
        out += f"  🟠 Commande {cmd} — opération '{op}' sur {machine}\n"
        out += f"     Lots: {nb_lots} | Taille lot effective: {lot_eff} | Capacité machine: {cap_m} | Taille lot recette: {lot_rec} | Quantité commande: {qte_cmd}\n"
    if len(rows) > cap:
        out += f"  ... ({len(rows) - cap} autres fragmentations)\n"
    return out


def _fmt_row_E(rows: list, cap: int,
               op_filter: Optional[str] = None,
               cmd_filter: Optional[str] = None) -> str:
    if not rows:
        return "  ⚠️  Aucune donnée de séquencement disponible pour ce planning.\n"

    if op_filter:
        filtered = [r for r in rows if op_filter in (r.get("NomOperation") or "").lower()]
        if filtered:
            rows = filtered

    if cmd_filter:
        filtered = [r for r in rows if cmd_filter in (r.get("NumeroCommande") or "").lower()]
        if filtered:
            rows = filtered

    out = ""
    for r in rows[:cap]:
        cmd     = r.get("NumeroCommande", "?")
        op      = r.get("NomOperation", "?")
        ordre   = r.get("Ordre", "?")
        mach    = r.get("MachineName", "?")
        ds      = r.get("DateStart", "?")
        de      = r.get("DateEnd", "?")
        lot     = r.get("LotIdx", "?")
        nb      = r.get("NbLots", "?")
        duree   = r.get("DureeMinutes", "?")
        out += f"  {cmd} | op {ordre}:{op} | {mach} | lot {lot}/{nb} | {ds} → {de} | cycle {duree}min\n"
    if len(rows) > cap:
        out += f"  ... ({len(rows) - cap} autres lignes de séquencement)\n"

    # Pre-computed summaries — unambiguous answers for "which machine did X"
    op_to_machines: dict[str, set] = {}
    cmd_to_ops: dict[str, set] = {}
    for r in rows:
        op_name  = r.get("NomOperation", "?")
        mach     = r.get("MachineName", "?")
        cmd_name = r.get("NumeroCommande", "?")
        op_to_machines.setdefault(op_name, set()).add(mach)
        cmd_to_ops.setdefault(cmd_name, set()).add(op_name)

    out += "\n  ── RÉSUMÉ MACHINES PAR OPÉRATION ──\n"
    for op_name, machs in sorted(op_to_machines.items()):
        out += f"  • {op_name} → {', '.join(sorted(machs))}\n"

    out += "\n  ── RÉSUMÉ OPÉRATIONS PAR COMMANDE ──\n"
    for cmd_name, ops in sorted(cmd_to_ops.items()):
        out += f"  • {cmd_name} → {', '.join(sorted(ops))}\n"

    return out


def _fmt_row_F(rows: list, cap: int) -> str:
    if not rows:
        return "  ✅ Toutes les machines fonctionnelles sont utilisées dans ce planning.\n"
    out = ""
    for r in rows[:cap]:
        nom     = r.get("NomMachine", "?")
        capa    = r.get("CapaciteMax", "?")
        ops_raw = r.get("OperationsList") or r.get("Operations", "?")
        ops     = ", ".join(ops_raw) if isinstance(ops_raw, list) else str(ops_raw)
        out += f"  ⚠️  {nom} — NON UTILISÉE | capacité: {capa} pièces/lot | opérations supportées: {ops}\n"
    if len(rows) > cap:
        out += f"  ... ({len(rows) - cap} autres machines non utilisées)\n"
    return out


async def run_sql_queries(planning_id: int, db_rows: dict, question: str = "") -> str:
    """
    Build the SQL context block for the LLM prompt.

    Query E row inclusion is controlled by question type (_classify_question):
      LOOKUP   → 0 rows   — summary block only (3 lines). Tiny prompt, fast answer.
      MAKESPAN → 0 rows   — E used for makespan computation only, not sent to LLM.
      SUMMARY  → 0 rows   — summary block only (machines per op, ops per cmd).
      SEQUENCE → 30 rows  — full detail needed for lot/timing questions.
      ANALYSIS → 0 rows   — no E needed (optimal? late? general).
    """
    if not db_rows:
        return (
            "⚠️  AUCUNE DONNÉE SQL REÇUE POUR CE PLANNING.\n"
            "Les données de diagnostic n'ont pas pu être récupérées depuis la base de données.\n"
            "Le modèle ne peut pas analyser ce planning sans données réelles.\n"
            "Veuillez réessayer ou contacter l'administrateur si le problème persiste."
        )

    qtype      = _classify_question(question)
    op_filter  = _extract_operation_filter(question)
    cmd_filter = _extract_command_filter(question)
    sections   = []

    print(f"[RAG] question_type={qtype!r} op_filter={op_filter!r} cmd_filter={cmd_filter!r}")

    # ── Makespan block (always shown — redundant safety net) ──────────────
    makespan_line = _compute_makespan_real(db_rows)
    sections.append(
        f"[MAKESPAN RÉEL DU PLANNING #{planning_id}]\n"
        f"  ✅ Durée totale réelle : {makespan_line}\n"
        f"  ⚠️  Ne pas utiliser MakespanDays=0 ni MakespanPM brut pour répondre à la durée.\n"
    )

    # ── For MACHINE_LOAD questions: only Query C (Etat column) is needed ────
    # "Quelles machines sont surchargées?" → the Etat field in row C IS the answer.
    # No FAISS, no Query E, no Query D/F. Tiny prompt → LLM_OPTIONS_LOOKUP tier.
    if qtype == _QTYPE_MACHINE_LOAD:
        rows_c = db_rows.get("C", [])
        if not rows_c:
            sections.append("[CHARGE MACHINES]\n  (aucune donnée de charge machine disponible)\n")
        else:
            lines = []
            for r in rows_c:
                nom   = r.get("NomMachine", "?")
                etat  = r.get("Etat", "NOMINAL")
                pct   = r.get("TauxChargePct", "?")
                mins  = r.get("MinutesPlanifiees", "?")
                nb    = r.get("NbCommandes", "?")
                icon  = "🔴" if etat == "SURCHARGE" else ("🟡" if etat == "SOUS-UTILISEE" else "🟢")
                lines.append(
                    f"  {icon} {nom} — {etat} | charge: {pct}% | {mins} min planifiées | {nb} commande(s)"
                )
            sections.append("[CHARGE MACHINES]\n" + "\n".join(lines) + "\n")
        return "\n".join(sections)

    # ── For LOOKUP questions: only show the operation→machine summary ─────
    # The full sections A/B/C/D/F are not needed and would waste context.
    if qtype == _QTYPE_LOOKUP:
        rows_e = db_rows.get("E", [])
        # Build just the RÉSUMÉ MACHINES PAR OPÉRATION block
        op_to_machines: dict = {}
        for r in rows_e:
            op_name = r.get("NomOperation", "?")
            mach    = r.get("MachineName", "?")
            op_to_machines.setdefault(op_name, set()).add(mach)

        if op_to_machines:
            summary = "\n".join(
                f"  • {op} → {', '.join(sorted(machs))}"
                for op, machs in sorted(op_to_machines.items())
            )
        else:
            summary = "  (aucune donnée d'opération disponible pour ce planning)"

        sections.append(f"[MACHINES PAR OPÉRATION]\n{summary}\n")
        return "\n".join(sections)

    # ── Standard sections for non-LOOKUP questions ────────────────────────
    sections.append(
        f"[INFO PLANNING #{planning_id}]\n"
        + _fmt_row_A(db_rows.get("A", []))
    )
    sections.append(
        "[RETARDS]\n"
        + _fmt_row_B(db_rows.get("B", []), ROW_CAPS["B"])
    )
    sections.append(
        "[MACHINES]\n"
        + _fmt_row_C(db_rows.get("C", []), ROW_CAPS["C"])
    )
    sections.append(
        "[FRAGMENTATION]\n"
        + _fmt_row_D(db_rows.get("D", []), ROW_CAPS["D"])
    )

    # ── Query E: rows vs summary vs nothing based on question type ────────
    rows_e = db_rows.get("E", [])
    if qtype == _QTYPE_SEQUENCE:
        # Full rows needed for lot-level / timing questions
        e_cap = min(30, ROW_CAPS["E"])
        sections.append(
            "[DÉTAIL OPÉRATIONS]\n"
            + _fmt_row_E(rows_e, e_cap, op_filter=op_filter, cmd_filter=cmd_filter)
        )
    elif qtype == _QTYPE_SUMMARY:
        # Summary only — no individual lot rows
        sections.append(
            "[DÉTAIL OPÉRATIONS — résumé uniquement]\n"
            + _fmt_row_E(rows_e, 0, op_filter=op_filter, cmd_filter=cmd_filter)
        )
    else:
        # MAKESPAN or ANALYSIS — E used for computation only, not shown
        sections.append(
            "[DÉTAIL OPÉRATIONS]\n"
            "  (non inclus pour cette question)\n"
        )

    sections.append(
        "[MACHINES NON UTILISÉES]\n"
        + _fmt_row_F(db_rows.get("F", []), ROW_CAPS["F"])
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Core RAG pipeline
# ---------------------------------------------------------------------------

async def analyze(
    planning_id: int,
    question:    str,
    db_rows:     dict,
) -> str:
    """
    Full RAG pipeline:
      0. Security: sanitise db_rows
      1. Gibberish guard
      2. Detect language
      3. Build HARD FACTS block (includes MAKESPAN_REAL — the key fix)
      4. Detect factual lookup → skip FAISS if yes
      5. Embed question + retrieve top-K domain chunks from FAISS
      6. Format SQL data (A–F) as structured sections
      7. Assemble prompt: HARD FACTS first, then SQL, then domain, then question
      8. Call Mistral via Ollama
      9. Return answer
    """

    # ── Step 0: security sanitisation ────────────────────────────────────────
    db_rows = _assert_db_rows_safe(db_rows)

    # ── Step 1: gibberish guard ───────────────────────────────────────────────
    if _is_gibberish(question):
        lang = _detect_language(question)
        if lang == "en":
            return "I did not understand your question. Could you please rephrase it?"
        return "Je n'ai pas compris votre question. Pouvez-vous la reformuler ?"

    # ── Step 2: detect language ───────────────────────────────────────────────
    lang = _detect_language(question)
    lang_directive = (
        "LANGUAGE: The user wrote in ENGLISH. Your entire response MUST be in English."
        if lang == "en" else
        "LANGUE: L'utilisateur a écrit en FRANÇAIS. Toute la réponse DOIT être en français."
    )

    # ── Step 3: build HARD FACTS (includes makespan fix) ─────────────────────
    hard_facts = _build_hard_facts(db_rows, lang)

    # ── Step 4: FAISS retrieval decision ─────────────────────────────────────
    # Skip for LOOKUP (answer is in E summary), MAKESPAN (computed from E),
    # and pure factual queries (answered from Query A alone).
    qtype_early = _classify_question(question)
    factual      = _is_factual_lookup(question)
    skip_faiss   = factual or qtype_early in (_QTYPE_LOOKUP, _QTYPE_MAKESPAN, _QTYPE_MACHINE_LOAD)

    if skip_faiss:
        vector_context = ""
        reason = "factual" if factual else qtype_early
        print(f"[RAG] FAISS bypassed ({reason}) for: {question[:80]}")
    else:
        try:
            q_vec = await embed([question])
        except Exception as e:
            print(f"[RAG] Embedding error: {e}")
            q_vec = np.zeros((1, EMBED_DIM), dtype=np.float32)

        retrieved = _faiss_index.search(q_vec[0], k=TOP_K)
        if retrieved:
            chunks_text    = [text[:400] for text, score, _ in retrieved]
            vector_context = "\n\n".join(chunks_text)
        else:
            vector_context = ""

    # ── Step 5: format SQL data ───────────────────────────────────────────────
    sql_context = await run_sql_queries(planning_id, db_rows, question)

    # ── Step 6: domain chunk section ─────────────────────────────────────────
    if vector_context:
        domain_section = (
            "\n[SOLVER RULES — context only, never use numbers from this section]\n"
            + vector_context
            + "\n"
        )
    else:
        domain_section = ""

    # ── Step 7: objective function section (always present) ───────────────────
    objective_section = _build_objective_section(lang)

    # ── Step 8: assemble prompt — HARD FACTS at top + compact reminder at bottom ──
    compact_reminder = _build_compact_reminder(db_rows, lang)
    user_prompt = f"""{hard_facts}

{lang_directive}

{objective_section}

[PLANNING #{planning_id} DATA — the ONLY values you are allowed to use in your response]
Do not invent, infer, or substitute any value not present in the data below.
The MAKESPAN_REAL in [HARD FACTS] above is the correct duration — use it for all duration questions.

{sql_context}
{domain_section}
[QUESTION]
{question}

Answer only from the data above. If information is absent, say so in one sentence. Never mention bracket labels in your response. Do not repeat the question.
{compact_reminder}"""

    messages = [
        {"role": "system", "content": EXPERT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    qtype   = _classify_question(question)
    if qtype in (_QTYPE_LOOKUP, _QTYPE_MACHINE_LOAD):
        options = LLM_OPTIONS_LOOKUP   # fast: tiny prompt, deterministic, 80 tokens
    elif qtype in (_QTYPE_MAKESPAN, _QTYPE_SUMMARY) or _is_short_followup(question):
        options = LLM_OPTIONS_SHORT
    else:
        options = LLM_OPTIONS_FULL
    payload = {
        "model":      LLM_MODEL,
        "messages":   messages,
        "stream":     False,
        "keep_alive": "10m",
        "options":    options,
    }

    # ── Step 9: call Mistral ──────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        reply = r.json().get("message", {}).get("content", "")
        return reply
    except httpx.TimeoutException:
        if lang == "en":
            return (
                "Mistral took too long to respond. "
                "Try a shorter or more specific question. "
                "For better performance, ensure Ollama is running on GPU."
            )
        return (
            "Mistral a mis trop de temps à répondre. "
            "Essayez une question plus courte ou plus ciblée. "
            "Pour de meilleures performances, vérifiez qu'Ollama tourne sur GPU."
        )
    except Exception as e:
        return f"[RAG ERROR] LLM call failed: {e}"