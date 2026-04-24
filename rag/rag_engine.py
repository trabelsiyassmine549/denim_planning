"""
rag/rag_engine.py  —  RAG Engine for Denim Washing Production Planner
======================================================================
FIXES in this version:
  1. num_predict raised to 600  — was 350, caused truncated answers mid-sentence
  2. num_ctx raised to 3072     — was 2048, caused context overflow on large plannings
  3. SQL formatter completely rewritten — structured tables instead of flat key-value
     dumps, so Mistral cannot confuse machines / commandes / numbers with each other
  4. EXPERT_SYSTEM_PROMPT hardened — explicit "ONLY use the numbers below, never invent"
  5. Hallucination guard added — if SQL data is empty, tell Mistral to say so rather
     than fabricate generic advice
  6. Query E row cap raised to 10 (was 5) — enough context for sequencing questions
  7. Row caps for B, C, D raised to 20 — more data visible to Mistral
  8. FAISS bypass for factual lookup questions — makespan, status, commande count, etc.
     are answered 100% from Query A; FAISS retrieval only pollutes these answers.
     Detect "factual" questions and set vector_context = "" to skip FAISS entirely.
  9. SQL-FIRST assertion in user prompt strengthened — SQL block is now preceded by
     an explicit "GROUND TRUTH" label and the domain block is moved after it with a
     clear instruction to never let domain knowledge override SQL numbers.

Architecture (unchanged):
  • Embeddings  : nomic-embed-text  (via Ollama)
  • Vector DB   : FAISS (in-process, persisted to disk)
  • LLM         : Mistral 7B  (via Ollama)
  • SQL Queries : 6 diagnostic queries (A–F) pre-fetched by .NET ChatController
  • Retriever   : Hybrid — FAISS semantic + SQL structured data

TIMEOUT CHAIN (must be consistent across all layers):
  rag_engine   httpx timeout : 10 min  (this file)
  .NET HttpClient timeout    : 12 min  (Program.cs — includes SQL + network overhead)
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
TOP_K             = 3
EMBED_DIM         = 768

# ── LLM generation options ──────────────────────────────────────────────────
# Two profiles: full (first question or complex) and short (follow-up / one-word).
# On CPU: full = ~3-6 min, short = ~1-2 min.
LLM_OPTIONS_FULL = {
    "num_predict": 400,    # ↓ was 600 — 600 caused 5+ min waits for simple answers
    "num_ctx":     2048,   # ↓ was 3072 — our formatted prompt fits in 2048 comfortably
    "temperature": 0.1,
    "top_p":       0.9,
}
LLM_OPTIONS_SHORT = {
    "num_predict": 150,    # Follow-ups like "pourquoi?" need at most 5-6 sentences
    "num_ctx":     1536,
    "temperature": 0.1,
    "top_p":       0.9,
}

# Keep backward-compat alias used in health endpoint
LLM_OPTIONS = LLM_OPTIONS_FULL

OLLAMA_TIMEOUT = 600  # 10 minutes

# Keywords that indicate the user wants Query E (sequencing detail).
SEQUENCE_KEYWORDS = [
    "séquence", "sequence", "séquencement",
    "ordre", "order",
    "startpm", "endpm", "start_pm", "end_pm",
    "lot", "lotidx", "loti",
    "heure", "hour", "timing",
    "quand", "when", "planifié",
    "opération", "operation",
]

# FIX #8: Keywords that indicate a pure factual lookup answerable entirely
# from Query A (planning summary). For these questions, FAISS retrieval is
# skipped because:
#   • The answer is already present verbatim in Query A.
#   • Domain chunks (e.g. "1440 min = 1 jour") contain numbers that Mistral
#     can confuse with the actual planning's makespan / duration.
FACTUAL_KEYWORDS = [
    # duration / makespan
    "combien de temps", "durée", "duree", "makespan", "combien d'heure",
    "combien d'heures", "how long", "duration",
    # status
    "statut", "status", "optimal", "faisable", "feasible",
    # count
    "combien de commande", "nombre de commande", "how many order",
    "combien de ligne", "nombre de ligne", "how many line",
    # dates
    "date de début", "date debut", "date de generation", "généré le",
    "start date", "generated on",
    # simple identity
    "quel est le planning", "what is the planning",
    "résumé du planning", "summary of the planning",
]

# FIX #7: raised caps so Mistral sees more real data rows
ROW_CAPS: dict[str, int] = {
    "A": 10,   # planning summary — always tiny
    "B": 20,   # ↑ was 15 — late orders
    "C": 20,   # ↑ was 15 — machine utilisation
    "D": 20,   # ↑ was 15 — lot fragmentation
    "E": 10,   # ↑ was 5  — sequencing detail
    "F": 15,   # unused machines
}

# ---------------------------------------------------------------------------
# Domain knowledge chunks (static — embedded once at startup)
# ---------------------------------------------------------------------------
DOMAIN_CHUNKS = [
    """MODÈLE DE TEMPS — Minutes productives
L'atelier tourne 24h/24, 7j/7. 1 journée = 1440 minutes productives.
Les heures de début et de fin de chaque opération sont exprimées en minutes
écoulées depuis le lancement du planning (jour 0).
Exemple : début à 4320 minutes = jour 3, 00h00.
Les week-ends sont exclus automatiquement du calendrier de livraison.""",

    """RÈGLE DES LOTS — Contrainte clé du planning
La taille effective d'un lot = min(taille de lot de la recette, capacité max de la machine, quantité de la commande).
Nombre de lots = quantité totale ÷ taille effective du lot (arrondi au supérieur).
Durée totale d'une opération = (chargement + cycle + déchargement) × nombre de lots.
POINT CLÉ : Une machine avec une petite capacité génère PLUS de lots → durée PLUS longue.
Si une commande a plus de 5 lots pour une opération, il faut chercher une machine avec une plus grande capacité.
Une machine plus grande réduit directement la durée totale du planning.""",

    """PRIORITÉ DU PLANNING — Ce qui compte le plus
Le solveur minimise d'abord les retards de livraison, ensuite la durée totale.
Poids des retards selon l'urgence : urgence 1 (critique) → poids 10, urgence 5 → poids 2, urgence 10 → poids 1.
RÈGLE ABSOLUE : Zéro retard sur une commande urgente est TOUJOURS préférable à un planning plus court avec une commande en retard.
Statut OPTIMAL = meilleure solution trouvée dans le temps imparti.
Statut FAISABLE = solution valide mais améliorable (limite de temps atteinte).""",

    """ORDRE DE PRIORITÉ DES COMMANDES
Les commandes sont planifiées dans cet ordre de priorité :
1. Urgence la plus haute (urgence 1 = le plus urgent)
2. Date de livraison la plus proche
3. Durée totale des opérations (les commandes longues passent en premier à durée égale)
Les commandes urgentes avec une livraison proche sont donc planifiées en tout premier.""",

    """AMÉLIORER UN PLANNING EN STATUT FAISABLE
Si le statut est "faisable" (pas encore optimal), il est possible d'améliorer le résultat :
- Augmenter le temps de résolution de 90s à 120-150s
- Augmenter le nombre d'itérations d'optimisation de 300 à 500
- Vérifier que toutes les machines disponibles sont bien incluses dans le calcul
- Utiliser le mode multi-machines (répartition des lots sur plusieurs machines du même type)
CONTRAINTE CP-SAT : Le solveur ne peut pas assigner deux lots d'une même opération sur la même machine simultanément.
CONTRAINTE CP-SAT : Les opérations d'une commande doivent respecter l'ordre de la recette.""",

    """CRITÈRES DE QUALITÉ D'UN PLANNING
Un bon planning présente :
- Zéro retard sur les commandes d'urgence 1, 2 ou 3
- Toutes les machines disponibles dont les opérations correspondent aux commandes sont utilisées
- Moins de 5 lots par opération par commande (si possible)

Problèmes à détecter :
- Machine disponible supportant les bonnes opérations mais NON UTILISÉE → opportunité de parallélisation
- Trop de lots (> 5) pour une seule machine → affecter à une machine avec plus de capacité
- Commande en retard sur une livraison urgente

IMPORTANT — taux de charge :
Le pourcentage de charge (minutes planifiées / makespan) n'est PAS un indicateur de surcharge.
Une machine à 100% sur un planning de 4 heures signifie qu'elle a travaillé sans interruption —
c'est un comportement normal et optimal, pas une surcharge.
Ne jamais diagnostiquer une "machine surchargée" à partir du seul taux de charge relatif au makespan.""",

    """RECOMMANDATIONS POUR LES PROBLÈMES COURANTS

PROBLÈME : Commande en retard
→ Vérifier le nombre de lots : si élevé, affecter à une machine avec plus de capacité
→ Vérifier si une machine disponible non utilisée peut prendre en charge l'opération bloquante (vérifier que ses opérations correspondent)
→ Relancer le planning avec un temps de résolution plus long (seulement si statut = Faisable)
→ Vérifier que la machine est bien en état "Fonctionnel"

PROBLÈME : Machine disponible non utilisée alors que ses opérations correspondent
→ Activer le mode multi-machines pour répartir les lots
→ La machine non utilisée peut réduire le makespan en prenant des lots en parallèle

PROBLÈME : Trop de lots sur une seule machine (> 5)
→ Affecter à une machine avec une plus grande capacité par lot
→ Activer le mode multi-machines pour distribuer les lots

PROBLÈME : Planning en statut Faisable (non Optimal)
→ Augmenter le temps de résolution à 120-150 secondes
→ Augmenter le nombre d'itérations d'optimisation à 500
→ Vérifier que toutes les machines fonctionnelles sont bien accessibles""",

    """STYLE DE RÉPONSE
- Aller droit au but. Pas d'introduction, pas de conclusion, pas de rappel de la question.
- Maximum 6 lignes. Utiliser des puces uniquement s'il y a 3 éléments distincts ou plus.
- Citer les chiffres exacts des données (%, jours, noms de machines, numéros de commandes).
- Zéro remplissage : pas de "Bien sûr", "En résumé", "Il convient de noter", "Il est important de".
- Langue de l'utilisateur (français ou anglais) — détecter automatiquement et s'y tenir.""",
]

# ---------------------------------------------------------------------------
# System prompt — concise, professional, grounded
# ---------------------------------------------------------------------------
EXPERT_SYSTEM_PROMPT = """Tu es un expert en planification industrielle pour l'atelier de lavage denim Micwic.

RÈGLES — sans exception :
1. Réponds directement à la question en 2 à 6 lignes. Jamais plus sauf si la question liste explicitement plusieurs points.
2. Zéro blabla : pas d'introduction, pas de conclusion, pas de "Bien sûr", pas de reformulation de la question.
3. Chiffres réels uniquement — ceux présents dans les données fournies. Ne jamais inventer ni extrapoler une valeur.
4. Si une information est absente des données, une seule phrase suffit : "Cette information n'est pas disponible pour ce planning."
5. Jamais de noms de colonnes SQL ni d'étiquettes internes (Requête A/B/C…, MakespanPM, TauxChargePct, NbLots…).
6. Réponds dans la langue exacte de l'utilisateur (français ou anglais — détecter et s'y tenir).
7. Recommandations : uniquement les machines et commandes visibles dans les données fournies.
8. Ne jamais suggérer de modifier l'urgence d'une commande pour améliorer le makespan — l'urgence est une donnée métier fixe, pas un levier du solveur.
9. STATUT OPTIMAL signifie que le solveur a prouvé mathématiquement qu'aucune meilleure solution n'existe. Ce n'est pas "dans le temps imparti" — c'est une preuve d'optimalité. STATUT FAISABLE = solution valide mais dont l'optimalité n'est pas prouvée.
10. Pour réduire le makespan, les seules actions valides sont : activer le mode multi-machines, affecter des lots à une machine non utilisée supportant la même opération, ou relancer avec un temps de résolution plus long. Ne jamais inventer d'autres solutions.
11. Si le statut du planning est OPTIMAL, ne jamais suggérer de relancer avec un temps de résolution plus long — c'est inutile, l'optimalité est déjà prouvée mathématiquement.
12. MAKESPAN EN HEURES : une durée affichée en heures (ex: "4h00") signifie que le planning se termine en moins d'une journée. Ce n'est PAS une anomalie — c'est normal pour un petit planning. Ne jamais traiter "moins d'une journée" comme un problème.
13. CHARGE DES MACHINES : les minutes planifiées indiquent la quantité de travail effectué, pas une surcharge. Une machine qui tourne 240 min sur un planning de 4h n'est pas surchargée — elle travaille en continu, ce qui est optimal. Ne jamais qualifier une machine de "surchargée" ou "sous-utilisée" à partir des seules minutes planifiées sans données de comparaison explicites.
14. MACHINES NON UTILISÉES : si une machine n'est pas utilisée dans un planning OPTIMAL, c'est parce que ses opérations supportées ne correspondent pas aux opérations requises par les commandes planifiées. Ne jamais recommander une machine non utilisée sans vérifier que ses opérations correspondent à celles du planning.
15. PRIORITÉ ABSOLUE DES DONNÉES : La section "VÉRITÉ TERRAIN SQL" contient les seules valeurs numériques autorisées. Les règles métier (section suivante) expliquent le contexte mais ne fournissent JAMAIS de chiffres à utiliser dans une réponse. Si un chiffre apparaît dans les règles métier (ex: 1440, 300, 500) et qu'il ne figure pas dans les données SQL, il est INTERDIT de l'utiliser dans la réponse."""

# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

async def embed(texts: List[str]) -> np.ndarray:
    """Call Ollama nomic-embed-text to embed a list of texts."""
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
# Startup: embed and index domain knowledge chunks
# ---------------------------------------------------------------------------

async def ensure_domain_knowledge_indexed():
    if _faiss_index.load():
        print(f"[RAG] Loaded FAISS index: {_faiss_index.index.ntotal} vectors")
        return

    print("[RAG] Building domain knowledge index...")
    try:
        vectors = await embed(DOMAIN_CHUNKS)
        metas   = [{"source": "domain", "idx": i} for i in range(len(DOMAIN_CHUNKS))]
        _faiss_index.add(vectors, DOMAIN_CHUNKS, metas)
        _faiss_index.save()
        print(f"[RAG] Indexed {len(DOMAIN_CHUNKS)} domain chunks")
    except Exception as e:
        print(f"[RAG] Failed to build index: {e}")


async def index_planning_rows(planning_id: int, planning_text: str):
    """Index a free-text summary of a planning into FAISS."""
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

def _needs_sequence_data(question: str) -> bool:
    q_lower = question.lower()
    return any(kw in q_lower for kw in SEQUENCE_KEYWORDS)


# FIX #8: Detect pure factual lookup questions where FAISS must be skipped.
# These questions are answered entirely by Query A; any domain chunk retrieved
# by FAISS (e.g. "1440 min = 1 jour") will only confuse Mistral.
def _is_factual_lookup(question: str) -> bool:
    """
    Returns True when the question is a factual lookup answerable from Query A alone.
    FAISS retrieval is suppressed for these questions to prevent domain chunk numbers
    (e.g. 1440, the minutes-per-day constant) from polluting the answer.
    """
    q_lower = question.lower().strip()
    return any(kw in q_lower for kw in FACTUAL_KEYWORDS)


# Short follow-up questions that need minimal token budget.
# "pourquoi", "why", "comment", single-word continuations, etc.
_SHORT_QUESTION_PATTERNS = re.compile(
    r"^\s*(pourquoi|why|comment|how|really|vraiment|et alors|so what"
    r"|explique|explain|détaille|detail|précise|clarif"
    r"|ok|oui|non|yes|no|sure|d'accord|agreed)\b",
    re.IGNORECASE,
)

def _is_short_followup(question: str) -> bool:
    """True when the question is a short follow-up that needs less token budget."""
    q = question.strip()
    if len(q) < 40:   # very short text — almost certainly a follow-up
        return True
    if _SHORT_QUESTION_PATTERNS.match(q):
        return True
    return False


# ---------------------------------------------------------------------------
# FIX #3: SQL formatter completely rewritten as structured, readable tables
# ---------------------------------------------------------------------------

def _fmt_makespan(days, minutes) -> str:
    """
    Human-readable makespan.
    When MakespanDays = 0 the planning finishes in under 1 day — express as hours
    so Mistral never reads '0 jours' and treats it as an anomaly.
    """
    try:
        d = int(days)
        m = int(minutes)
    except (TypeError, ValueError):
        return f"{days} jours ({minutes} minutes)"

    if d == 0:
        h   = m // 60
        rem = m % 60
        if rem:
            return f"{h}h{rem:02d} (moins d'une journée complète)"
        return f"{h}h00 (moins d'une journée complète)"
    return f"{d} jour(s) ({m} minutes)"


def _fmt_row_A(rows: list) -> str:
    """Planning summary — single row, formatted as a clear fact sheet."""
    if not rows:
        return "  (aucune donnée)\n"
    out = ""
    for r in rows:
        makespan_str = _fmt_makespan(r.get('MakespanDays'), r.get('MakespanPM'))
        out += f"  • Planning ID       : {r.get('Id', '?')}\n"
        out += f"  • Statut            : {r.get('Statut', '?')}\n"
        out += f"  • Généré le         : {r.get('DateGeneration', '?')}\n"
        out += f"  • Date de début     : {r.get('DateDebut', '?')}\n"
        out += f"  • Durée totale      : {makespan_str}\n"
        out += f"  • Commandes         : {r.get('NombreCommandes', '?')}\n"
        out += f"  • Lignes planifiées : {r.get('NombreLignes', '?')}\n"
    return out


def _fmt_row_B(rows: list, cap: int) -> str:
    """Late orders — each row on one clearly-labelled line."""
    if not rows:
        return "  ✅ Aucune commande en retard.\n"
    out = ""
    for r in rows[:cap]:
        cmd      = r.get('NumeroCommande', '?')
        urg      = r.get('Urgence', '?')
        deadline = r.get('Deadline', '?')
        fin      = r.get('FinPlanifiee', '?')
        retard   = r.get('JoursRetard', '?')
        recette  = r.get('NomRecette', '?')
        out += f"  🔴 Commande {cmd} (urgence {urg}, recette {recette})\n"
        out += f"     Date limite: {deadline} | Fin planifiée: {fin} | Retard: {retard} jour(s)\n"
    if len(rows) > cap:
        out += f"  ... ({len(rows) - cap} autres commandes en retard)\n"
    return out


def _fmt_row_C(rows: list, cap: int) -> str:
    """
    Machine utilisation — factual minutes and lot counts only.

    WHY TauxChargePct IS NOT SHOWN:
    The formula (MinutesPlanifiees / MakespanPM) is meaningless for short plannings.
    A machine running 4 lots × 60 min on a 4-hour planning gets 100% — not because
    it is overloaded, but because it ran at full speed on a short job.
    Showing that percentage to Mistral caused it to hallucinate "machine surchargée"
    recommendations on perfectly optimal plannings.
    We show raw minutes and lot counts instead; the LLM can reason correctly from those.
    """
    if not rows:
        return "  (aucune donnée)\n"
    out = ""
    for r in rows[:cap]:
        nom     = r.get('NomMachine', '?')
        capa    = r.get('CapaciteMax', '?')
        ops_raw = r.get('OperationsList') or r.get('Operations', '?')
        ops     = ', '.join(ops_raw) if isinstance(ops_raw, list) else str(ops_raw)
        nb_cmd  = r.get('NbCommandes', '?')
        mins    = r.get('MinutesPlanifiees', '?')
        nb_lots = r.get('NbLots') or r.get('NbLotsTotal', '?')

        # Derive a meaningful label without the broken % threshold:
        # Only flag UNUSED (0 minutes) or genuinely idle. Let Mistral read the raw
        # numbers for everything else — it handles factual comparisons fine.
        if mins in (0, '0', None, '?'):
            icon  = "⚪"
            label = "NON UTILISÉE"
        else:
            icon  = "🟢"
            label = "EN SERVICE"

        out += f"  {icon} {nom} (capacité {capa} pièces/lot | opérations: {ops}) — {label}\n"
        out += f"     Commandes affectées: {nb_cmd} | Minutes planifiées: {mins}"
        if nb_lots not in ('?', None):
            out += f" | Lots traités: {nb_lots}"
        out += "\n"

    if len(rows) > cap:
        out += f"  ... ({len(rows) - cap} autres machines)\n"
    return out


def _fmt_row_D(rows: list, cap: int) -> str:
    """Lot fragmentation — highlight high-lot operations."""
    if not rows:
        return "  ✅ Aucune fragmentation excessive détectée (tous NbLots ≤ 3).\n"
    out = ""
    for r in rows[:cap]:
        cmd     = r.get('NumeroCommande', '?')
        op      = r.get('NomOperation', '?')
        machine = r.get('MachineName', '?')
        nb_lots = r.get('NbLots', '?')
        lot_eff = r.get('LotSizeEffectif', '?')
        cap_m   = r.get('CapaciteMaxMachine', '?')
        lot_rec = r.get('LotSizeRecette', '?')
        qte_cmd = r.get('QuantiteCommande', '?')
        out += f"  🟠 Commande {cmd} — opération '{op}' sur {machine}\n"
        out += f"     Lots: {nb_lots} | Taille lot effective: {lot_eff} pièces | Capacité machine: {cap_m} | Taille lot recette: {lot_rec} | Quantité commande: {qte_cmd}\n"
    if len(rows) > cap:
        out += f"  ... ({len(rows) - cap} autres fragmentations)\n"
    return out


def _fmt_row_E(rows: list, cap: int) -> str:
    """Sequencing detail — compact but unambiguous."""
    if not rows:
        return "  (aucune donnée)\n"
    out = ""
    for r in rows[:cap]:
        cmd   = r.get('NumeroCommande', '?')
        op    = r.get('NomOperation', '?')
        ordre = r.get('Ordre', '?')
        mach  = r.get('MachineName', '?')
        ds    = r.get('DateStart', '?')
        de    = r.get('DateEnd', '?')
        lot   = r.get('LotIdx', '?')
        nb    = r.get('NbLots', '?')
        duree = r.get('DureeMinutes', '?')
        out += f"  {cmd} | op {ordre}:{op} | {mach} | lot {lot}/{nb} | {ds} → {de} | cycle {duree}min\n"
    if len(rows) > cap:
        out += f"  ... ({len(rows) - cap} autres lignes de séquencement)\n"
    return out


def _fmt_row_F(rows: list, cap: int) -> str:
    """Unused available machines."""
    if not rows:
        return "  ✅ Toutes les machines fonctionnelles sont utilisées dans ce planning.\n"
    out = ""
    for r in rows[:cap]:
        nom     = r.get('NomMachine', '?')
        capa    = r.get('CapaciteMax', '?')
        # OperationsList is a proper list sent by .NET; fall back to raw Operations string if absent
        ops_raw = r.get('OperationsList') or r.get('Operations', '?')
        ops     = ', '.join(ops_raw) if isinstance(ops_raw, list) else str(ops_raw)
        out += f"  ⚠️  {nom} — NON UTILISÉE | capacité: {capa} pièces/lot | opérations supportées: {ops}\n"
    if len(rows) > cap:
        out += f"  ... ({len(rows) - cap} autres machines non utilisées)\n"
    return out


async def run_sql_queries(planning_id: int, db_rows: dict, question: str = "") -> str:
    """
    FIX #3: Rewritten to produce clearly structured, readable sections instead of
    flat key=value dumps. Each query section uses a dedicated formatter that labels
    every field explicitly, making it impossible for Mistral to confuse machines,
    commandes or percentages with each other.
    """
    include_e = _needs_sequence_data(question)
    sections  = []

    # ── A — Planning summary ─────────────────────────────────────────────────
    sections.append(
        f"=== RÉSUMÉ DU PLANNING #{planning_id} ===\n"
        + _fmt_row_A(db_rows.get("A", []))
    )

    # ── B — Late orders ───────────────────────────────────────────────────────
    sections.append(
        "=== COMMANDES EN RETARD ===\n"
        + _fmt_row_B(db_rows.get("B", []), ROW_CAPS["B"])
    )

    # ── C — Machine utilisation ───────────────────────────────────────────────
    sections.append(
        "=== CHARGE ET UTILISATION DES MACHINES ===\n"
        + _fmt_row_C(db_rows.get("C", []), ROW_CAPS["C"])
    )

    # ── D — Lot fragmentation ─────────────────────────────────────────────────
    sections.append(
        "=== FRAGMENTATION DES LOTS (opérations avec plus de 3 lots) ===\n"
        + _fmt_row_D(db_rows.get("D", []), ROW_CAPS["D"])
    )

    # ── E — Sequencing (only when question asks for it) ───────────────────────
    if include_e:
        sections.append(
            "=== SÉQUENCEMENT DES OPÉRATIONS (extrait) ===\n"
            + _fmt_row_E(db_rows.get("E", []), ROW_CAPS["E"])
        )
    else:
        sections.append(
            "=== SÉQUENCEMENT DES OPÉRATIONS ===\n"
            "  (Omis — posez une question sur les heures ou l'ordre des opérations pour voir ce détail)\n"
        )

    # ── F — Unused machines ───────────────────────────────────────────────────
    sections.append(
        "=== MACHINES DISPONIBLES NON UTILISÉES ===\n"
        + _fmt_row_F(db_rows.get("F", []), ROW_CAPS["F"])
    )

    # ── FIX #5: Hallucination guard ───────────────────────────────────────────
    # If the .NET layer sent empty db_rows (network issue / auth error),
    # inject an explicit warning so Mistral says "no data" instead of fabricating.
    if not db_rows:
        return (
            "⚠️  AUCUNE DONNÉE SQL REÇUE POUR CE PLANNING.\n"
            "Les données de diagnostic n'ont pas pu être récupérées depuis la base de données.\n"
            "Le modèle ne peut pas analyser ce planning sans données réelles.\n"
            "Veuillez réessayer ou contacter l'administrateur si le problème persiste."
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
      1. Detect whether this is a pure factual lookup (FIX #8)
         → If yes, skip FAISS entirely (vector_context = "")
         → If no, embed the question and retrieve top-K domain chunks
      2. Format SQL data (A–F) as structured readable sections
      3. Assemble the enriched prompt with SQL labelled as "GROUND TRUTH"
         and domain knowledge clearly marked as secondary context only
      4. Call Mistral via Ollama
      5. Return the structured answer
    """
    # FIX #8: Skip FAISS for pure factual lookups.
    # Factual questions (makespan, status, commande count, dates) are answered
    # 100% from Query A. FAISS retrieval only risks surfacing domain chunk numbers
    # (e.g. "1440 minutes = 1 journée") that Mistral may anchor on instead of the
    # real SQL value.
    factual = _is_factual_lookup(question)

    if factual:
        vector_context = ""
        print(f"[RAG] Factual lookup detected — FAISS bypassed for: {question[:80]}")
    else:
        # Step 1: embed question
        try:
            q_vec = await embed([question])
        except Exception as e:
            print(f"[RAG] Embedding error: {e}")
            q_vec = np.zeros((1, EMBED_DIM), dtype=np.float32)

        # Step 2: retrieve relevant domain knowledge chunks
        retrieved = _faiss_index.search(q_vec[0], k=TOP_K)
        if retrieved:
            chunks_text    = [text[:300] for text, score, _ in retrieved]
            vector_context = "\n\n".join(chunks_text)
        else:
            vector_context = ""

    # Step 3: format SQL data
    sql_context = await run_sql_queries(planning_id, db_rows, question)

    # FIX #9: Prompt restructured so SQL is explicitly labelled "VÉRITÉ TERRAIN"
    # (ground truth) and domain knowledge is positioned as secondary context with
    # an explicit warning never to use its numbers in answers.
    #
    # When FAISS was bypassed (factual question), the domain block is omitted
    # entirely — there is no risk of confusion and the prompt is shorter/faster.
    if vector_context:
        domain_section = f"""
=== RÈGLES MÉTIER ET CONTRAINTES DU SOLVEUR (contexte secondaire) ===
ATTENTION : Cette section explique les règles de fonctionnement. Elle ne contient PAS de chiffres
réels du planning. Ne jamais utiliser un chiffre de cette section dans ta réponse — utilise
UNIQUEMENT les chiffres de la section "VÉRITÉ TERRAIN SQL" ci-dessus.
{vector_context}
"""
    else:
        domain_section = ""

    user_prompt = f"""=== VÉRITÉ TERRAIN SQL — Planning #{planning_id} ===
Ces données proviennent directement de la base de données. Ce sont les SEULES valeurs autorisées.
Ne jamais inventer, extrapoler ou remplacer ces valeurs par des chiffres génériques.

{sql_context}
{domain_section}
=== QUESTION DE L'UTILISATEUR ===
{question}

Réponds en te basant EXCLUSIVEMENT sur les données SQL ci-dessus. Si une information n'est pas présente dans ces données, dis-le clairement en une phrase."""

    messages = [
        {"role": "system", "content": EXPERT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    # Step 4: call Mistral with adaptive token budget
    options = LLM_OPTIONS_SHORT if _is_short_followup(question) else LLM_OPTIONS_FULL
    payload = {
        "model":      LLM_MODEL,
        "messages":   messages,
        "stream":     False,
        "keep_alive": "10m",
        "options":    options,
    }

    # Step 5: call Mistral
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        reply = r.json().get("message", {}).get("content", "")
        return reply
    except httpx.TimeoutException:
        return (
            "[RAG TIMEOUT] Mistral a mis trop de temps à répondre. "
            "Essayez une question plus courte ou plus ciblée. "
            "Pour de meilleures performances, vérifiez qu'Ollama tourne sur GPU."
        )
    except Exception as e:
        return f"[RAG ERROR] LLM call failed: {e}"