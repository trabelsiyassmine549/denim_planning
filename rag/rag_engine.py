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
# FIX #1: num_predict raised from 350 → 600.  350 was cutting answers mid-sentence.
# FIX #2: num_ctx raised from 2048 → 3072.  Needed for large plannings with all 6 queries.
# On CPU: expect ~3-6 min. On GPU: ~30-60s.
LLM_OPTIONS = {
    "num_predict": 600,    # ↑ was 350 — too low caused truncated answers
    "num_ctx":     3072,   # ↑ was 2048 — needed for full SQL context
    "temperature": 0.1,    # ↓ was 0.2 — lower = more deterministic, fewer hallucinations
    "top_p":       0.9,
}

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
- Taux de charge des machines inférieur à 85%
- Moins de 5 lots par opération par commande
- Toutes les machines disponibles utilisées

Problèmes à détecter :
- Machine SURCHARGÉE : taux de charge > 90% → besoin de redistribution
- Machine SOUS-UTILISÉE : taux de charge < 30% → opportunité de parallélisation manquée
- Trop de lots (> 5) : assigner à une machine avec plus de capacité
- Machine disponible mais non utilisée : peut débloquer une opération en retard""",

    """RECOMMANDATIONS POUR LES PROBLÈMES COURANTS

PROBLÈME : Commande en retard
→ Vérifier le nombre de lots : si élevé, affecter à une machine avec plus de capacité
→ Vérifier si une machine disponible non utilisée peut prendre en charge l'opération bloquante
→ Relancer le planning avec un temps de résolution plus long
→ Vérifier que la machine est bien en état "Fonctionnel"

PROBLÈME : Machine surchargée (> 90% de charge)
→ Redistribuer des commandes vers des machines sous-utilisées du même type d'opération
→ Activer le mode multi-machines (répartition des lots sur 2 ou 3 machines)
→ La machine surchargée est le goulot d'étranglement — vérifier s'il existe une machine parallèle

PROBLÈME : Machine sous-utilisée (< 30% de charge)
→ Les opérations de cette machine pourraient être traitées plus vite
→ Possibilité de paralléliser avec le mode multi-machines
→ Augmenter la capacité de la machine ou ajuster la taille de lot de la recette

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
10. Pour réduire le makespan, les seules actions valides sont : activer le mode multi-machines, affecter des lots à une machine non utilisée supportant la même opération, ou relancer avec un temps de résolution plus long. Ne jamais inventer d'autres solutions."""

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


# ---------------------------------------------------------------------------
# FIX #3: SQL formatter completely rewritten as structured, readable tables
# ---------------------------------------------------------------------------

def _fmt_row_A(rows: list) -> str:
    """Planning summary — single row, formatted as a clear fact sheet."""
    if not rows:
        return "  (aucune donnée)\n"
    out = ""
    for r in rows:
        out += f"  • Planning ID       : {r.get('Id', '?')}\n"
        out += f"  • Statut            : {r.get('Statut', '?')}\n"
        out += f"  • Généré le         : {r.get('DateGeneration', '?')}\n"
        out += f"  • Date de début     : {r.get('DateDebut', '?')}\n"
        out += f"  • Durée totale      : {r.get('MakespanDays', '?')} jours ({r.get('MakespanPM', '?')} minutes)\n"
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
    """Machine utilisation — clear table with charge state."""
    if not rows:
        return "  (aucune donnée)\n"
    out = ""
    for r in rows[:cap]:
        nom     = r.get('NomMachine', '?')
        capa    = r.get('CapaciteMax', '?')
        ops     = r.get('Operations', '?')
        nb_cmd  = r.get('NbCommandes', '?')
        mins    = r.get('MinutesPlanifiees', '?')
        taux    = r.get('TauxChargePct', '?')
        etat    = r.get('Etat', '?')
        icon    = "🔴" if etat == "SURCHARGE" else ("🟡" if etat == "SOUS-UTILISEE" else "🟢")
        taux_str = f"{round(float(taux), 1)}%" if taux not in ('?', None) else '?'
        out += f"  {icon} {nom} (capacité {capa} pièces/lot | opérations: {ops})\n"
        out += f"     Commandes: {nb_cmd} | Minutes planifiées: {mins} | Taux de charge: {taux_str} | État: {etat}\n"
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
        nom  = r.get('NomMachine', '?')
        capa = r.get('CapaciteMax', '?')
        ops  = r.get('Operations', '?')
        out += f"  ⚠️  {nom} (capacité {capa} pièces/lot | opérations supportées: {ops}) — NON UTILISÉE\n"
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
      1. Embed the question
      2. Retrieve top-K relevant domain chunks from FAISS (TOP_K=3)
      3. Format SQL data (A–F) as structured readable sections
      4. Assemble the enriched prompt
      5. Call Mistral via Ollama (num_predict=600, num_ctx=3072)
      6. Return the structured answer
    """
    # Step 1: embed question
    try:
        q_vec = await embed([question])
    except Exception as e:
        print(f"[RAG] Embedding error: {e}")
        q_vec = np.zeros((1, EMBED_DIM), dtype=np.float32)

    # Step 2: retrieve relevant domain knowledge chunks
    retrieved     = _faiss_index.search(q_vec[0], k=TOP_K)
    vector_context = ""
    if retrieved:
        chunks_text    = [text[:300] for text, score, _ in retrieved]
        vector_context = "\n\n".join(chunks_text)

    # Step 3: format SQL data
    sql_context = await run_sql_queries(planning_id, db_rows, question)

    # Step 4: assemble prompt
    # IMPORTANT: SQL data comes FIRST so Mistral focuses on it before
    # reading domain knowledge. This reduces the chance of generic answers.
    user_prompt = f"""=== DONNÉES RÉELLES DE LA BASE DE DONNÉES (Planning #{planning_id}) ===
IMPORTANT : Utilise UNIQUEMENT les chiffres et valeurs ci-dessous. Ne jamais inventer de valeurs.

{sql_context}

=== RÈGLES MÉTIER ET CONTRAINTES DU SOLVEUR ===
{vector_context}

=== QUESTION DE L'UTILISATEUR ===
{question}

Réponds en te basant EXCLUSIVEMENT sur les données ci-dessus. Si une information n'est pas présente, dis-le clairement."""

    messages = [
        {"role": "system", "content": EXPERT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]

    payload = {
        "model":      LLM_MODEL,
        "messages":   messages,
        "stream":     False,
        "keep_alive": "10m",
        "options":    LLM_OPTIONS,
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