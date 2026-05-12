"""
rag/rag_engine.py  -  RAG Engine v5.12 - Validator extended: recipe paraphrases + hardware capacity check
================================================================================================

v5.12 CHANGES (over v5.11):
  FIX-v5.12: _validate_llm_output_v2() extended with new violations caught.

  OBSERVED VIOLATIONS in v5.11 output (Query F empty → Mistral ran, validation passed):
    Point 1: "Ajustez les recettes de Poudre pour minimiser le temps de setup overhead"
             → Two violations: "ajustez les recettes" (recipe modification) and
               "minimiser le temps de setup" (setup duration modification).
               Neither phrase was in Check-3's exact match list.
    Point 2: "Augmentez la capacité de production des machines Brongo 1 et Brongo 5"
             → Machine physical capacity is a hardware constant, not a scheduling lever.
               No existing check covered this category.
    Point 3: "Examinez les délais de chargement et de déchargement... améliorez-les"
             → "améliorez les délais de chargement" = loading time modification.
               Not in Check-3's exact match list.

  Check-3 extensions (exact phrase blocklist):
    - "ajustez/ajuster les recettes", "ajustez/ajuster la recette"  (recipe paraphrase)
    - "minimiser le/les temps de setup", "minimiser le setup"       (setup time paraphrase)
    - "améliorez/améliorer les délais de chargement/déchargement"   (loading time paraphrase)
    - "examinez/examiner les délais de chargement"                  (examination + improvement)
    - "improve loading/unloading time", "improve the loading/unloading" (EN equivalents)

  Check-7 (new): Hardware capacity modification.
    Regex: recommendation verb (.{0,40}) capacité/throughput/output rate.
    Catches: "augmentez la capacité de production", "increase machine throughput", etc.
    NOT caught (false positive guard): data references like "la capacité de la machine est 100".
    Only fires when a recommendation verb precedes "capacité" within 40 characters.

v5.11 CHANGES (over v5.10):
  FIX-v5.11-A: _reason_with_mistral() — early exit when incompatible_unused is non-empty.

  OBSERVATION (v5.9 + v5.10 logs):
    When Tupesa 1/2 (or any MACHINE_INCOMPATIBLE) appear in the data table,
    Mistral ALWAYS recommends them despite the MACHINES_INTERDITES fence block.
    Check-6 fires on every attempt-1 and attempt-2 is usually too short (86s)
    to complete, causing a guaranteed ~280s round-trip before reaching the
    deterministic fallback.

  ROOT CAUSE:
    A 7B model on CPU cannot reliably suppress named entities that appear in
    the input. "MACHINE_INCOMPATIBLE nom=Tupesa 2 -> NE PAS RECOMMANDER" is
    processed as a machine mention, not a hard constraint. Temperature and
    greedy decoding do not fix attention-level entity suppression.

  FIX:
    If incompatible_str is non-empty, skip both LLM attempts entirely and
    return _deterministic_fallback(context, lang) immediately (<1ms).
    The deterministic fallback is now binding-constraint-aware (v5.10) and
    produces correct, quantified, solver-coherent output.
    The LLM path is preserved for the case where no incompatible machines
    exist (incompatible_unused=[]), where Check-6 cannot fire and Mistral
    generates better-phrased output than the fallback.

  FIX-v5.11-B: _reason_with_mistral() system prompt — setup lever instruction
    corrected to be binding-constraint-aware (was: always "machine de plus
    grande capacité", now: check contrainte_liante from FRAGMENTATION data).
    Previously the system prompt contradicted the v5.10 data table output,
    causing Mistral to recommend the wrong lever even when it followed the rules.

  NET EFFECT on this planning (Tupesa 1/2 incompatible):
    Before v5.11: ~280s total (attempt-1 193s + attempt-2 timeout)
    After v5.11:  <1ms (direct deterministic fallback)

v5.10 CHANGES (over v5.9):
  FIX-v5.10: Fragmentation lever is now binding-constraint-aware.

  ROOT CAUSE OF FIXED BUG (reported after v5.9 fallback output):
    Points 2 & 3 in the deterministic fallback both recommended
    "affecter à une machine de plus grande capacité" for Poudre on Brongo 2 and Brongo 4.
    This lever is only valid when machine.CapaciteMax < op.LotSizeRecette (binding=machine).
    When cap_machine >= lot_size_recette (binding=recette), the lot count is driven by the
    recipe lot size, not machine capacity. Moving to a bigger machine changes nothing.
    The correct lever in that case is parallelization across more compatible machines.

  CHANGES:
  1. _extract_fragmentation_facts(): now reads CapaciteMaxMachine and LotSizeRecette
     (using LotSizeRecette column name as primary, QuantiteLot as fallback) and stores:
       cap_machine:         int — machine capacity from Query D
       binding_constraint:  "machine" | "recette"
         "machine" → cap_machine < lot_size_recette → higher-cap machine IS a valid lever
         "recette" → cap_machine >= lot_size_recette → recipe lot size is the bottleneck,
                     machine-capacity upgrade does nothing; only lever is parallelization.

  2. _build_data_table(): FRAGMENTATION lines now include:
       cap_machine=<value>  contrainte_liante=<machine|recette>
       levier=machine-plus-grande-capacite-valide  (when binding=machine)
       levier=machine-capacite-INVALIDE(recette-fixe-nb-lots) levier-valide=paralleliser  (when binding=recette)
     This gives Mistral the ground truth needed to choose the correct lever.

  3. _deterministic_fallback(): setup overhead bullets now emit the correct lever:
       binding="machine" → "affecter à une machine de plus grande capacité"
       binding="recette" → "paralléliser sur plusieurs machines compatibles" (machine-cap lever NOT mentioned)
     Uses a (op, machine) lookup table built from context["fragmentation"] binding_constraint fields.

  4. _build_improvement_prompt(): FOCUS FRAGMENTATION instruction extended to instruct
     Mistral to check contrainte_liante before recommending a machine-capacity upgrade.

v5.9 CHANGES (over v5.8):
  FIX-v5.9-A: Check-3 (forbidden actions) extended to cover loading/unloading time reduction.
    The previous list only caught "réduire le temps de traitement" and recipe-change phrases.
    HARD RULE 1 also forbids recommending changes to TempsChargementMinutes and
    TempsDecharementMinutes (both fixed by industrial recipe specialists).
    New forbidden phrases added (FR + EN):
      "réduire le/les temps de chargement/déchargement",
      "reduce loading/unloading time", "shorten loading/unloading time",
      "reduce setup time", "réduire le setup",
      "optimiser le/les temps de chargement"
    Still NOT blocked (valid levers): "réduire les temps morts", "réduire les temps
    d'attente", "réduire la fragmentation", "réduire le nombre de lots".

  FIX-v5.9-B: Check-6 (incompatible machine recommendation) verb list extended to include
    maintenance and repair verbs. The previous list only covered scheduling verbs
    (activer, utiliser, affecter, add, redirect, move, enable). Mistral bypassed the
    check by using "réparer", "remplacer", "mettre en service", "remettre en marche" etc.
    These verbs are now included because MACHINE_INCOMPATIBLE machines must never be
    recommended regardless of mechanical state — their OperationsList doesn't include the
    planned operations, so repairing them changes nothing about scheduling compatibility.
    New FR verbs: réparer, remplacer, remettre, mettre en service, remettre en marche,
                  remettre en état, réactiver
    New EN verbs: repair, replace, restore, service, put back, bring back,
                  recommission, reintroduce, reactivate

  FIX-v5.9-C: _build_improvement_prompt() system prompt hardened with two explicit
    prohibition lines injected before the focus directive:
      "MACHINES INTERDIT: ne jamais recommander d'activer, utiliser, réparer ou remplacer les machines INTERDIT."
      "DURÉES RECETTE FIXES: ne jamais recommander de réduire/modifier les temps de chargement, déchargement ou traitement."
    These were previously only in CPSAT_RULES_PROMPT (not injected in the compact CPU prompt)
    and in the user_prompt constraints block, but Mistral ignored them. Making them the
    first lines of the system prompt (before the focus directive) increases compliance.

ROOT CAUSE OF FIXED BUG:
  Question "comment je peut améliorer ce planning?" produced:
    Point 1: "Réduire le temps de chargement et déchargement" — violated HARD RULE 1.
             Check-3 missed it because "chargement/déchargement" was not in the forbidden list.
    Point 2: "Les machines Tupesa 2 et Tupesa 1 ... Vous pouvez les réparer ou les remplacer"
             — recommended forbidden machines via repair/replace verbs.
             Check-6 missed them because "réparer" and "remplacer" were not in recommend_verbs.
  Both violations caused `[VALIDATION] LLM output passed all checks.` to print incorrectly.
  With v5.9: both violations now trigger Check-3 and Check-6 respectively, causing the
  validator to return _deterministic_fallback(context, lang) — a safe, grounded response.


v5.8 CHANGES (over v5.7):
  ARCH-CHANGE: Improvement path redesigned so Mistral reasons from scratch.
    - _build_grounded_analysis() renamed to _build_data_table().
      Output changed from formatted prose bullets to raw key=value structured facts.
      Python no longer writes the final answer - it writes the data table.
    - _rewrite_with_mistral() renamed to _reason_with_mistral(context, question).
      System prompt changed from "here are bullets, produce final analysis"
      to "here are raw facts, identify issues, reason, compose your own analysis".
      Mistral now does the intellectual work: prioritisation, framing, phrasing.
    - temperature raised 0.1->0.3 to encourage varied, non-repetitive answers.
    - num_ctx reduced 2048->1024 (data_table is more compact than prose bullets).
    - Fallback: _reason_with_mistral() falls back to _deterministic_fallback(context)
      (not the old data_table string) so the fallback is always human-readable.
  WHY: Pre-formatted prose bullets fed as "user message" caused Mistral to copy them
  verbatim, regardless of the rewrite instruction. Raw data tables force Mistral to
  genuinely compose its response, producing varied and creative (yet grounded) output.

v5.7 FIXES (over v5.6):
  FIX-23: _classify_question() -- removed "analyse"/"analysis"/"analyze" from
          IMPROVEMENT_KEYWORDS.  These bare substrings matched "analyser la charge
          machine" and misrouted it to the improvement path, bypassing
          MACHINE_LOAD_PATTERNS entirely.  The C# ChatController already handles
          "analyser" correctly via _improvementRx and fetches the right SQL tier;
          Python's classifier must not override MACHINE_LOAD_PATTERNS.

  FIX-23: MACHINE_LOAD_PATTERNS strengthened - added patterns for
          "nom du/de la machine", "machine la plus surchargée", "machine occupée"
          to catch phrasing variants not covered by the previous regex.

  FIX-23: _answer_machine_load_deterministically() - new function.
          Machine-load questions now bypass Mistral entirely.  Python reads
          Query C rows (Etat, TauxChargePct, MinutesPlanifiees, NbCommandes) and
          Query F rows (unused machines) and produces a fully grounded answer
          without any LLM call.  Benefits: zero timeout risk, zero hallucination,
          instant response.  The Etat field (SURCHARGE / SOUS-UTILISEE / NOMINAL)
          is computed by the SQL query and is authoritative.

  FIX-23: OLLAMA_TIMEOUT raised 180->300s for analysis/lookup paths.
          LLM_OPTIONS_ANALYSIS num_predict reduced 400->120, num_ctx 1536→1024.
          At ~2s/token CPU, 400 tokens ≈ 800s -> guaranteed timeout.
          120 tokens ≈ 240s -> safe under 300s for typical CPU setups.



v5.6 FIXES (over v5.5):
  FIX-22: _rewrite_with_mistral() -> _rewrite_with_mistral() repurposed as constrained analyst.
          The function is renamed semantically: Mistral is no longer a "rewriter" but a
          "constrained planning analyst". The system prompt is replaced with the full
          industrial constraint ruleset (CP-SAT rules, forbidden suggestions, allowed
          analysis types, response style). Python still builds 100% accurate grounded
          bullets from SQL - these become the DATA input for Mistral, not something it
          reasons from scratch. Mistral produces the final voiced answer under hard rules.
          Fallback to grounded text unchanged if LLM fails or reply is too short (<50%).
          num_ctx bumped from 2048 consideration: system prompt is now ~600 tokens,
          grounded input ~260 tokens - total ~860 input tokens, fits in 2048 safely.

v5.5 FIXES (over v5.4):
  FIX-21: _rewrite_with_mistral() cutoff fix.
          num_predict 250->450 - grounded analysis is ~262 output tokens; 250 caused
          truncation mid-sentence ("L'activation permet de par...").
          num_ctx 1024->2048 - grounded input (~1037 chars) + system + user prompt
          did not fit in 1024 tokens, causing silent prefill truncation.
          timeout 120->300s - at ~2s/token CPU, 450 tokens requires up to ~300s.
          OLLAMA_TIMEOUT_IMPROVEMENT also updated 120->300s to match.
          Program.cs HttpClient timeout is 5 minutes - Python now always times out
          first so the error message stays in French, not a raw TCP reset.

v5.4 FIXES (over v5.3):
  FIX-16: Aggressive token budget reduction to fix timeout on CPU-only hardware.
          num_predict 150->100 (~200s worst case, 100s margin before 300s timeout).
          num_ctx 1024->768 (prompt fits comfortably; frees prefill time).
          facts cap 12->8 lines (~50 input tokens saved).
          RAG chunk removed from user_prompt (~40 input tokens saved, minimal value).
          System prompt and focus instructions compressed to single-line form.
          Total estimated prompt: ~280-320 input tokens, well within 768 ctx window.
          - OPERATIONS_PLANIFIEES: the exact op names Mistral may cite
          - MACHINES_ACTIVABLES: compatible unused machines + which ops they cover
          - MACHINES_INTERDITES: incompatible machines Mistral must never recommend
          Previously these whitelists were only described in the system prompt as
          prose instructions - Mistral ignored them and hallucinated ops (Rinçage)
          and recommended forbidden machines (Tupesa 2). Injecting them as DATA in
          the user_prompt forces Mistral to treat them as ground truth.
  FIX-15: Check-6 re-enabled as a smarter targeted check. Instead of naive
          denim-op text matching (old disabled approach), the new Check-6 detects
          when Mistral uses a recommendation verb (activer, utiliser, affecter...)
          near an incompatible machine name. This catches "activer Tupesa 2 pour
          Rinçage" without firing on legitimate compatibility explanations like
          "Tupesa 2 ne peut pas traiter Rinçage".
  FIX-12: _build_improvement_prompt() is now question-aware. A FOCUS directive
          is injected into the system prompt based on what the user is asking:
          - makespan/reduce/shorten -> FOCUS makespan levers only, no delay talk
          - retard/delay            -> FOCUS delay recovery actions
          - fragmentation/lot/batch -> FOCUS lot fragmentation & capacity upgrade
          - machine/charge/load     -> FOCUS load balancing with thresholds
          - anything else           -> top-3 highest-impact actions (global)
          This ensures "comment améliorer" and "comment réduire le makespan"
          return meaningfully different, focused answers from Mistral.

  FIX-13: Check-3 (forbidden action) forbidden phrase list tightened.
          Old list included "réduire le temps de traitement" but also caught
          "réduire les temps de chargement" (valid: loading/setup time IS a
          legitimate optimization lever). New list only matches exact recipe
          treatment duration phrases, not setup/loading/unloading time reductions.
          This eliminates false-positive Check-3 violations that caused good
          Mistral answers to fall back to deterministic mode.

ARCHITECTURE: SQL -> Fact Extraction → Mistral (primary reasoning) → Post-LLM Validation → Fallback

Previous v4.3 used deterministic Python analyzers to produce final-answer prose
that Mistral only formatted (or skipped entirely). v5.0 makes Mistral the primary
industrial reasoning engine:

  SQL A-F (sanitised, capped)
    + FAISS domain knowledge retrieval
    + CP-SAT hard rules (injected as system constraints)
    + whitelist facts (entity names, KPIs, capacity values)
    ↓
  Mistral - PRIMARY industrial reasoning engine
    ↓
  Python post-LLM validator
    (entity hallucination guard, forbidden-action guard,
     numeric plausibility guard, recipe-change guard, optimal-rerun guard)
    ↓
  Fallback to deterministic bullets on validation failure or timeout

Python's role: sanitise -> extract whitelists → build prompt → validate output → fallback if needed.

v5.2 FIXES (over v5.1):
  FIX-10: num_predict reduced 400->250 for improvement path. Streaming per-token
          mitigates timeout edge cases on CPU-only hardware. Keeps token generation
          well under 300s budget even with overhead.
  FIX-11: Check-6 (operation hallucination) disabled for improvement path.
          Mistral legitimately mentions op names in compatibility reasoning
          ("this machine doesn't do rinsing"), and naive text-match validator
          can't distinguish context from hallucination. Removing Check-6 avoids
          false positives that trigger fallback on CPU machines.

v5.1 FIXES (over v5.0):
  FIX-1: _validate_numeric_claims - skip calendar year numbers (2000-2100) to avoid
          false-positive violations when Mistral includes dates in its response.
  FIX-2: _validate_operations_in_reply - guard returns empty list when known_ops is
          empty (Query E missing NomOperation column) to prevent false OP_HALLUCINATION.
  FIX-3: OLLAMA_TIMEOUT raised to 300s for improvement path; _call_ollama_improvement()
          helper uses a dedicated 300s timeout so CPU-only machines don't silently
          time out and fall back to deterministic mode.
  FIX-4: _validate_llm_output_v2 now logs which specific check fired before returning
          the fallback, so the root cause is visible in FastAPI console.
  FIX-5: LLM_OPTIONS_IMPROVEMENT num_ctx reduced 3000->2048, num_predict 800→600 to
          keep total generation time under 300s on CPU-only hardware.
  FIX-6: Check 2 (active-machine-called-inactive) now de-duplicates violations per
          machine so one bad sentence doesn't multiply violation count.

PRESERVED FROM v4.x (do not modify):
  _assert_db_rows_safe(), _build_known_machines(), _build_known_commands(),
  _build_known_operations(), _build_active_machines(), _compute_makespan_real(),
  _classify_question(), _detect_language(), _is_gibberish(), _is_out_of_scope(),
  _minutes_to_hhmm(), _get_machine_name(), _safe_int()
  All non-improvement paths in analyze() are unchanged.

CHANGED FROM v4.x:
  - Deleted: FactLine dataclass, _analyze_delays/machines/fragmentation/
             bottlenecks/setup_overhead/status, _run_all_analyzers,
             _validate_llm_output (old), SKIP_FORMATTER_LLM path
  - Added:   _extract_*_facts() family, _build_improvement_context(),
             _build_improvement_prompt(), _validate_llm_output_v2(),
             _deterministic_fallback(), _build_machine_mention_pattern(),
             _validate_operations_in_reply(), _extract_numeric_bounds(),
             _validate_numeric_claims(), _retrieve_rag_chunks(),
             _call_ollama_improvement(),
             CPSAT_RULES_PROMPT, LLM_OPTIONS_IMPROVEMENT
  - Replaced: improvement branch in analyze()
"""

import json
import math
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
PPD               = 1440        # 1 PM = 1 real minute, 1 day = 1440 min
MINUTES_PER_PM    = 1

OLLAMA_TIMEOUT             = 300   # FIX-23: 180->300s - analysis path with num_predict=120 needs ~240s on CPU worst-case
OLLAMA_TIMEOUT_IMPROVEMENT = 300   # FIX-21: rewriter needs up to ~450 tokens -> ~300s worst-case on CPU
FORMATTER_TIMEOUT          = 60    # kept for compatibility

# FIX-C: setup overhead ratio threshold - 40% non-productive = significant
SETUP_RATIO_THRESHOLD = 0.40

# Security
_SQL_DANGER = re.compile(
    r'\b(DROP|DELETE|UPDATE|INSERT|TRUNCATE|ALTER|EXEC|EXECUTE|xp_|sp_|UNION)\b',
    re.IGNORECASE,
)

ROW_CAPS: Dict[str, int] = {"A": 10, "B": 20, "C": 20, "D": 20, "E": 60, "F": 15}

# ---------------------------------------------------------------------------
# Hard industrial rules - injected into every LLM prompt
# ---------------------------------------------------------------------------

HARD_RULES_FR = """\
═══════════════════════════════════════════════════════════
RÈGLES INDUSTRIELLES ABSOLUES - NE JAMAIS VIOLER :
1. Les durées de recette (traitement, chargement, déchargement) sont FIXÉES
   par des spécialistes industriels. Ne JAMAIS recommander de les modifier.
2. L'optimisation porte UNIQUEMENT sur : séquencement, allocation machines,
   distribution des lots, équilibrage, activation/désactivation machines.
3. Une machine présente dans la liste des machines actives (données Query C)
   EST UTILISÉE dans ce planning. Ne JAMAIS dire qu'elle est inactive ou
   non activée si elle figure dans cette liste.
4. Toute recommandation doit être chiffrée (minutes gagnées, % de charge).
   Une recommandation sans KPI n'a pas de valeur industrielle.
═══════════════════════════════════════════════════════════
"""

HARD_RULES_EN = """\
═══════════════════════════════════════════════════════════
ABSOLUTE INDUSTRIAL RULES - NEVER VIOLATE:
1. Recipe durations (treatment, loading, unloading) are FIXED by industrial
   specialists. NEVER recommend changing them.
2. Optimization applies ONLY to: sequencing, machine allocation, lot
   distribution, load balancing, machine activation/deactivation.
3. A machine present in the active machine list (Query C data) IS USED in
   this planning. NEVER say it is inactive or not activated if it appears
   in that list.
4. Every recommendation must be quantified (minutes saved, % utilisation
   change). Unquantified advice has no industrial value.
═══════════════════════════════════════════════════════════
"""

# ---------------------------------------------------------------------------
# CP-SAT rules - injected as declarative constraints into the system prompt
# ---------------------------------------------------------------------------

CPSAT_RULES_PROMPT = """\
═══════════════════════════════════════════════════════════
LOGIQUE CP-SAT - CONTRAINTES INDUSTRIELLES
═══════════════════════════════════════════════════════════

RÈGLES DE RECETTE (jamais négociables) :
- DureeMinutes, TempsChargementMinutes, TempsDecharementMinutes sont fixés
  par des spécialistes industriels. Le solveur ne peut pas les modifier.
  Vous non plus. Ne jamais les recommander comme levier d'amélioration.

LEVIERS D'OPTIMISATION AUTORISÉS :
1. Allocation machines : réaffecter des lots à des machines compatibles sous-utilisées.
2. Activation de machines non utilisées : machines fonctionnelles dans la liste
   des machines non utilisées peuvent être activées si leur OperationsList
   contient l'opération planifiée.
3. Parallélisation multi-machines : plusieurs machines compatibles en parallèle
   réduisent le makespan proportionnellement.
4. Réduction de la fragmentation : un lot plus grand (machine à plus grande capacité)
   réduit le nombre de lots -> moins de setups → makespan plus court.
5. Équilibrage de charge : redistribuer des lots d'une machine surchargée (>85%)
   vers une machine sous-utilisée (<40%) avec opérations compatibles.
6. Paramètres solveur (UNIQUEMENT si statut = FEASIBLE) :
   Augmenter solve_time_limit ou nb_iterations_lns.

RÈGLE STATUT :
- OPTIMAL : le solveur a prouvé l'optimalité de son modèle. Les leviers 1-5
  ci-dessus sont HORS modèle - ils restent actionnables manuellement.
  Ne jamais dire "relancer le solveur" si statut = OPTIMAL.
- FEASIBLE : optimalité non prouvée. Relancer avec plus de temps EST valide
  en PLUS des leviers 1-5.

COMPATIBILITÉ MACHINE :
- Une machine est compatible avec une opération UNIQUEMENT si cette opération
  figure dans la liste OperationsList de la machine.
- Ne jamais recommander une machine pour une opération absente de sa liste.
═══════════════════════════════════════════════════════════
"""

# ---------------------------------------------------------------------------
# LLM Options
# ---------------------------------------------------------------------------

LLM_OPTIONS_LOOKUP = {
    "num_predict": 80,
    "num_ctx":     1024,
    "temperature": 0.0,
    "top_p":       1.0,
}

LLM_OPTIONS_ANALYSIS = {
    "num_predict": 120,   # FIX-23: 400->120 - at ~2s/token CPU, 120 tokens ≈ 240s.
    "num_ctx":     1024,  # Raise OLLAMA_TIMEOUT to 300s for analysis path too (see below).
    "temperature": 0.0,
    "top_p":       1.0,
}

# FIX-10b: num_predict reduced 400->180, num_ctx 1536→1024 (more aggressive).
# Target: <200s total generation, safe margin before 300s timeout.
# At ~2s/token on CPU, 150 tokens = ~300s = right at the limit (no margin).
# FIX-16: num_predict 150->100, num_ctx 1024→768.
# 100 tokens = ~200s worst case -> 100s margin before timeout.
# 3 tight bullet points fit in 80-100 tokens. num_ctx 768 is enough for the
# compressed prompt (facts + constraints block ≈ 350-400 input tokens).
LLM_OPTIONS_IMPROVEMENT = {
    "num_predict": 250,  # FIX-20: rewriter only needs ~150 tokens; 250 gives margin
    "num_ctx":     1024, # FIX-20: rewriter prompt is short (~200 input tokens)
    "temperature": 0.1,
    "top_p":       0.9,
}

# ---------------------------------------------------------------------------
# Question type constants
# ---------------------------------------------------------------------------

_QTYPE_LOOKUP       = "lookup"
_QTYPE_MACHINE_LOAD = "machine_load"
_QTYPE_MAKESPAN     = "makespan"
_QTYPE_IMPROVEMENT  = "improvement"
_QTYPE_SEQUENCE     = "sequence"
_QTYPE_SUMMARY      = "summary"
_QTYPE_ANALYSIS     = "analysis"

LOOKUP_PATTERNS = re.compile(
    r"(quell?es?\s+machines?\s+(ont|a|on[t]?|did|have|has|effectu|fait|r[eé]alis|utilis)"
    r"|which\s+machines?\s+(did|performed|ran|processed|used|handled)"
    r"|(ont|a)\s+fait\s+l.op[eé]ration"
    r"|machines?\s+(pour|for|sur|on|doing|used\s+for)\s+(l.op[eé]ration|the\s+op)"
    r"|op[eé]ration\s+\w+\s+(machines?|sur\s+quell?es?))",
    re.IGNORECASE,
)

SEQUENCE_KEYWORDS = [
    "séquence", "sequence", "séquencement", "ordre des", "order of",
    "startpm", "endpm", "start_pm", "end_pm", "lotidx", "lot idx",
    "quand", "when", "planifié", "scheduled", "fragmentation",
    "show me the", "montre moi le", "détail", "detail", "timeline", "chronologie",
]

MAKESPAN_KEYWORDS = [
    "makespan", "durée", "duree", "combien de temps", "how long",
    "combien d'heure", "combien d'heures", "how many hour",
    "temps total", "total time", "how long does", "combien dure",
    "planning dure", "planning took",
]

SUMMARY_KEYWORDS = [
    "machine", "fait", "effectué", "réalisé", "utilisé",
    "quel", "quelle", "quels", "quelles", "who did", "which machine",
    "opération", "operation", "poudre", "javellisation", "stonage",
    "lavage", "rinçage", "essorage", "séchage", "sechage", "finition", "trempage",
    "cmd",
]

FACTUAL_KEYWORDS = [
    "combien de commande", "nombre de commande", "how many order",
    "combien de ligne", "nombre de ligne", "how many line",
    "date de début", "date debut", "date de generation", "généré le",
    "start date", "generated on",
    "résumé du planning", "summary of the planning",
]

COMPOUND_SIGNALS = [
    "pourquoi", "why", "comment", "how", "réduire", "reduce",
    "améliorer", "improve", "expliquer", "explain", "analyse",
    "recommande", "suggest", "conseille", "optimal", "makespan",
    "statut", "combien de temps", "durée", "duree",
]

IMPROVEMENT_KEYWORDS = [
    "améliorer", "amélioration", "optimiser", "optimisation",
    "recommandation", "recommande", "recommandez", "suggestion",
    "comment réduire", "comment améliorer", "comment optimiser",
    "réduire le makespan", "réduire les retards", "réduire la fragmentation",
    "raccourcir", "accélérer le planning",
    "bilan", "évaluation", "évaluer", "analyse globale",
    "que pensez", "que proposez", "pistes d'amélioration",
    "axe d'amélioration", "leviers", "plan d'action",
    "points à améliorer", "points faibles", "bottleneck", "goulot",
    "inefficacité", "inefficiencies",
    "how to reduce", "reduce makespan", "reduce delay", "reduce fragmentation",
    "recommend", "what should", "what can", "what actions",
    "suggestions", "advise", "advice",
    "how to improve", "how to optimize", "performance issues",
    "assess", "assessment", "your assessment",
    # NOTE: "analyse", "analysis", "analyze" intentionally removed - they are too
    # broad and misroute legitimate machine-load / summary questions ("analyser la
    # charge machine", "analyze which machine is overloaded") to the improvement
    # path.  C# ChatController correctly classifies these via _improvementRx and
    # sends full C+D+E+F data when needed; Python's own classifier must not
    # override MACHINE_LOAD_PATTERNS with a bare substring match on "analyse".
]

_MAKESPAN_IMPROVE_PATTERNS = re.compile(
    r"(comment\s+(r[eé]duire|raccourcir|diminuer|am[eé]liorer|optimiser)"
    r"|how\s+to\s+(reduce|shorten|improve|cut)"
    r"|r[eé]duire\s+(le\s+)?makespan"
    r"|reduce\s+(the\s+)?makespan"
    r"|raccourcir\s+(le\s+)?makespan)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Out-of-scope topic detector
# ---------------------------------------------------------------------------

_OUT_OF_SCOPE_TOPICS = re.compile(
    r"\b("
    r"op[eé]rateur|employ[eé]|personnel|worker|staff|ouvrier|technicien|"
    r"chef d.[eé]quipe|team leader|absent|cong[eé]|vacation|shift|"
    r"salaire|salary|wage|paye|"
    r"co[uû]t|cost|prix|price|budget|facturation|invoice|tarif|"
    r"rentabilit[eé]|profitabilit|marge|margin|chiffre d.affaire|revenue|"
    r"qualit[eé]|quality|d[eé]faut|defect|rejet|reject|non.conform|"
    r"temp[eé]rature|temperature|pression|pressure|ph|concentration|"
    r"chimique|chemical|produit|product|formul|"
    r"maintenance|panne|breakdown|r[eé]paration|repair|pi[eè]ce de rechange|"
    r"spare part|lubrifi|calibr|"
    r"stock|inventory|approvisionnement|supply|fournisseur|supplier|"
    r"mati[eè]re premi[eè]re|raw material|livraison|delivery|"
    r"mot de passe|password|login|compte|account|email|"
    r"m[eé]t[eé]o|weather|news|actualit[eé]"
    r")\b",
    re.IGNORECASE,
)


def _is_out_of_scope(question: str) -> bool:
    return bool(_OUT_OF_SCOPE_TOPICS.search(question))


MACHINE_LOAD_PATTERNS = re.compile(
    r"(surcharg[eéèê]?e?s?|surcharges|overload|over.load"
    r"|sous.utilis|under.utilis|underus"
    r"|taux.charg|taux de charge|utilisation.rate|charge.rate"
    r"|quelle.*machine.*charg|which.*machine.*load"
    r"|machine.*surcharg|machine.*sous.utilis"
    r"|quel.*taux|what.*utilisation|what.*utilization"
    r"|machines?\s+(les\s+)?(plus\s+)?(charg|activ|occup|busy|load)"
    r"|how\s+(loaded|busy|utilized)\s+are"
    r"|nom\s+(du|de\s+la|d.une?)\s+machine"
    r"|machine\s+(la\s+plus\s+)?surcharg"
    r"|machine\s+(la\s+plus\s+)?occup"
    r"|charge\s+(des\s+)?machines?"     # "charge machine", "charge des machines"
    r"|machines?\s+charg)",
    re.IGNORECASE,
)

_SPECIFIC_ENTITY = re.compile(
    r'\b(cmd\s*\w+|lot\w*|poudre|javellisation|stonage|lavage|rin[çc]age|'
    r'essorage|s[eé]chage|finition|trempage|'
    r'brongo|tupesa|machine\s+\w+)\b',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Security guard
# ---------------------------------------------------------------------------

def _assert_db_rows_safe(db_rows: dict) -> dict:
    safe = {}
    for key, rows in db_rows.items():
        if not isinstance(rows, list):
            safe[key] = []
            continue
        cap = ROW_CAPS.get(key, 50)
        cleaned = []
        for row in rows[:cap]:
            if not isinstance(row, dict):
                continue
            clean_row = {}
            for col, val in row.items():
                if isinstance(val, str) and _SQL_DANGER.search(val):
                    clean_row[col] = "[REDACTED]"
                else:
                    clean_row[col] = val
            cleaned.append(clean_row)
        safe[key] = cleaned
    return safe

# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    print("[RAG] WARNING: faiss-cpu not installed - vector search disabled")


class FaissIndex:
    def __init__(self):
        self.index = None
        self.texts: List[str] = []

    def build(self, texts: List[str], embeddings: np.ndarray):
        dim = embeddings.shape[1]
        if _FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)
        self.texts = texts

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[str, float, int]]:
        if not _FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        D, I = self.index.search(query_vec.reshape(1, -1).astype("float32"), k)
        return [(self.texts[i], float(D[0][j]), i) for j, i in enumerate(I[0]) if i >= 0]


_faiss_index = FaissIndex()

if FAISS_INDEX_PATH.exists() and FAISS_META_PATH.exists() and _FAISS_AVAILABLE:
    try:
        _faiss_index.index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_META_PATH, "rb") as f:
            _faiss_index.texts = pickle.load(f)
        print(f"[RAG] FAISS index loaded: {_faiss_index.index.ntotal} vectors")
    except Exception as e:
        print(f"[RAG] Could not load FAISS index: {e}")

# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

async def embed(texts: List[str]) -> np.ndarray:
    async with httpx.AsyncClient(timeout=60) as client:
        vecs = []
        for t in texts:
            r = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": t},
            )
            r.raise_for_status()
            vecs.append(r.json()["embedding"])
    return np.array(vecs, dtype="float32")

# ---------------------------------------------------------------------------
# Index planning rows (called after /api/planning/run)
# ---------------------------------------------------------------------------

async def index_planning_rows(planning_id: int, text: str):
    try:
        vec = await embed([text])
        _faiss_index.texts.append(text)
        if _FAISS_AVAILABLE:
            if _faiss_index.index is None:
                _faiss_index.index = faiss.IndexFlatL2(vec.shape[1])
            _faiss_index.index.add(vec)
            faiss.write_index(_faiss_index.index, str(FAISS_INDEX_PATH))
            with open(FAISS_META_PATH, "wb") as f:
                pickle.dump(_faiss_index.texts, f)
        print(f"[RAG] Indexed planning {planning_id}")
    except Exception as e:
        print(f"[RAG] Indexing error: {e}")

# ---------------------------------------------------------------------------
# Makespan helpers
# ---------------------------------------------------------------------------

def _pm_to_real_minutes(pm_value) -> Optional[int]:
    try:
        return int(pm_value) * MINUTES_PER_PM
    except (TypeError, ValueError):
        return None


def _minutes_to_hhmm(total_minutes: int) -> str:
    h   = total_minutes // 60
    rem = total_minutes % 60
    return f"{h}h{rem:02d}" if rem else f"{h}h00"


def _derive_makespan_from_query_e(rows_e: list) -> Optional[int]:
    if not rows_e:
        return None
    try:
        starts = [int(r["StartPM"]) for r in rows_e if "StartPM" in r]
        ends   = [int(r["EndPM"])   for r in rows_e if "EndPM"   in r]
        if not starts or not ends:
            return None
        return (max(ends) - min(starts)) * MINUTES_PER_PM
    except (TypeError, ValueError, KeyError):
        return None


def _compute_makespan_real(db_rows: dict) -> str:
    def _fmt(mins: int, src: str) -> str:
        hhmm = _minutes_to_hhmm(mins)
        days = mins // (24 * 60)
        note = f"(< 1 jour - {src})" if days == 0 else f"({days} jour(s) - {src})"
        print(f"[RAG] MAKESPAN_REAL={hhmm} via {src}")
        return f"{hhmm} {note}"

    rows_e = db_rows.get("E", [])
    if rows_e:
        m = _derive_makespan_from_query_e(rows_e)
        if m and m > 0:
            return _fmt(m, "Query E")

    rows_a = db_rows.get("A", [])
    if rows_a:
        m = _pm_to_real_minutes(rows_a[0].get("MakespanPM"))
        if m and m > 0:
            return _fmt(m, "MakespanPM")
        d = rows_a[0].get("MakespanDays")
        if d is not None:
            try:
                dv = int(d)
                return f"{'< 1 jour' if dv == 0 else str(dv) + ' jour(s)'} (MakespanDays)"
            except (TypeError, ValueError):
                pass

    return "non disponible"

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_FRENCH_HARD_TRIGGERS = re.compile(
    r'\b(améliorer|amélioration|optimiser|optimisation|analyser|'
    r'bilan|évaluation|évaluer|goulot|recommande|recommandation|conseil|'
    r'pourquoi|combien|pistes?|axes?|leviers?|proposez|suggères?|suggérez|'
    r'séquence|séquencement|fragmentation|planifié)\b'
    r'|ce planning|ce résultat|cette planification|comment améliorer|comment optimiser',
    re.IGNORECASE,
)

_FRENCH_MARKERS = re.compile(
    r'\b(le|la|les|un|une|des|est|sont|pas|ne|ce|cette|ces|mon|ton|son|'
    r'que|qui|quoi|quel|quelle|quels|quelles|et|ou|mais|donc|or|ni|car|'
    r'est-ce|y a-t-il|combien|comment|pourquoi|quand|votre|notre|leur|leurs|'
    r'avec|sans|pour|sur|dans|par|de|du|au|aux|en|entre|vers|chez|'
    r'retard|commande|planning|machine|optimal|makespan|lot|'
    r'améliorer|amélioration|optimiser|optimisation|analyse|analyser|'
    r'bilan|évaluation|goulot|recommande|conseil|fragmentation|'
    r'séquence|séquencement|planifié|planifier|piste|axe)\b',
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
    import unicodedata as _ud
    q = _ud.normalize("NFC", question)
    if _FRENCH_HARD_TRIGGERS.search(q):
        return "fr"
    fr = len(_FRENCH_MARKERS.findall(q))
    en = len(_ENGLISH_MARKERS.findall(q))
    return "en" if en > fr else "fr"


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
    if re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{5,}', q):
        return True
    return False

# ---------------------------------------------------------------------------
# Question classifier
# ---------------------------------------------------------------------------

def _classify_question(question: str) -> str:
    import unicodedata as _ud
    q_nfc = _ud.normalize("NFC", question)
    q_lower = q_nfc.lower()

    if LOOKUP_PATTERNS.search(q_nfc):
        return _QTYPE_LOOKUP
    if MACHINE_LOAD_PATTERNS.search(q_nfc):
        return _QTYPE_MACHINE_LOAD

    # Improvement check BEFORE makespan: "comment réduire le makespan" must
    # go to improvement path, not free-LLM makespan path.
    if _MAKESPAN_IMPROVE_PATTERNS.search(q_nfc):
        return _QTYPE_IMPROVEMENT
    if any(kw in q_lower for kw in IMPROVEMENT_KEYWORDS):
        if _SPECIFIC_ENTITY.search(q_nfc):
            return _QTYPE_SEQUENCE
        return _QTYPE_IMPROVEMENT

    if any(kw in q_lower for kw in MAKESPAN_KEYWORDS):
        return _QTYPE_MAKESPAN

    if any(kw in q_lower for kw in SEQUENCE_KEYWORDS):
        return _QTYPE_SEQUENCE
    if any(kw in q_lower for kw in SUMMARY_KEYWORDS):
        return _QTYPE_SUMMARY
    return _QTYPE_ANALYSIS


def _is_factual_lookup(question: str) -> bool:
    q = question.lower().strip()
    if any(sig in q for sig in COMPOUND_SIGNALS):
        return False
    return any(kw in q for kw in FACTUAL_KEYWORDS)


# ---------------------------------------------------------------------------
# Whitelist builders (keep deterministic - produce validated INPUT for LLM)
# ---------------------------------------------------------------------------

def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _validate_machine_name(name: str, known_machines: Set[str]) -> bool:
    """Return True only if name appears in db_rows - prevents phantom machine names."""
    return name.strip() in known_machines


def _get_machine_name(row: dict) -> str:
    """
    Read machine name using all known column aliases.
    Tries NomMachine first (Query C/F/D style), then MachineName (Query E/D style),
    then Machine as a last-resort fallback.
    """
    for col in ("NomMachine", "MachineName", "Machine"):
        v = str(row.get(col, "")).strip()
        if v:
            return v
    return ""


def _build_known_machines(db_rows: dict) -> Set[str]:
    """Collect all machine names from any query - the whitelist."""
    names: Set[str] = set()
    for key in ("C", "D", "E", "F"):
        for r in db_rows.get(key, []):
            v = _get_machine_name(r)
            if v:
                names.add(v)
    return names


def _build_known_commands(db_rows: dict) -> Set[str]:
    """Collect all command numbers from any query - the whitelist."""
    cmds: Set[str] = set()
    for key in ("B", "D", "E"):
        for r in db_rows.get(key, []):
            v = str(r.get("NumeroCommande", "")).strip()
            if v:
                cmds.add(v)
    return cmds


def _build_known_operations(db_rows: dict) -> Set[str]:
    """
    Collect all planned operation names.
    Primary source: Query E. Fallback: Query C OperationsList.
    """
    ops: Set[str] = set()
    for r in db_rows.get("E", []):
        v = str(r.get("NomOperation", "")).strip()
        if v:
            ops.add(v)
    if not ops:
        for r in db_rows.get("C", []):
            ops_raw = r.get("OperationsList") or r.get("Operations", "")
            if isinstance(ops_raw, list):
                ops.update(o.strip() for o in ops_raw if o.strip())
            elif ops_raw:
                ops.update(o.strip() for o in str(ops_raw).split(",") if o.strip())
    return ops


def _build_active_machines(db_rows: dict) -> Set[str]:
    """Return the set of machines ACTUALLY USED in this planning."""
    active: Set[str] = set()
    for r in db_rows.get("C", []):
        if _safe_int(r.get("MinutesPlanifiees", 0)) > 0:
            name = _get_machine_name(r)
            if name:
                active.add(name)
    for r in db_rows.get("E", []):
        name = _get_machine_name(r)
        if name:
            active.add(name)
    return active


# ===========================================================================
# v5.0 - FACT EXTRACTION LAYER
# These replace the old _analyze_* functions.
# Each returns structured dicts (raw data for Mistral), NOT final prose.
# ===========================================================================

def _extract_delay_facts(db_rows: dict, known_cmds: Set[str]) -> List[dict]:
    """Extract delay data from Query B as structured dicts."""
    facts = []
    for r in db_rows.get("B", []):
        cmd = str(r.get("NumeroCommande", "")).strip()
        if not cmd:
            continue
        # Only cross-check against whitelist when whitelist is non-empty.
        # If known_cmds is empty (D and E had no NumeroCommande column),
        # accept all B rows unconditionally - they came directly from the DB.
        if known_cmds and cmd not in known_cmds:
            continue
        facts.append({
            "type":              "delay",
            "commande":          cmd,
            "jours_retard":      _safe_int(r.get("JoursRetard", 0)),
            "retard_structurel": bool(_safe_int(r.get("RetardStructurel", 0))),
            "date_export":       str(r.get("DateExport", "")),
        })
    return facts


def _extract_machine_facts(db_rows: dict, known_machines: Set[str]) -> List[dict]:
    """Extract machine load state from Query C and unused machines from Query F."""
    facts = []

    # Active machines (Query C)
    for r in db_rows.get("C", []):
        name = _get_machine_name(r)
        if not _validate_machine_name(name, known_machines):
            continue
        ops_raw = r.get("OperationsList") or r.get("Operations", "")
        ops = (
            [o.strip() for o in ops_raw if o.strip()]
            if isinstance(ops_raw, list)
            else [o.strip() for o in str(ops_raw).split(",") if o.strip()]
        )
        facts.append({
            "type":               "machine_active",
            "machine":            name,
            "etat":               str(r.get("Etat", "NOMINAL")).upper().strip(),
            "taux_charge_pct":    r.get("TauxChargePct", 0),
            "minutes_planifiees": _safe_int(r.get("MinutesPlanifiees", 0)),
            "nb_commandes":       _safe_int(r.get("NbCommandes", 0)),
            "operations":         ops,
        })

    # Unused machines (Query F)
    for r in db_rows.get("F", []):
        name = _get_machine_name(r)
        if not name:
            continue
        ops_raw = r.get("OperationsList") or r.get("Operations", "")
        ops = (
            [o.strip() for o in ops_raw if o.strip()]
            if isinstance(ops_raw, list)
            else [o.strip() for o in str(ops_raw).split(",") if o.strip()]
        )
        facts.append({
            "type":        "machine_unused",
            "machine":     name,
            "capacite":    r.get("CapaciteMax", "?"),
            "operations":  ops,
            "statut":      str(r.get("Statut", "Fonctionnel")),
        })

    return facts


def _extract_fragmentation_facts(
    db_rows: dict,
    known_machines: Set[str],
    known_cmds: Set[str],
) -> List[dict]:
    """Extract fragmentation data from Query D as structured dicts."""
    facts = []
    for r in db_rows.get("D", []):
        nb_lots = _safe_int(r.get("NbLots", 0))
        if nb_lots <= 5:
            continue
        cmd = str(r.get("NumeroCommande", "")).strip()
        # Same guard as _extract_delay_facts: skip whitelist check when empty
        if known_cmds and cmd not in known_cmds:
            continue
        machine = _get_machine_name(r)
        if machine and not _validate_machine_name(machine, known_machines):
            continue
        lot_size_recette = _safe_int(r.get("LotSizeRecette", 0)) or _safe_int(r.get("QuantiteLot", 0))
        cap_machine      = _safe_int(r.get("CapaciteMaxMachine", 0))
        # Binding constraint: whichever is smaller drives lot count.
        # If cap_machine >= lot_size_recette, the recipe lot size is the bottleneck
        # → machine-capacity upgrade will NOT reduce lot count (invalid lever).
        # If cap_machine < lot_size_recette, machine capacity IS the bottleneck
        # → switching to higher-capacity machine reduces lot count (valid lever).
        binding = "machine" if (cap_machine > 0 and cap_machine < lot_size_recette) else "recette"
        facts.append({
            "type":       "fragmentation",
            "commande":   cmd,
            "operation":  str(r.get("NomOperation", "")).strip(),
            "machine":    machine,
            "nb_lots":    nb_lots,
            # ⚠️ lot_size is a RECIPE CONSTRAINT - LLM must never recommend changing it
            "lot_size_recette": lot_size_recette,
            "cap_machine":      cap_machine,
            "binding_constraint": binding,   # "machine" or "recette"
            "quantite":         _safe_int(r.get("QuantiteCommande", 0)) or _safe_int(r.get("Quantite", 0)),
        })
    return facts


def _extract_bottleneck_facts(db_rows: dict, known_machines: Set[str]) -> List[dict]:
    """Extract bottleneck data - machines with highest treatment minutes from Query E."""
    agg: Dict[str, int] = {}
    for r in db_rows.get("E", []):
        machine = _get_machine_name(r)
        if not _validate_machine_name(machine, known_machines):
            continue
        mins = _safe_int(
            r.get("MinutesTraitement") or r.get("DureeMinutes") or r.get("DureeTotale") or 0
        )
        agg[machine] = agg.get(machine, 0) + mins

    facts = []
    for machine, total_mins in sorted(agg.items(), key=lambda x: -x[1])[:5]:
        facts.append({
            "type":                   "bottleneck",
            "machine":                machine,
            "total_traitement_min":   total_mins,
            "total_traitement_hhmm":  _minutes_to_hhmm(total_mins),
        })
    return facts


def _extract_setup_facts(db_rows: dict, known_machines: Set[str]) -> List[dict]:
    """
    Extract setup overhead data from Query E.
    Only includes (op, machine) pairs where setup >= 40% of cycle time.

    BUG-FIX: previous version accumulated NbLots (the total-lots field) from every
    row for the same (op, machine) key.  Query E has one row *per lot execution*, so
    a 10-lot operation on Brongo 2 produced 10 rows each carrying NbLots=10, giving
    nb_lots_total=100 instead of 10 - and waste inflated by 10x.

    Correct approach: count actual rows (= actual lot executions) within each
    (op, machine, commande) triple, then sum those real lot counts per (op, machine).
    The ratio filter uses per-commande NbLots to decide significance, not the sum.
    """
    # Step 1: aggregate per (op, machine, commande) - one entry per unique combination.
    # Row count within the triple = actual number of lot executions.
    per_cmd: Dict[Tuple[str, str, str], dict] = {}
    for r in db_rows.get("E", []):
        machine = _get_machine_name(r)
        if not _validate_machine_name(machine, known_machines):
            continue
        op       = str(r.get("NomOperation", "")).strip()
        cmd      = str(r.get("NumeroCommande", "")).strip()
        dur      = _safe_int(r.get("DureeMinutes", 0))
        charge   = _safe_int(r.get("TempsChargementMinutes", 0))
        decharge = _safe_int(r.get("TempsDecharementMinutes", 0))
        nb_lots  = _safe_int(r.get("NbLots", 0))   # total lots for this (cmd, op, machine)
        setup    = charge + decharge
        total    = dur + setup
        if total == 0 or setup / total < SETUP_RATIO_THRESHOLD:
            continue

        key = (op, machine, cmd)
        if key not in per_cmd:
            per_cmd[key] = {
                "op": op, "machine": machine, "cmd": cmd,
                "dur": dur, "charge": charge, "decharge": decharge,
                "setup": setup, "total": total,
                "nb_lots": nb_lots,   # from the NbLots field (constant per group)
                "row_count": 0,       # actual rows = actual lot executions
            }
        per_cmd[key]["row_count"] += 1

    # Step 2: for each (op, machine, commande), use the MINIMUM of NbLots and row_count
    # as the true lot count.  NbLots is the solver's declared total; row_count is what
    # actually appears in Query E.  They should agree - taking min() guards against
    # Query E being capped (ROW_CAPS["E"] = 60).
    # Only keep triples where the commande has > 2 lots (avoid trivial cases).
    agg: Dict[Tuple[str, str], dict] = {}
    for (op, machine, cmd), d in per_cmd.items():
        true_lots = min(d["nb_lots"], d["row_count"])
        if true_lots <= 2:
            continue
        key2 = (op, machine)
        if key2 not in agg:
            agg[key2] = {
                "operation": op, "machine": machine,
                "duree_traitement_min":   d["dur"],
                "temps_chargement_min":   d["charge"],
                "temps_dechargement_min": d["decharge"],
                "setup_total_min":        d["setup"],
                "ratio_setup_pct":        round(d["setup"] / d["total"] * 100),
                "nb_lots_total": 0,
                "commandes": set(),
            }
        agg[key2]["nb_lots_total"] += true_lots
        agg[key2]["commandes"].add(cmd)

    facts = []
    for d in agg.values():
        waste = d["setup_total_min"] * d["nb_lots_total"]
        d["commandes"] = sorted(d["commandes"])
        d["gaspillage_total_min"]  = waste
        d["gaspillage_total_hhmm"] = _minutes_to_hhmm(waste)
        d["type"] = "setup_overhead"
        facts.append(d)
    return facts


def _extract_status_facts(db_rows: dict) -> dict:
    """Extract solver status from Query A."""
    rows_a = db_rows.get("A", [])
    if not rows_a:
        return {}
    return {
        "statut":          str(rows_a[0].get("Statut", "")).lower().strip(),
        "nb_commandes":    rows_a[0].get("NombreCommandes", "?"),
        "nb_lignes_gantt": rows_a[0].get("NombreLignes", "?"),
        "makespan_reel":   _compute_makespan_real(db_rows),
    }


# ---------------------------------------------------------------------------
# v5.0 - Context builder (replaces _run_all_analyzers)
# ---------------------------------------------------------------------------

def _build_improvement_context(db_rows: dict) -> Tuple[dict, str]:
    """
    Extract all structured facts for injection into Mistral's reasoning prompt.
    Returns (context_dict, refusal_msg).
    refusal_msg is non-empty only if critical queries are missing.
    """
    rows_c = db_rows.get("C", [])
    rows_e = db_rows.get("E", [])
    rows_f = db_rows.get("F", [])

    # Log what we received for debugging
    for key in ("A", "B", "C", "D", "E", "F"):
        rows = db_rows.get(key, [])
        if rows:
            cols = list(rows[0].keys()) if rows else []
            print(f"[ANALYZER] Query {key}: {len(rows)} rows | columns={cols}")
        else:
            print(f"[ANALYZER] Query {key}: EMPTY")

    if not rows_c and not rows_e and not rows_f:
        return {}, (
            "Données insuffisantes pour analyser ce planning. "
            "Les requêtes SQL C, E et F sont vides."
        )

    known_machines = _build_known_machines(db_rows)
    known_cmds     = _build_known_commands(db_rows)
    planned_ops    = _build_known_operations(db_rows)

    print(f"[ANALYZER] known_machines ({len(known_machines)}): {sorted(known_machines)}")
    print(f"[ANALYZER] known_cmds    ({len(known_cmds)}): {sorted(known_cmds)}")
    print(f"[ANALYZER] planned_ops   ({len(planned_ops)}): {sorted(planned_ops)}")

    delays        = _extract_delay_facts(db_rows, known_cmds)
    machines      = _extract_machine_facts(db_rows, known_machines)
    fragmentation = _extract_fragmentation_facts(db_rows, known_machines, known_cmds)
    bottlenecks   = _extract_bottleneck_facts(db_rows, known_machines)
    setup         = _extract_setup_facts(db_rows, known_machines)
    status        = _extract_status_facts(db_rows)

    print(f"[IMPROVEMENT] delays={len(delays)} machines={len(machines)} "
          f"frag={len(fragmentation)} bottlenecks={len(bottlenecks)} setup={len(setup)}")

    # Build machine->ops map for ALL machines (active + unused).
    # This is the ground truth for compatibility - if a machine is not in this
    # map for a given operation, it MUST NOT be recommended for that operation.
    machine_ops_map: dict = {}
    for r in db_rows.get("C", []):
        name = _get_machine_name(r)
        if not name:
            continue
        ops_raw = r.get("OperationsList") or r.get("Operations", "")
        ops = (
            [o.strip() for o in ops_raw if o.strip()]
            if isinstance(ops_raw, list)
            else [o.strip() for o in str(ops_raw).split(",") if o.strip()]
        )
        machine_ops_map[name] = ops
    for r in db_rows.get("F", []):
        name = _get_machine_name(r)
        if not name:
            continue
        ops_raw = r.get("OperationsList") or r.get("Operations", "")
        ops = (
            [o.strip() for o in ops_raw if o.strip()]
            if isinstance(ops_raw, list)
            else [o.strip() for o in str(ops_raw).split(",") if o.strip()]
        )
        machine_ops_map[name] = ops

    # Filter unused machines to only those compatible with at least one
    # planned operation - these are the ONLY valid activation candidates.
    planned_ops_lower = {op.lower() for op in planned_ops}
    compatible_unused: list = []
    incompatible_unused: list = []
    for r in rows_f:
        name = _get_machine_name(r)
        if not name:
            continue
        machine_ops = machine_ops_map.get(name, [])
        machine_ops_lower = {o.lower() for o in machine_ops}
        overlap = machine_ops_lower & planned_ops_lower
        if overlap:
            compatible_unused.append({
                "machine":    name,
                "operations": machine_ops,
                "compatible_with": sorted(
                    op for op in planned_ops if op.lower() in overlap
                ),
            })
        else:
            incompatible_unused.append({
                "machine":    name,
                "operations": machine_ops,
            })

    print(
        f"[ANALYZER] compatible_unused={[m['machine'] for m in compatible_unused]} | "
        f"incompatible_unused={[m['machine'] for m in incompatible_unused]}"
    )

    context = {
        "status":         status,
        "delays":         delays,
        "machines":       machines,
        "fragmentation":  fragmentation,
        "bottlenecks":    bottlenecks,
        "setup_overhead": setup,
        "_meta": {
            "known_machines":      sorted(known_machines),
            "known_cmds":          sorted(known_cmds),
            "planned_ops":         sorted(planned_ops),
            "active_machines":     sorted(_build_active_machines(db_rows)),
            "unused_machines":     [_get_machine_name(r) for r in rows_f if _get_machine_name(r)],
            "compatible_unused":   compatible_unused,
            "incompatible_unused": incompatible_unused,
            "machine_ops_map":     machine_ops_map,
        },
    }
    return context, ""


# ---------------------------------------------------------------------------
# v5.0 - Prompt builder
# ---------------------------------------------------------------------------

def _build_improvement_prompt(
    planning_id: int,
    question: str,
    context: dict,
    rag_chunks: List[str],
    lang: str,
) -> Tuple[str, str]:
    # FIX-8 + FIX-9: Compact prompt for CPU-only machines with strict
    # machine-operation compatibility enforcement.
    # - CPSAT_RULES_PROMPT and HARD_RULES removed (too many tokens).
    # - Facts as flat key=value lines instead of indented JSON (~40% token saving).
    # - compatible_unused: only machines whose OperationsList overlaps planned_ops.
    # - incompatible_unused listed explicitly so Mistral knows NOT to suggest them.
    # - RAG chunk capped at 1, truncated to 300 chars.
    # FIX-Q-AWARE: system prompt is tailored to the question's specific focus
    # so "comment réduire le makespan" gets makespan-focused bullets instead of
    # generic improvement bullets.
    fr = (lang == "fr")
    meta   = context["_meta"]
    status = context.get("status", {})

    lang_line     = "REPONDRE EN FRANCAIS." if fr else "RESPOND IN ENGLISH."
    solver_status = status.get("statut", "?").upper()
    makespan      = status.get("makespan_reel", "?")

    # Build per-machine ops summary for active machines
    active_with_ops = []
    for m in context.get("machines", []):
        if m["type"] == "machine_active":
            ops_str = ",".join(m.get("operations", []))
            active_with_ops.append(f"{m['machine']}[{ops_str}]")

    # Build compatible unused machines list with their compatible operations
    compatible_unused = meta.get("compatible_unused", [])
    incompatible_unused = meta.get("incompatible_unused", [])

    compatible_lines = []
    for m in compatible_unused:
        ops_str  = ",".join(m.get("operations", []))
        comp_str = ",".join(m.get("compatible_with", []))
        compatible_lines.append(f"{m['machine']}[ops:{ops_str}][compatible:{comp_str}]")

    incompatible_names = [m["machine"] for m in incompatible_unused]

    # ── Question-specific focus directive ──────────────────────────────────
    # Detect what the user is specifically asking about and adapt the instructions.
    q_lower = question.lower()
    _is_makespan_q    = any(kw in q_lower for kw in [
        "makespan", "réduire", "réduction", "raccourcir", "accélérer",
        "reduce", "shorten", "faster", "speed",
    ])
    _is_delay_q       = any(kw in q_lower for kw in [
        "retard", "délai", "late", "delay", "export", "deadline",
    ])
    _is_fragmentation = any(kw in q_lower for kw in [
        "fragmentation", "lots", "lot", "batch", "fractionné",
    ])
    _is_machine_q     = any(kw in q_lower for kw in [
        "machine", "charge", "load", "surchargée", "sous-utilisée", "unused", "inactive",
    ])

    if _is_makespan_q and not _is_delay_q:
        focus_instruction = (
            "FOCUS MAKESPAN: activer machines compatibles, rééquilibrer goulots, réduire fragmentation. PAS de retards."
            if fr else
            "FOCUS MAKESPAN: activate compatible machines, rebalance bottlenecks, reduce fragmentation. NO delay talk."
        )
    elif _is_delay_q:
        focus_instruction = (
            "FOCUS RETARDS: paralléliser sur machines compatibles, prioriser urgences, livraison partielle si structurel."
            if fr else
            "FOCUS DELAYS: parallelize on compatible machines, prioritize urgent orders, partial delivery if structural."
        )
    elif _is_fragmentation:
        focus_instruction = (
            "FOCUS FRAGMENTATION: identifier commandes fragmentées. "
            "Si contrainte_liante=machine: proposer machine plus grande capacite. "
            "Si contrainte_liante=recette: NE PAS proposer machine plus grande capacite (lot_recette fixe), proposer parallelisation."
            if fr else
            "FOCUS FRAGMENTATION: identify fragmented orders. "
            "If binding_constraint=machine: suggest higher-capacity machine. "
            "If binding_constraint=recette: DO NOT suggest higher-capacity machine (lot size is recipe-fixed), suggest parallelization."
        )
    elif _is_machine_q:
        focus_instruction = (
            "FOCUS CHARGE: machines >85% = surchargée, <40% = sous-utilisée. Proposer rééquilibrages chiffrés."
            if fr else
            "FOCUS LOAD: >85% = overloaded, <40% = underused. Suggest quantified rebalancing."
        )
    else:
        focus_instruction = (
            "FOCUS GLOBAL: 3 actions prioritaires les plus impactantes avec KPI chiffré."
            if fr else
            "FOCUS GLOBAL: top 3 highest-impact actions with quantified KPI."
        )

    system_prompt = (
        f"{lang_line} Expert planif. denim. MAX 100 tokens.\n"
        f"COMPAT MACHINE ABSOLUE: machine ne peut traiter op QUE si op dans sa liste.\n"
        f"MACHINES INTERDIT: ne jamais recommander d'activer, utiliser, réparer ou remplacer les machines INTERDIT.\n"
        f"DURÉES RECETTE FIXES: ne jamais recommander de réduire/modifier les temps de chargement, déchargement ou traitement.\n"
        f"{focus_instruction}\n"
        "Format: 1. action -> impact chiffré. Pas de phrases longues."
    )

    fact_lines = []
    for d in context.get("delays", []):
        tag = "RETARD-STRUCTUREL" if d["retard_structurel"] else "RETARD"
        fact_lines.append(
            f"{tag} cmd={d['commande']} export={d['date_export']} retard={d['jours_retard']}j"
        )
    for m in context.get("machines", []):
        ops = ",".join(m.get("operations", []))
        if m["type"] == "machine_active":
            fact_lines.append(
                f"MACHINE-ACTIVE {m['machine']} charge={m['taux_charge_pct']}% "
                f"mins={m['minutes_planifiees']} ops={ops}"
            )
        else:
            fact_lines.append(f"MACHINE-INACTIVE {m['machine']} ops={ops}")
    for f in context.get("fragmentation", []):
        fact_lines.append(
            f"FRAGMENTATION cmd={f['commande']} op={f['operation']} "
            f"machine={f['machine']} lots={f['nb_lots']}"
        )
    for b in context.get("bottlenecks", []):
        fact_lines.append(f"GOULOT {b['machine']} traitement={b['total_traitement_hhmm']}")
    for s in context.get("setup_overhead", []):
        fact_lines.append(
            f"SETUP {s['operation']} sur {s['machine']} "
            f"ratio={s['ratio_setup_pct']}% gaspillage={s['gaspillage_total_hhmm']}"
        )

    facts_text = "\n".join(fact_lines[:8]) if fact_lines else "(aucune donnee)"  # FIX-16: cap 12->8, save ~50 input tokens

    # ── FIX-14 + FIX-16: Compact grounding block in user_prompt ───────────────
    # Injected as DATA (not system-prompt prose) so Mistral treats it as ground truth.
    # Kept ultra-compact to save input tokens (num_ctx=768 budget).
    planned_ops_str = ", ".join(sorted(meta.get("planned_ops", []))) or "?"
    can_activate_str = (
        "; ".join(
            f"{m['machine']}->{','.join(m['compatible_with'])}"
            for m in compatible_unused
        ) or "aucune"
    )
    forbidden_str = ", ".join(incompatible_names) if incompatible_names else "aucune"

    constraints_block = (
        f"[CONTRAINTES]\n"
        f"OPS={planned_ops_str} | ACTIVER={can_activate_str} | INTERDIT={forbidden_str}\n"
        f"-> Ne citer que ces ops. Ne recommander que les machines ACTIVER. JAMAIS les machines INTERDIT.\n"
        if fr else
        f"[CONSTRAINTS]\n"
        f"OPS={planned_ops_str} | ACTIVATE={can_activate_str} | FORBIDDEN={forbidden_str}\n"
        f"-> Only cite listed ops. Only recommend ACTIVATE machines. NEVER FORBIDDEN machines.\n"
    )

    # FIX-16: RAG chunk not injected - costs ~40 input tokens for minimal value on CPU.
    user_prompt = (
        f"[PLANNING #{planning_id}]\n"
        f"{facts_text}\n"
        f"{constraints_block}\n"
        f"[QUESTION] {question}"
    )

    return system_prompt, user_prompt



# ---------------------------------------------------------------------------
# v5.0 - FAISS retrieval helper
# ---------------------------------------------------------------------------

async def _retrieve_rag_chunks(question: str, k: int = TOP_K) -> List[str]:
    """
    Embed the question and retrieve the top-k most relevant FAISS chunks.
    Returns empty list on failure (FAISS degraded mode).
    """
    if _faiss_index.index is None or _faiss_index.index.ntotal == 0:
        return []
    try:
        vecs = await embed([question])
        hits = _faiss_index.search(vecs[0:1], k)
        return [text for text, score, idx in hits]
    except Exception as e:
        print(f"[RAG] FAISS retrieval failed: {e}")
        return []


# ---------------------------------------------------------------------------
# v5.0 - Machine mention pattern builder (for validation)
# ---------------------------------------------------------------------------

def _build_machine_mention_pattern(known_machines: Set[str]) -> Optional[re.Pattern]:
    """
    Build a regex that matches any known machine name variant in LLM output.
    Also matches partial mentions (e.g. 'Brongo' without the number).
    Returns None when known_machines is empty (nothing to check against).
    """
    if not known_machines:
        return None

    bases = set()
    for name in known_machines:
        parts = name.split()
        if parts:
            bases.add(re.escape(parts[0]))

    full_escaped  = [re.escape(n) for n in known_machines]
    base_patterns = [f"{b}(?:\\s+\\d+)?" for b in bases]
    all_patterns  = full_escaped + base_patterns
    return re.compile(
        r'\b(' + '|'.join(all_patterns) + r')\b',
        re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# v5.0 - Operation validation
# ---------------------------------------------------------------------------

def _validate_operations_in_reply(reply: str, known_ops: Set[str]) -> List[str]:
    """
    Check that every denim operation name cited in the LLM reply appears in Query E.

    FIX-2: returns empty list immediately when known_ops is empty.
    Without this guard, every mention of lavage/poudre/etc. would be flagged
    as hallucinated when Query E has no NomOperation column, causing the
    validator to always fall back to deterministic mode.
    """
    violations = []

    # FIX-2: can't validate operations if we have no whitelist to check against
    if not known_ops:
        print("[VALIDATION] known_ops is empty - skipping operation hallucination check")
        return violations

    rl = reply.lower()

    denim_ops = {
        "poudre", "javellisation", "stonage", "lavage", "rinçage",
        "essorage", "séchage", "finition", "trempage",
    }

    for op in denim_ops:
        if op in rl and op.lower() not in {o.lower() for o in known_ops}:
            violations.append(
                f"[OP_HALLUCINATION] Opération '{op}' citée mais absente de Query E"
            )
    return violations


# ---------------------------------------------------------------------------
# v5.0 - Numeric bounds & claims validator
# ---------------------------------------------------------------------------

def _extract_numeric_bounds(context: dict) -> dict:
    """Derive plausible numeric bounds from SQL context for LLM output validation."""
    bounds = {
        "max_minutes":    0,
        "max_lots":       0,
        "max_load_pct":   100,
        "max_delay_days": 0,
    }

    for m in context.get("machines", []):
        mins = m.get("minutes_planifiees", 0)
        pct  = m.get("taux_charge_pct", 0)
        if isinstance(mins, (int, float)):
            bounds["max_minutes"] = max(bounds["max_minutes"], int(mins))
        if isinstance(pct, (int, float)):
            bounds["max_load_pct"] = max(bounds["max_load_pct"], int(pct))

    for f in context.get("fragmentation", []):
        lots = f.get("nb_lots", 0)
        bounds["max_lots"] = max(bounds["max_lots"], int(lots))

    for d in context.get("delays", []):
        days = d.get("jours_retard", 0)
        bounds["max_delay_days"] = max(bounds["max_delay_days"], int(days))

    # Also check bottleneck minutes
    for b in context.get("bottlenecks", []):
        mins = b.get("total_traitement_min", 0)
        if isinstance(mins, (int, float)):
            bounds["max_minutes"] = max(bounds["max_minutes"], int(mins))

    # 20% tolerance for derived calculations
    bounds["max_minutes"]    = int(bounds["max_minutes"] * 1.2) + 120
    bounds["max_lots"]       = int(bounds["max_lots"] * 1.2) + 10
    bounds["max_delay_days"] = int(bounds["max_delay_days"] * 1.2) + 5

    return bounds


def _validate_numeric_claims(reply: str, bounds: dict) -> List[str]:
    """
    Flag numbers in the reply that exceed known data bounds (4+ digit numbers only).

    FIX-1: calendar years (2000-2100) are excluded from the check.
    Without this, any date string in Mistral's response (e.g. '2026-05-09')
    would be flagged as an impossible minute value and trigger the fallback.
    """
    violations = []
    for n_str in re.findall(r'\b(\d{4,})\b', reply):
        n = int(n_str)
        # FIX-1: skip calendar years - they are dates, not minute counts
        if 2000 <= n <= 2100:
            continue
        if n > bounds["max_minutes"]:
            violations.append(
                f"[NUMERIC] {n} dépasse le maximum observé "
                f"({bounds['max_minutes']} min). Valeur potentiellement inventée."
            )
    return violations


# ---------------------------------------------------------------------------
# v5.0 - Post-LLM validator (replaces _validate_llm_output)
# ---------------------------------------------------------------------------

def _validate_llm_output_v2(
    reply: str,
    context: dict,
    lang: str,
) -> str:
    """
    Validate Mistral's improvement analysis against the SQL whitelists.
    Returns validated reply or deterministic fallback.

    Checks:
    1. Machine hallucination: machine cited not in known_machines
    2. Active-machine-called-inactive (FIX-Q line-by-line logic preserved)
    3. Forbidden action: recipe/loading/unloading duration modification recommended.
       FIX-v5.9-A: extended to loading/unloading time reduction phrases.
       FIX-v5.11: extended to recipe paraphrases ("ajustez les recettes",
       "minimiser le temps de setup", "améliorez les délais de chargement", etc.)
    4. Numeric plausibility: no number larger than data bounds (FIX-1: years excluded)
    5. Optimal-rerun: "relancer le solveur" when status = OPTIMAL
    6. Incompatible machine recommendation via scheduling OR maintenance verbs.
       FIX-v5.9-B: extended verb list to include réparer, remplacer, remettre, etc.
    7. Hardware capacity modification: "augmentez la capacité de production" etc.
       Machine CapaciteMax is a hardware constant — not a scheduling lever.
       FIX-v5.11: new check catches recommendation verbs near "capacité" / "débit".
       [OP hallucination check disabled — FIX-11: false positives on CPU machines.]

    FIX-4: each check logs which check number fired so root cause is visible
    in the FastAPI console without needing to parse violation strings.
    FIX-6: Check 2 de-duplicates violations per machine to avoid multiplying
    violation count from a single bad sentence.
    """
    meta           = context["_meta"]
    known_machines = set(meta["known_machines"])
    active_set     = set(meta["active_machines"])
    known_ops      = set(meta["planned_ops"])
    status         = context.get("status", {}).get("statut", "")

    violations = []

    # ── Check 1: Machine hallucination ──────────────────────────────────────
    machine_pattern = _build_machine_mention_pattern(known_machines)
    if machine_pattern:
        cited = set(machine_pattern.findall(reply))
        for name in cited:
            matched = [m for m in known_machines
                       if m.lower().startswith(name.lower().split()[0])]
            if not matched:
                violations.append(
                    f"[CHECK-1][HALLUCINATION] Machine non reconnue: '{name}'"
                )

    # ── Check 2: Active machine called inactive (FIX-Q + FIX-6) ────────────
    inactive_phrases = [
        "non activée", "pas activée", "n'est pas activée", "non utilisée",
        "n'est pas utilisée", "is not activated", "not activated",
        "is not used", "not used", "is inactive",
    ]
    # FIX-6: track which machines we've already flagged to avoid duplicates
    already_flagged_inactive: Set[str] = set()
    for line in reply.splitlines():
        ll = line.lower()
        for machine in active_set:
            if machine in already_flagged_inactive:
                continue
            if machine.lower() not in ll:
                continue
            for phrase in inactive_phrases:
                if phrase in ll:
                    violations.append(
                        f"[CHECK-2][HALLUCINATION] '{machine}' est active mais décrite "
                        f"comme inactive (ligne: {line.strip()[:120]})"
                    )
                    already_flagged_inactive.add(machine)
                    break

    # ── Check 3: Forbidden actions ───────────────────────────────────────────
    # HARD RULE 1: recipe durations (treatment, loading, unloading) are FIXED
    # by industrial specialists and must NEVER be recommended as levers.
    # This block catches both treatment-duration changes AND loading/unloading
    # time reductions, which are equally fixed by the recipe specification.
    #
    # FIX-v5.9-A: extended forbidden list to cover:
    #   - "réduire le(s) temps de chargement/déchargement" (RULE 1 violation)
    #   - "reduce/shorten loading/unloading time" (EN equivalent)
    # These were previously missed because FIX-13 only kept exact treatment-duration
    # phrases. Loading/unloading times are also recipe-fixed and equally forbidden.
    #
    # Still excluded (these ARE valid levers):
    #   - "réduire les temps morts" (idle/dead time between lots — not recipe-fixed)
    #   - "réduire les temps d'attente" (queue wait time — optimizer-controlled)
    #   - "réduire la fragmentation" / "réduire le nombre de lots" (valid sequencing lever)
    forbidden = [
        # Recipe treatment duration — always forbidden
        "modifier la recette",
        "ajustez les recettes",      # v5.11 Mistral paraphrase caught here
        "ajuster les recettes",
        "ajustez la recette",
        "ajuster la recette",
        "changer la durée de traitement",
        "réduire la durée de traitement",
        "augmenter la durée de traitement",
        "modifier le temps de traitement",
        "changer le temps de traitement",
        "réduire le temps de traitement",
        "minimiser le temps de setup",   # v5.11: "minimiser le temps de setup overhead"
        "minimiser les temps de setup",
        "minimiser le setup",
        "modify the recipe",
        "adjust the recipe",
        "change the treatment time",
        "reduce the treatment time",
        "change the recipe",
        "alter the recipe duration",
        "modify treatment duration",
        "shorten the recipe",
        # Loading/unloading time — also recipe-fixed, equally forbidden
        # FIX-v5.9-A: new entries
        "réduire le temps de chargement",
        "réduire les temps de chargement",
        "réduire le temps de déchargement",
        "réduire les temps de déchargement",
        "réduire le temps de dechargement",
        "réduire les temps de dechargement",
        "réduire les temps de chargement et déchargement",
        "réduire les temps de chargement et de déchargement",
        "améliorez les délais de chargement",   # v5.11: "améliorez-les si nécessaire"
        "améliorer les délais de chargement",
        "améliorez les délais de déchargement",
        "améliorer les délais de déchargement",
        "examinez les délais de chargement",    # v5.11: examination + improvement intent
        "examiner les délais de chargement",
        "reduce loading time",
        "reduce the loading time",
        "reduce unloading time",
        "reduce the unloading time",
        "shorten loading time",
        "shorten unloading time",
        "improve loading time",
        "improve unloading time",
        "improve the loading",
        "improve the unloading",
        "reduce setup time",       # "setup" in French context often = chargement+déchargement
        "réduire le setup",
        "optimiser le temps de chargement",
        "optimiser les temps de chargement",
        "ajustez les temps de chargement",
        "ajuster les temps de chargement",
        "réduire leur durée totale",
        "réduire la durée totale",
        "raccourcir le chargement",
        "raccourcir le déchargement",
        "raccourcir les temps de chargement",
        "raccourcir les temps de déchargement",
        "diminuer le temps de chargement",
        "diminuer le temps de déchargement",
        "minimize loading time",
        "decrease loading time",
        "decrease unloading time",
    ]
    rl = reply.lower()
    for phrase in forbidden:
        if phrase in rl:
            violations.append(
                f"[CHECK-3][FORBIDDEN_ACTION] Recommandation interdite: '{phrase}'"
            )

    # ── Check 7: Hardware capacity modification ──────────────────────────────
    # HARD RULE: machine physical capacity (CapaciteMax) is a hardware constant.
    # Recommending "augmenter la capacité de production" or "increase machine
    # capacity" is not a scheduling lever — it requires capital investment and
    # is out of scope for a production planner. Mistral v5.11 output showed
    # "Augmentez la capacité de production des machines Brongo 1 et Brongo 5"
    # which is both wrong (not a scheduling lever) and potentially misleading.
    # Note: "capacité machine" used as a data reference (e.g. "la capacité de
    # la machine est 100 pièces") is NOT a violation — only recommendation verbs
    # (augmenter, increase, agrandir, upgrade, expand) near "capacité" are caught.
    hardware_capacity_patterns = re.compile(
        r'\b(augment(?:ez|er|ons)?|accro(?:ître|issez|ître)|agrand(?:ir|issez|ir)?|'
        r'augment(?:er|ez)|am[eé]liorer?|increase|upgrade|expand|boost|improve)\b'
        r'.{0,40}'
        r'\b(capacit[eé]|throughput|d[eé]bit|output\s+rate|production\s+rate)\b',
        re.IGNORECASE | re.DOTALL,
    )
    hw_match = hardware_capacity_patterns.search(reply)
    if hw_match:
        snippet = hw_match.group(0).replace('\n', ' ')[:120]
        violations.append(
            f"[CHECK-7][HARDWARE_LEVER] Recommande de modifier la capacité physique "
            f"des machines (non-actionnable par ordonnancement): '{snippet}'"
        )

    # ── Check 4: Numeric plausibility (FIX-1: years skipped) ─────────────────
    bounds = _extract_numeric_bounds(context)
    numeric_violations = _validate_numeric_claims(reply, bounds)
    # Tag with check number for log clarity
    violations.extend(
        v.replace("[NUMERIC]", "[CHECK-4][NUMERIC]") for v in numeric_violations
    )

    # ── Check 5: Optimal + rerun ──────────────────────────────────────────────
    if status == "optimal":
        rerun_phrases = [
            "relancer le solveur", "rerun the solver", "re-run the solver",
            "augmenter le temps de calcul", "increase solve time",
            "increase the solve time", "relancer avec", "rerun with",
        ]
        for phrase in rerun_phrases:
            if phrase in rl:
                violations.append(
                    "[CHECK-5][RULE_VIOLATION] Recommande de relancer le solveur "
                    "alors que statut = OPTIMAL."
                )
                break

    # ── Check 6: Incompatible machine recommendation ─────────────────────────
    # FIX-11 kept for pure operation text-match (too many false positives).
    # FIX-14: NEW targeted check - detect when Mistral recommends a machine
    # that is in the incompatible_unused list. This is a direct hallucination:
    # the prompt explicitly listed these machines as FORBIDDEN.
    # We only fire on recommendation verbs near the machine name (not mere mentions).
    #
    # FIX-v5.9-B: Extended verb list to include maintenance/repair verbs.
    # Previous list only covered scheduling verbs (activer, utiliser, affecter...).
    # Mistral can recommend forbidden machines via "réparer", "remplacer", "mettre
    # en service", "remettre en marche" etc. — all equally forbidden because
    # MACHINE_INCOMPATIBLE machines don't support the planned operations regardless
    # of their mechanical state. Adding them here ensures Check-6 catches these.
    incompatible_machine_names: Set[str] = set(
        m["machine"] for m in context.get("_meta", {}).get("incompatible_unused", [])
    )
    if incompatible_machine_names:
        recommend_verbs = re.compile(
            r'\b(activer|activ|utiliser|utilise|réaffecter|affecter|ajouter|déplacer|'
            r'activate|use|assign|add|redirect|move|enable|'
            # FIX-v5.9-B: maintenance / repair verbs — new entries
            r'réparer|reparer|r[eé]parer|remplacer|remettre|mettre\s+en\s+service|'
            r'remettre\s+en\s+marche|remettre\s+en\s+état|r[eé]activer|'
            r'repair|replace|restore|service|put\s+back|bring\s+back|'
            r'recommission|reintroduce|reactivate)\b',
            re.IGNORECASE,
        )
        # FIX-v5.9-C: scan whole reply (verb and machine name often on different lines)
        if recommend_verbs.search(rl):
            for machine in incompatible_machine_names:
                if machine.lower() in rl:
                    violations.append(
                        f"[CHECK-6][INCOMPATIBLE_MACHINE] '{machine}' recommandée "
                        f"(incompatible avec les opérations planifiées)"
                    )

    # ── Result ────────────────────────────────────────────────────────────────
    if violations:
        # FIX-4: log each violation with its check number before falling back
        print(f"[VALIDATION] {len(violations)} violation(s) detected - "
              "returning deterministic fallback:")
        for v in violations:
            print(f"  {v}")
        return _deterministic_fallback(context, lang)

    print("[VALIDATION] LLM output passed all checks.")
    return reply


# ---------------------------------------------------------------------------
# v5.0 - Deterministic fallback (last-resort safety net)
# ---------------------------------------------------------------------------

def _deterministic_fallback(context: dict, lang: str) -> str:
    """
    Emergency fallback: return structured, actionable bullet facts without LLM reasoning.
    Used only when Mistral times out or validation fails.

    v5.9 improvements:
    - Structural delays are collapsed into a single grouped bullet instead of one
      per commande (avoids filling all 5 slots with identical boilerplate).
    - Recoverable delays each get their own bullet (actionable: parallelize).
    - Setup overhead bullet now explains the root cause (lot fragmentation x fixed
      recipe setup) and the lever (machine with higher capacity or parallelization)
      rather than just stating the time figure.
    - Bottleneck bullet kept; only top bottleneck reported.
    - Cap raised from 5 to 6 bullets to fit both delay summary + improvement levers.
    """
    fr = (lang == "fr")
    bullets = []

    # ── Delays ────────────────────────────────────────────────────────────────
    # Collapse all structural delays into ONE bullet — they all have the same
    # resolution (renegotiate / partial delivery) and repeating it per commande
    # wastes all 5 bullet slots without adding actionable information.
    structural_cmds   = []
    recoverable_items = []  # list of (cmd, days)
    for d in context.get("delays", []):
        cmd  = d["commande"]
        days = d["jours_retard"]
        if d["retard_structurel"]:
            structural_cmds.append(cmd)
        else:
            recoverable_items.append((cmd, days))

    if structural_cmds:
        cmd_list = ", ".join(structural_cmds)
        if fr:
            bullets.append(
                f"Retards structurels ({cmd_list}) : deadlines dépassées avant le debut du planning. "
                "Impossible a eliminer par ordonnancement — renégocier les délais ou proposer une livraison partielle."
            )
        else:
            bullets.append(
                f"Structural delays ({cmd_list}): deadlines were past before planning started. "
                "Cannot be resolved by scheduling — renegotiate deadlines or offer partial delivery."
            )

    for cmd, days in recoverable_items:
        if fr:
            bullets.append(
                f"Retard recuperable - commande {cmd} : {days}j. "
                "Activer des machines compatibles supplementaires pour paralleliser et rattraper le retard."
            )
        else:
            bullets.append(
                f"Recoverable delay - order {cmd}: {days}d. "
                "Activate additional compatible machines to parallelize and recover the delay."
            )

    # ── Compatible unused machines ─────────────────────────────────────────────
    # FIX-20: use compatible_unused - unused_machines includes incompatible machines too
    meta = context.get("_meta", {})
    for _mu in meta.get("compatible_unused", []):
        _name = _mu["machine"] if isinstance(_mu, dict) else _mu
        _ops  = ", ".join(_mu.get("compatible_with", [])) if isinstance(_mu, dict) else ""
        _ops_str = f" (compatible : {_ops})" if _ops else ""
        if fr:
            bullets.append(
                f"Activer {_name}{_ops_str} : machine fonctionnelle non utilisee dans ce planning. "
                "L'ajouter au parallelisme reduit le makespan proportionnellement au nombre de machines actives."
            )
        else:
            bullets.append(
                f"Activate {_name}{_ops_str}: functional machine unused in this planning. "
                "Adding it to parallel processing reduces makespan proportionally to active machine count."
            )

    # ── Overloaded machines ────────────────────────────────────────────────────
    for m in context.get("machines", []):
        if m.get("etat") == "SURCHARGE":
            pct  = m.get("taux_charge_pct", "?")
            mach = m["machine"]
            if fr:
                bullets.append(
                    f"Machine surchargee : {mach} a {pct}% de charge. "
                    "Redistribuer des lots vers une machine compatible sous-utilisee pour equilibrer la charge."
                )
            else:
                bullets.append(
                    f"Overloaded machine: {mach} at {pct}% load. "
                    "Redistribute batches to a compatible underutilised machine to balance the load."
                )

    # ── Setup overhead ─────────────────────────────────────────────────────────
    # v5.9: bullet now explains the root cause and names the lever to pull,
    # not just the time figure. The recipe durations (chargement/dechargement)
    # are fixed — the lever is reducing the NUMBER of lots by using a machine
    # with higher capacity, which cuts total setup repetitions.
    # v5.10: lever depends on the binding constraint:
    #   binding="machine"  -> machine cap < recipe lot size  -> higher-cap machine IS valid
    #   binding="recette"  -> recipe lot size <= machine cap -> lot count is recipe-fixed,
    #                         higher-cap machine does nothing; only lever is parallelization
    setup_sorted = sorted(
        context.get("setup_overhead", []),
        key=lambda s: s.get("gaspillage_total_min", 0),
        reverse=True,
    )
    # Map (op, machine) -> binding constraint from fragmentation facts
    _frag_binding: dict = {}
    for f in context.get("fragmentation", []):
        _frag_binding[(f["operation"], f["machine"])] = f.get("binding_constraint", "recette")

    for s in setup_sorted[:2]:   # report at most 2 worst operations
        op      = s["operation"]
        machine = s["machine"]
        ratio   = s["ratio_setup_pct"]
        waste   = s["gaspillage_total_hhmm"]
        nb_lots = s.get("nb_lots_total", "?")
        charge  = s.get("temps_chargement_min", "?")
        dech    = s.get("temps_dechargement_min", "?")
        binding = _frag_binding.get((op, machine), "recette")
        if binding == "machine":
            # Machine capacity is the binding constraint → higher-cap machine IS valid
            if fr:
                lever_text = "Levier : affecter a une machine de plus grande capacite pour reduire le nombre de lots."
            else:
                lever_text = "Lever: assign to a higher-capacity machine to reduce lot count."
        else:
            # Recipe lot size is the binding constraint → machine-cap upgrade useless
            # The only valid lever is running more machines in parallel
            if fr:
                lever_text = (
                    "Levier : la taille de lot est fixee par la recette (machine deja adaptee) — "
                    "paralleliser l'operation sur plusieurs machines compatibles pour reduire le makespan."
                )
            else:
                lever_text = (
                    "Lever: batch size is recipe-fixed (machine already adequate) — "
                    "parallelize this operation across more compatible machines to reduce makespan."
                )
        if fr:
            bullets.append(
                f"Fragmentation {op} sur {machine} : {nb_lots} lots x ({charge}+{dech} min setup) "
                f"= {waste} improductif ({ratio}% du cycle). "
                f"{lever_text}"
            )
        else:
            bullets.append(
                f"Fragmentation {op} on {machine}: {nb_lots} lots x ({charge}+{dech} min setup) "
                f"= {waste} non-productive ({ratio}% of cycle). "
                f"{lever_text}"
            )

    # ── Top bottleneck ─────────────────────────────────────────────────────────
    for b in context.get("bottlenecks", []):
        if b.get("total_traitement_min", 0) > 0:
            if fr:
                bullets.append(
                    f"Goulot : {b['machine']} cumule {b['total_traitement_hhmm']} de traitement. "
                    "Redistribuer une partie de sa charge vers une machine compatible reduit le makespan."
                )
            else:
                bullets.append(
                    f"Bottleneck: {b['machine']} has {b['total_traitement_hhmm']} of cumulative processing. "
                    "Redistributing part of its load to a compatible machine reduces makespan."
                )
            break  # only report top bottleneck in fallback

    if not bullets:
        return (
            "Aucune amelioration detectable depuis les donnees disponibles."
            if fr else
            "No improvements detectable from the available data."
        )

    intro = (
        "Analyse du planning - points d'action prioritaires :"
        if fr else
        "Planning analysis - priority actions:"
    )
    numbered = "\n".join(f"{i+1}. {b}" for i, b in enumerate(bullets[:6]))
    return f"{intro}\n\n{numbered}"


# ===========================================================================
# UNCHANGED FROM v4.x - Hard facts & SQL context formatters
# ===========================================================================

def _build_hard_facts(db_rows: dict, lang: str) -> str:
    facts = []

    makespan_real = _compute_makespan_real(db_rows)
    if lang == "en":
        facts.append(
            f"MAKESPAN_REAL={makespan_real}  ← ONLY correct duration to quote. "
            "MakespanDays=0 = less than one full day, NOT zero hours."
        )
    else:
        facts.append(
            f"MAKESPAN_REAL={makespan_real}  ← SEULE durée correcte à citer. "
            "MakespanDays=0 = moins d'une journée complète, PAS zéro heure."
        )

    rows_a = db_rows.get("A", [])
    if rows_a:
        status = str(rows_a[0].get("Statut", "")).lower().strip()
        nb_cmd = rows_a[0].get("NombreCommandes", "?")
        nb_lig = rows_a[0].get("NombreLignes",    "?")
        dg     = rows_a[0].get("DateGeneration",  "?")

        if lang == "en":
            facts.append(f"STATUS={status.upper()} | ORDERS={nb_cmd} | GANTT_ROWS={nb_lig} | GENERATED={dg}")
            if status == "optimal":
                facts.append(
                    "STATUS=OPTIMAL means the solver's mathematical objective is proven optimal "
                    "(minimize tardiness + makespan). It does NOT mean no operational improvements exist."
                )
        else:
            facts.append(f"STATUT={status.upper()} | COMMANDES={nb_cmd} | LIGNES_GANTT={nb_lig} | GÉNÉRÉ={dg}")
            if status == "optimal":
                facts.append(
                    "STATUT=OPTIMAL signifie que l'objectif mathématique du solveur est prouvé optimal "
                    "(minimiser retard + makespan). Cela NE signifie PAS qu'aucune amélioration industrielle n'existe."
                )

    rows_b = db_rows.get("B", [])
    late_cmds = [str(r.get("NumeroCommande", "?")) for r in rows_b]
    if late_cmds:
        label = "LATE_ORDERS" if lang == "en" else "COMMANDES_EN_RETARD"
        facts.append(f"{label}={', '.join(late_cmds)}")
    else:
        facts.append("NO_LATE_ORDERS=true" if lang == "en" else "AUCUN_RETARD=true")

    rows_f = db_rows.get("F", [])
    if rows_f:
        unused_names = [_get_machine_name(r) or "?" for r in rows_f]
        label = "UNUSED_FUNCTIONAL_MACHINES" if lang == "en" else "MACHINES_FONCTIONNELLES_NON_UTILISEES"
        facts.append(f"{label}={', '.join(unused_names)}")

    rows_c = db_rows.get("C", [])
    zero_min = [
        _get_machine_name(r) or "?"
        for r in rows_c
        if _safe_int(r.get("MinutesPlanifiees", 0)) == 0
    ]
    if zero_min:
        label = "MACHINES_WITH_ZERO_PLANNED_MINUTES" if lang == "en" else "MACHINES_ZERO_MINUTES_PLANIFIEES"
        facts.append(f"{label}={', '.join(zero_min)} (assigned but no work recorded)")

    header = "[CERTIFIED FACTS - use only these values]\n" if lang == "en" else "[FAITS CERTIFIÉS - utiliser uniquement ces valeurs]\n"
    return header + "\n".join(f"  {f}" for f in facts) + "\n"


def _fmt_row_A(rows: list) -> str:
    if not rows:
        return "  (aucune donnée)\n"
    r = rows[0]
    return (
        f"  Statut: {r.get('Statut', '?')} | "
        f"Commandes: {r.get('NombreCommandes', '?')} | "
        f"Lignes Gantt: {r.get('NombreLignes', '?')} | "
        f"Généré: {r.get('DateGeneration', '?')} | "
        f"Début: {r.get('DateDebut', '?')}\n"
    )


def _fmt_row_B(rows: list, cap: int) -> str:
    if not rows:
        return "  ✅ Aucune commande en retard.\n"
    out = ""
    for r in rows[:cap]:
        cmd        = r.get("NumeroCommande", "?")
        urg        = r.get("Urgence", "?")
        deadline   = r.get("DateExport", "?")
        fin        = r.get("FinPlanifiee", "?")
        retard     = r.get("JoursRetard", "?")
        recette    = r.get("NomRecette", "?")
        structural = _safe_int(r.get("RetardStructurel", 0))

        out += f"  🔴 Commande {cmd} (urgence {urg}, recette {recette})\n"
        out += f"     Deadline: {deadline} | Fin planifiée: {fin} | Retard: {retard} jour(s)\n"
        if structural:
            out += (
                "  ⛔ RETARD STRUCTUREL : deadline dépassée avant le début du planning. "
                "Impossible à éliminer par ordonnancement.\n"
            )
    if len(rows) > cap:
        out += f"  ... ({len(rows) - cap} autres commandes en retard)\n"
    return out


def _fmt_row_C(rows: list, cap: int, unused_rows: list = None) -> str:
    if unused_rows is None:
        unused_rows = []
    used_names = {_get_machine_name(r) for r in rows}

    if not rows and not unused_rows:
        return "  (aucune donnée)\n"

    out = ""
    for r in rows[:cap]:
        nom     = _get_machine_name(r) or "?"
        capa    = r.get("CapaciteMax", "?")
        ops_raw = r.get("OperationsList") or r.get("Operations", "?")
        ops     = ", ".join(ops_raw) if isinstance(ops_raw, list) else str(ops_raw)
        nb      = r.get("NbCommandes", "?")
        mins    = r.get("MinutesPlanifiees", "?")
        etat    = r.get("Etat", "")
        if _safe_int(mins) == 0:
            out += f"  ⚪ {nom} (cap. {capa} | ops: {ops}) - ZÉRO MINUTE PLANIFIÉE\n"
        else:
            out += f"  🟢 {nom} (cap. {capa} | ops: {ops}) - {etat} | {mins} min | {nb} cmd(s)\n"
    if len(rows) > cap:
        out += f"  ... ({len(rows) - cap} autres)\n"

    for r in unused_rows:
        nom = _get_machine_name(r)
        if nom in used_names:
            continue
        capa    = r.get("CapaciteMax", "?")
        ops_raw = r.get("OperationsList") or r.get("Operations", "?")
        ops     = ", ".join(ops_raw) if isinstance(ops_raw, list) else str(ops_raw)
        out += f"  ⚪ {nom} (cap. {capa} | ops: {ops}) - NON UTILISÉE (fonctionnelle)\n"
    return out


def _fmt_row_D(rows: list, cap: int, rows_e: list = None) -> str:
    if rows_e is None:
        rows_e = []
    if rows:
        out = ""
        for r in rows[:cap]:
            cmd   = r.get("NumeroCommande", "?")
            op    = r.get("NomOperation",   "?")
            mach  = _get_machine_name(r) or "?"
            nb    = r.get("NbLots",         "?")
            cap_m = r.get("CapaciteMaxMachine", "?")
            lot_r = r.get("LotSizeRecette",     "?")
            out += f"  🟠 {cmd} - '{op}' sur {mach}: {nb} lots (recette {lot_r} FIXE vs cap. machine {cap_m})\n"
        if len(rows) > cap:
            out += f"  ... ({len(rows) - cap} autres)\n"
        return out

    agg: Dict[Tuple[str, str, str], int] = {}
    for r in rows_e:
        key = (
            str(r.get("NumeroCommande", "")),
            str(r.get("NomOperation", "")),
            _get_machine_name(r),
        )
        agg[key] = max(agg.get(key, 0), _safe_int(r.get("NbLots", 0)))
    high = {k: v for k, v in agg.items() if v > 5}
    if not high:
        return "  ✅ Aucune fragmentation excessive (tous les lots ≤ 5).\n"
    out = ""
    for (cmd, op, mach), nb in sorted(high.items(), key=lambda x: -x[1]):
        out += f"  🟠 {cmd} - '{op}' sur {mach}: {nb} lots (source: Query E)\n"
    return out


def _fmt_row_E(rows: list, cap: int,
               op_filter: Optional[str] = None,
               cmd_filter: Optional[str] = None) -> str:
    if not rows:
        return "  ⚠️  Aucune donnée de séquencement.\n"
    if op_filter:
        f_rows = [r for r in rows if op_filter in (r.get("NomOperation") or "").lower()]
        if f_rows:
            rows = f_rows
    if cmd_filter:
        f_rows = [r for r in rows if cmd_filter in (r.get("NumeroCommande") or "").lower()]
        if f_rows:
            rows = f_rows

    out = ""
    for r in rows[:cap] if cap else rows:
        cmd  = r.get("NumeroCommande", "?")
        op   = r.get("NomOperation",   "?")
        mach = _get_machine_name(r) or "?"
        lot  = r.get("LotIdx",  "?")
        nb   = r.get("NbLots",  "?")
        ds   = r.get("DateStart", "?")
        de   = r.get("DateEnd",   "?")
        dur  = r.get("DureeMinutes", "?")
        out += f"  {cmd} | {op} | {mach} | lot {lot}/{nb} | {ds}->{de} | {dur}min\n"
    if cap and len(rows) > cap:
        out += f"  ... ({len(rows) - cap} autres)\n"

    op_to_machs: Dict[str, Set[str]] = {}
    for r in rows:
        op_to_machs.setdefault(r.get("NomOperation", "?"), set()).add(_get_machine_name(r) or "?")
    out += "\n  ── RÉSUMÉ MACHINES PAR OPÉRATION ──\n"
    for op_n, machs in sorted(op_to_machs.items()):
        out += f"  • {op_n} -> {', '.join(sorted(machs))}\n"
    return out


def _fmt_row_F(rows: list, cap: int) -> str:
    if not rows:
        return "  ✅ Toutes les machines fonctionnelles sont utilisées.\n"
    out = ""
    for r in rows[:cap]:
        nom     = _get_machine_name(r) or "?"
        capa    = r.get("CapaciteMax", "?")
        ops_raw = r.get("OperationsList") or r.get("Operations", "?")
        ops     = ", ".join(ops_raw) if isinstance(ops_raw, list) else str(ops_raw)
        out += f"  ⚠️  {nom} - NON UTILISÉE | cap. {capa} | supporte: {ops}\n"
    if len(rows) > cap:
        out += f"  ... ({len(rows) - cap} autres)\n"
    return out


def _answer_machine_load_deterministically(db_rows: dict, lang: str) -> str:
    """
    FIX-23: Bypass Mistral entirely for machine-load questions.
    Builds a grounded, deterministic answer from Query C (used machines) and
    Query F (unused machines).  No LLM call -> no timeout, no hallucination.

    Returns a ready-to-display string in the user's language.
    """
    rows_c = db_rows.get("C", [])
    rows_f = db_rows.get("F", [])

    if not rows_c and not rows_f:
        return (
            "Aucune donnée de charge machines disponible pour ce planning."
            if lang == "fr" else
            "No machine load data available for this planning."
        )

    # Classify machines from Query C
    surcharges   = []
    sous_util    = []
    nominaux     = []

    for r in rows_c:
        nom  = _get_machine_name(r) or "?"
        etat = str(r.get("Etat", "NOMINAL")).upper()
        pct  = r.get("TauxChargePct", "?")
        mins = _safe_int(r.get("MinutesPlanifiees", 0))
        nb   = r.get("NbCommandes", "?")
        pct_str = f"{pct:.1f}%" if isinstance(pct, float) else f"{pct}%"
        entry = (nom, pct_str, mins, nb, etat)
        if "SURCHARG" in etat:
            surcharges.append(entry)
        elif "SOUS" in etat:
            sous_util.append(entry)
        else:
            nominaux.append(entry)

    lines = []

    if lang == "fr":
        if surcharges:
            lines.append("🔴 **Machines surchargées (taux > 90 %) :**")
            for nom, pct, mins, nb, _ in surcharges:
                lines.append(f"   • {nom} - {pct} ({_minutes_to_hhmm(mins)}, {nb} commande(s))")
        else:
            lines.append("✅ Aucune machine surchargée dans ce planning.")

        if sous_util:
            lines.append("\n🟡 **Machines sous-utilisées (taux < 20 %) :**")
            for nom, pct, mins, nb, _ in sous_util:
                lines.append(f"   • {nom} - {pct} ({_minutes_to_hhmm(mins)}, {nb} commande(s))")

        if nominaux:
            lines.append("\n🟢 **Machines à charge normale :**")
            for nom, pct, mins, nb, _ in nominaux:
                lines.append(f"   • {nom} - {pct} ({_minutes_to_hhmm(mins)}, {nb} commande(s))")

        if rows_f:
            lines.append("\n⚪ **Machines fonctionnelles non utilisées :**")
            for r in rows_f:
                nom     = _get_machine_name(r) or "?"
                capa    = r.get("CapaciteMax", "?")
                ops_raw = r.get("OperationsList") or r.get("Operations", "")
                ops     = ", ".join(ops_raw) if isinstance(ops_raw, list) else str(ops_raw)
                lines.append(f"   • {nom} (capacité {capa}, opérations : {ops})")
    else:
        if surcharges:
            lines.append("🔴 **Overloaded machines (load > 90%) :**")
            for nom, pct, mins, nb, _ in surcharges:
                lines.append(f"   • {nom} - {pct} ({_minutes_to_hhmm(mins)}, {nb} order(s))")
        else:
            lines.append("✅ No overloaded machines in this planning.")

        if sous_util:
            lines.append("\n🟡 **Underutilised machines (load < 20%) :**")
            for nom, pct, mins, nb, _ in sous_util:
                lines.append(f"   • {nom} - {pct} ({_minutes_to_hhmm(mins)}, {nb} order(s))")

        if nominaux:
            lines.append("\n🟢 **Normally loaded machines :**")
            for nom, pct, mins, nb, _ in nominaux:
                lines.append(f"   • {nom} - {pct} ({_minutes_to_hhmm(mins)}, {nb} order(s))")

        if rows_f:
            lines.append("\n⚪ **Functional but unused machines :**")
            for r in rows_f:
                nom     = _get_machine_name(r) or "?"
                capa    = r.get("CapaciteMax", "?")
                ops_raw = r.get("OperationsList") or r.get("Operations", "")
                ops     = ", ".join(ops_raw) if isinstance(ops_raw, list) else str(ops_raw)
                lines.append(f"   • {nom} (capacity {capa}, operations: {ops})")

    # Add the most overloaded machine highlight at the top if any
    if surcharges:
        top = surcharges[0]
        if lang == "fr":
            highlight = (
                f"La machine la plus surchargée est **{top[0]}** avec un taux de charge "
                f"de {top[1]} ({top[3]} commande(s) planifiée(s)).\n\n"
            )
        else:
            highlight = (
                f"The most overloaded machine is **{top[0]}** with a load rate of "
                f"{top[1]} ({top[3]} planned order(s)).\n\n"
            )
        return highlight + "\n".join(lines)

    return "\n".join(lines)


async def _build_sql_context(planning_id: int, db_rows: dict, question: str) -> str:
    # FIX-G: improvement path must never reach here
    qtype = _classify_question(question)
    if qtype == _QTYPE_IMPROVEMENT:
        raise RuntimeError(
            "_build_sql_context called for IMPROVEMENT question - routing bug. "
            "Improvement questions must go through _build_improvement_context()."
        )

    if not db_rows:
        return "⚠️  AUCUNE DONNÉE SQL REÇUE. Impossible d'analyser ce planning.\n"

    op_filter  = None
    cmd_filter = None
    m = re.search(r'\bcmd\s*(\w+)', question.lower())
    if m:
        cmd_filter = "cmd" + m.group(1)
    for op in ("poudre", "javellisation", "stonage", "lavage", "rinçage", "essorage", "séchage", "finition", "trempage"):
        if op in question.lower():
            op_filter = op
            break

    makespan_line = _compute_makespan_real(db_rows)
    sections = [
        f"[MAKESPAN RÉEL PLANNING #{planning_id}]\n"
        f"  Durée totale : {makespan_line}\n"
    ]

    if qtype == _QTYPE_MACHINE_LOAD:
        rows_c = db_rows.get("C", [])
        rows_f = db_rows.get("F", [])
        lines  = []
        used   = set()
        for r in rows_c:
            nom  = _get_machine_name(r) or "?"
            used.add(nom)
            etat = r.get("Etat", "NOMINAL")
            pct  = r.get("TauxChargePct", "?")
            mins = r.get("MinutesPlanifiees", "?")
            nb   = r.get("NbCommandes", "?")
            icon = "🔴" if etat == "SURCHARGE" else ("🟡" if "SOUS" in str(etat) else "🟢")
            lines.append(f"  {icon} {nom} - {etat} | {pct}% | {mins} min | {nb} cmd(s)")
        for r in rows_f:
            nom = _get_machine_name(r)
            if nom in used:
                continue
            capa    = r.get("CapaciteMax", "?")
            ops_raw = r.get("OperationsList") or r.get("Operations", "?")
            ops     = ", ".join(ops_raw) if isinstance(ops_raw, list) else str(ops_raw)
            lines.append(f"  ⚪ {nom} - NON UTILISÉE | cap. {capa} | ops: {ops}")
        sections.append("[CHARGE MACHINES]\n" + "\n".join(lines) + "\n")

        planned_ops = _build_known_operations(db_rows)
        if planned_ops:
            sections.append(f"[OPÉRATIONS PLANIFIÉES]\n  {', '.join(sorted(planned_ops))}\n")
        return "\n".join(sections)

    if qtype == _QTYPE_LOOKUP:
        rows_e = db_rows.get("E", [])
        op_to_machs: Dict[str, Set[str]] = {}
        for r in rows_e:
            op_to_machs.setdefault(r.get("NomOperation", "?"), set()).add(_get_machine_name(r) or "?")
        summary = "\n".join(
            f"  • {op} -> {', '.join(sorted(machs))}"
            for op, machs in sorted(op_to_machs.items())
        ) if op_to_machs else "  (aucune donnée)"
        sections.append(f"[MACHINES PAR OPÉRATION]\n{summary}\n")
        return "\n".join(sections)

    sections.append(f"[INFO PLANNING #{planning_id}]\n" + _fmt_row_A(db_rows.get("A", [])))
    sections.append("[RETARDS]\n" + _fmt_row_B(db_rows.get("B", []), ROW_CAPS["B"]))
    sections.append("[MACHINES]\n" + _fmt_row_C(db_rows.get("C", []), ROW_CAPS["C"], unused_rows=db_rows.get("F", [])))
    sections.append("[FRAGMENTATION]\n" + _fmt_row_D(db_rows.get("D", []), ROW_CAPS["D"], rows_e=db_rows.get("E", [])))

    rows_e = db_rows.get("E", [])
    if qtype == _QTYPE_SEQUENCE:
        e_cap = min(30, ROW_CAPS["E"])
        sections.append("[DÉTAIL OPÉRATIONS]\n" + _fmt_row_E(rows_e, e_cap, op_filter=op_filter, cmd_filter=cmd_filter))
    elif qtype == _QTYPE_SUMMARY:
        sections.append("[DÉTAIL OPÉRATIONS - résumé]\n" + _fmt_row_E(rows_e, 0, op_filter=op_filter, cmd_filter=cmd_filter))
    else:
        sections.append("[DÉTAIL OPÉRATIONS]\n  (non inclus pour cette question)\n")

    sections.append("[MACHINES NON UTILISÉES]\n" + _fmt_row_F(db_rows.get("F", []), ROW_CAPS["F"]))
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Ollama call helpers
# ---------------------------------------------------------------------------

async def _call_ollama(messages: list, options: dict) -> str:
    """Standard Ollama call - uses OLLAMA_TIMEOUT (300s, FIX-23)."""
    payload = {
        "model":      LLM_MODEL,
        "messages":   messages,
        "stream":     False,
        "keep_alive": "10m",
        "options":    options,
    }
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


async def _call_ollama_improvement(messages: list, options: dict) -> str:
    """
    FIX-3: Dedicated Ollama call for the improvement path.
    Uses OLLAMA_TIMEOUT_IMPROVEMENT (300s) because the improvement prompt is
    much larger than analysis prompts (system + JSON facts + CP-SAT rules)
    and requires num_predict=600, which takes longer on CPU-only machines.
    """
    payload = {
        "model":      LLM_MODEL,
        "messages":   messages,
        "stream":     False,
        "keep_alive": "10m",
        "options":    options,
    }
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_IMPROVEMENT) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


async def _call_ollama_streaming(messages: list, options: dict, timeout: int = OLLAMA_TIMEOUT) -> str:
    payload = {
        "model":      LLM_MODEL,
        "messages":   messages,
        "stream":     True,
        "keep_alive": "10m",
        "options":    options,
    }
    parts = []
    # FIX-17: per-phase Timeout - connect/write fail fast; read gets the full budget
    _ht = httpx.Timeout(connect=10.0, write=30.0, read=float(timeout), pool=10.0)
    async with httpx.AsyncClient(timeout=_ht) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            parts.append(token)
                    except json.JSONDecodeError:
                        pass
    return "".join(parts)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """\
You are a production scheduling assistant for the Micwic denim washing workshop.
Answer ONLY using values from [CERTIFIED FACTS] and [PLANNING DATA] sections.
ABSOLUTE RULES - violations are caught by post-validation:
1. NEVER invent machine names, order numbers, operation names, durations, or lot counts.
2. If a fact is absent from the data, say exactly: "Cette information n'est pas disponible." or "This information is not available." Do NOT infer from general knowledge.
3. NEVER mention a machine for an operation unless that machine+operation pair appears explicitly in the data.
4. NEVER invent late orders, delays, or operation types not present in [RETARDS] or [MACHINES NON UTILISÉES].
5. Speak like a production manager: no SQL column names, no code formulas.
Answer in the same language as the question. Be concise and direct.
"""


# ---------------------------------------------------------------------------
# Core analyze function
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# v5.8: Data table builder + Mistral primary reasoner
# ---------------------------------------------------------------------------
# Python extracts 100% accurate raw facts from SQL into a key=value data table.
# Mistral reads the data table and composes its own analysis from scratch —
# it is NOT given pre-formatted prose to reformat.
# This ensures Mistral actually reasons (prioritises, frames, phrases) rather
# than copying. Fallback = _deterministic_fallback(context) on timeout or error.
# ---------------------------------------------------------------------------

def _build_data_table(context: dict, lang: str) -> str:
    """
    Build a structured raw data table from SQL facts.

    ARCHITECTURE NOTE (v5.8):
    Previous versions (_build_grounded_analysis) produced fully-formed prose bullets,
    which Mistral simply copied. This function instead outputs raw key=value facts —
    the INPUT for Mistral's reasoning, not the output. Mistral reads these facts and
    composes its own analysis from scratch, producing varied, natural language.

    The table is intentionally NOT prose — it is structured data:
    machine names, percentages, durations, delay flags. Mistral must do the work of
    turning these into insights. Python's job is to ensure every number is accurate.
    """
    meta   = context.get("_meta", {})
    status = context.get("status", {})
    fr     = (lang == "fr")

    lines = []

    # ── Planning overview ──────────────────────────────────────────────────
    solver_status = status.get("statut", "?").upper()
    makespan      = status.get("makespan_reel", "?")
    nb_cmd        = status.get("nb_commandes", "?")
    lines.append(f"STATUT={solver_status} | MAKESPAN={makespan} | COMMANDES={nb_cmd}")

    planned_ops = sorted(meta.get("planned_ops", []))
    if planned_ops:
        lines.append(f"OPERATIONS_PLANIFIEES={', '.join(planned_ops)}")

    # ── Delays (Query B) ───────────────────────────────────────────────────
    for d in context.get("delays", []):
        cmd     = d["commande"]
        days    = d["jours_retard"]
        export  = d.get("date_export", "?")
        struct  = d["retard_structurel"]
        kind    = "STRUCTUREL" if struct else "RECUPERABLE"
        lines.append(f"RETARD type={kind} commande={cmd} export={export} jours_retard={days}")

    # ── Active machines (Query C) ──────────────────────────────────────────
    for m in context.get("machines", []):
        if m.get("type") != "machine_active":
            continue
        name  = m["machine"]
        pct   = m.get("taux_charge_pct", "?")
        mins  = m.get("minutes_planifiees", "?")
        etat  = m.get("etat", "NOMINAL")
        ops   = ",".join(m.get("operations", []))
        nb    = m.get("nb_commandes", "?")
        lines.append(
            f"MACHINE_ACTIVE nom={name} charge={pct}% etat={etat} "
            f"minutes={mins} commandes={nb} ops=[{ops}]"
        )

    # ── Compatible unused machines (Query F filtered) ──────────────────────
    for _m in meta.get("compatible_unused", []):
        name     = _m["machine"] if isinstance(_m, dict) else _m
        ops      = ",".join(_m.get("operations", [])) if isinstance(_m, dict) else ""
        compat   = ",".join(_m.get("compatible_with", [])) if isinstance(_m, dict) else ""
        lines.append(
            f"MACHINE_NON_UTILISEE nom={name} ops=[{ops}] compatible_avec=[{compat}]"
        )

    # ── Incompatible unused machines — Mistral must NEVER recommend these ──
    for _m in meta.get("incompatible_unused", []):
        name = _m["machine"] if isinstance(_m, dict) else _m
        ops  = ",".join(_m.get("operations", [])) if isinstance(_m, dict) else ""
        lines.append(f"MACHINE_INCOMPATIBLE nom={name} ops=[{ops}] -> NE PAS RECOMMANDER")

    # ── Bottlenecks (Query E aggregated) ──────────────────────────────────
    bottlenecks = sorted(
        context.get("bottlenecks", []),
        key=lambda b: b.get("total_traitement_min", 0),
        reverse=True
    )
    for b in bottlenecks[:3]:
        name  = b["machine"]
        hhmm  = b["total_traitement_hhmm"]
        mins  = b.get("total_traitement_min", 0)
        lines.append(f"GOULOT nom={name} traitement_total={hhmm} ({mins} min)")

    # ── Fragmentation (Query D) ────────────────────────────────────────────
    # Group identical fragmentation entries (same op + lots) by machine so Mistral
    # sees ONE combined line instead of N duplicates that waste output slots.
    # Each group also emits the binding_constraint so Mistral knows whether the
    # machine-capacity lever is valid (binding="machine") or not (binding="recette").
    _frag_groups: dict = {}  # key=(op, nb_lots, lot_sz, cap_mach, binding) -> list of machines
    for f in context.get("fragmentation", []):
        _key = (
            f["operation"],
            f["nb_lots"],
            f.get("lot_size_recette", "?"),
            f.get("cap_machine", "?"),
            f.get("binding_constraint", "recette"),
        )
        _frag_groups.setdefault(_key, []).append(f["machine"])
    for (op, lots, lot_sz, cap_m, binding), machs in _frag_groups.items():
        mach_str = ", ".join(sorted(set(machs)))
        # binding="machine" -> higher-capacity machine WILL reduce lot count (valid lever)
        # binding="recette" -> recipe lot size ≤ machine cap -> lever invalid, parallelization only
        lever_note = (
            "levier=machine-plus-grande-capacite-valide"
            if binding == "machine" else
            "levier=machine-capacite-INVALIDE(recette-fixe-nb-lots) levier-valide=paralleliser-sur-plus-de-machines"
        )
        if len(machs) > 1:
            lines.append(
                f"FRAGMENTATION op={op} machines=[{mach_str}] "
                f"nb_lots={lots} lot_recette={lot_sz}(fixe) cap_machine={cap_m} "
                f"contrainte_liante={binding} {lever_note} "
                f"[{len(machs)} machines concernées]"
            )
        else:
            lines.append(
                f"FRAGMENTATION op={op} machine={mach_str} "
                f"nb_lots={lots} lot_recette={lot_sz}(fixe) cap_machine={cap_m} "
                f"contrainte_liante={binding} {lever_note}"
            )

    # ── Setup overhead (Query E - top 2 worst) ─────────────────────────────
    setup_sorted = sorted(
        context.get("setup_overhead", []),
        key=lambda s: s.get("gaspillage_total_min", 0),
        reverse=True
    )
    for s in setup_sorted[:2]:
        op      = s["operation"]
        machine = s["machine"]
        ratio   = s["ratio_setup_pct"]
        waste   = s["gaspillage_total_hhmm"]
        lots    = s.get("nb_lots_total", "?")
        charge  = s.get("temps_chargement_min", "?")
        dech    = s.get("temps_dechargement_min", "?")
        lines.append(
            f"SETUP_OVERHEAD op={op} machine={machine} ratio_setup={ratio}% "
            f"gaspillage={waste} sur {lots} lots "
            f"(chargement={charge}min dechargement={dech}min - FIXES PAR RECETTE)"
        )

    return "\n".join(lines)


async def _reason_with_mistral(data_table: str, context: dict, lang: str, question: str) -> str:
    """
    v5.9: Mistral reasons from raw data + explicit entity whitelist fences.

    v5.9 FIXES over v5.8:
    - FIX-A: Entity whitelist injected directly into user_prompt as a hard fence.
      Mistral may ONLY cite machines/commands/ops from these lists.
      This eliminates hallucination of Brongo 1 underutilized, Tupesa 3/4, etc.
    - FIX-B: SETUP_OVERHEAD lines in data_table now carry "INTOUCHABLE" tag.
      Mistral sees setup as a data observation, not a lever — can't suggest reducing it.
    - FIX-C: Output capped at exactly 3 points. 5-point output at ~50 words/point
      overflows 300 tokens → truncation. 3 × ~35 words = ~105 tokens, always fits.
    - FIX-D: Post-call validation via _validate_llm_output_v2() before returning.
      If validation fails (forbidden suggestion, hallucinated machine, etc.),
      _deterministic_fallback() is returned instead.

    v5.9.1: Added single retry at temperature=0 before falling back to deterministic.
      When validation fails on the first attempt (usually a forbidden setup-reduction
      or repair recommendation), a second call at temperature=0 (greedy decoding)
      is made. Greedy decoding forces Mistral to pick the most probable token at each
      step, which is almost always compliant given the explicit constraints in the prompt.
      If the retry also fails validation, _deterministic_fallback() is returned.
      This means the user gets a Mistral-generated response in the large majority of
      cases, and the deterministic fallback is truly a last resort.
    """
    fr = (lang == "fr")
    meta = context.get("_meta", {})

    # ── Entity whitelist fences (FIX-A) ────────────────────────────────────
    # Built from the same whitelists Python already validated → ground truth.
    known_machines_str  = ", ".join(sorted(meta.get("known_machines", [])))
    known_cmds_str      = ", ".join(sorted(meta.get("known_cmds", [])))
    known_ops_str       = ", ".join(sorted(meta.get("planned_ops", [])))
    active_machines_str = ", ".join(sorted(meta.get("active_machines", [])))
    incompatible_str    = ", ".join(
        m["machine"] if isinstance(m, dict) else m
        for m in meta.get("incompatible_unused", [])
    )
    compatible_str = "; ".join(
        f"{m['machine']} -> {', '.join(m.get('compatible_with', []))}"
        for m in meta.get("compatible_unused", [])
        if isinstance(m, dict)
    ) or "aucune"

    if fr:
        fence_block = (
            f"\n[LISTE EXACTE DES ENTITÉS - NE CITER QUE CES VALEURS]\n"
            f"MACHINES_ACTIVES={active_machines_str}\n"
            f"MACHINES_ACTIVABLES={compatible_str}\n"
            f"MACHINES_INTERDITES={incompatible_str} (ne jamais recommander, meme pour reparation)\n"
            f"COMMANDES={known_cmds_str}\n"
            f"OPERATIONS={known_ops_str}\n"
            f"Toute machine, commande ou opération absente de ces listes = HALLUCINATION INTERDITE.\n"
        )
        system_prompt = (
            "Tu es un expert en ordonnancement industriel pour un atelier de lavage denim.\n"
            "On te donne des données brutes SQL d'un planning CP-SAT. "
            "Produis un diagnostic en 3 points numérotés, chacun = 1 à 2 phrases.\n\n"
            "RÈGLES ABSOLUES :\n"
            "- Ne citer QUE les entités de [LISTE EXACTE DES ENTITÉS].\n"
            "- SETUP (chargement/déchargement) : durées FIXÉES par recette industrielle. "
            "Ne JAMAIS suggérer de les réduire, optimiser, modifier ou raccourcir. "
            "Pour réduire le nombre de lots: utiliser la contrainte_liante du FRAGMENTATION — "
            "si contrainte_liante=machine → proposer machine plus grande capacité; "
            "si contrainte_liante=recette → paralléliser sur plus de machines (capacité machine déjà suffisante).\n"
            "- Retards STRUCTUREL : ne peuvent pas être résolus par ordonnancement → renégocier ou livraison partielle.\n"
            "- Ne jamais recommander une MACHINE_INTERDITE (ni activer, ni réparer, ni remplacer).\n"
            "- Chaque point doit contenir un chiffre (%, minutes, jours).\n"
            "LANGUE : français uniquement."
        )
        user_prompt = (
            f"DONNÉES PLANNING :\n---\n{data_table}\n---\n"
            f"{fence_block}\n"
            f"QUESTION : {question}\n\n"
            f"Rédige exactement 3 points d'amélioration prioritaires. "
            f"1 à 2 phrases par point. Chiffré. Actionnable. Sans répétition."
        )
    else:
        fence_block = (
            f"\n[EXACT ENTITY LIST - ONLY CITE THESE VALUES]\n"
            f"ACTIVE_MACHINES={active_machines_str}\n"
            f"ACTIVATABLE_MACHINES={compatible_str}\n"
            f"FORBIDDEN_MACHINES={incompatible_str} (never recommend, not even for repair)\n"
            f"ORDERS={known_cmds_str}\n"
            f"OPERATIONS={known_ops_str}\n"
            f"Any machine, order, or operation absent from these lists = FORBIDDEN HALLUCINATION.\n"
        )
        system_prompt = (
            "You are an industrial scheduling expert for a denim washing workshop.\n"
            "You receive raw SQL data from a CP-SAT planning. "
            "Produce a diagnosis in exactly 3 numbered points, each 1-2 sentences.\n\n"
            "ABSOLUTE RULES:\n"
            "- Only cite entities from [EXACT ENTITY LIST].\n"
            "- SETUP (loading/unloading): durations are FIXED by industrial recipe. "
            "NEVER suggest reducing, optimising, modifying or shortening them. "
            "To reduce lot count: use binding_constraint from FRAGMENTATION — "
            "if binding_constraint=machine → suggest higher-capacity machine; "
            "if binding_constraint=recette → parallelize across more machines (machine capacity already adequate).\n"
            "- STRUCTURAL delays: cannot be resolved by scheduling → renegotiate or partial delivery.\n"
            "- Never recommend a FORBIDDEN_MACHINE (not even to activate, repair or replace).\n"
            "- Each point must contain a number (%, minutes, days).\n"
            "LANGUAGE: English only."
        )
        user_prompt = (
            f"PLANNING DATA:\n---\n{data_table}\n---\n"
            f"{fence_block}\n"
            f"QUESTION: {question}\n\n"
            f"Write exactly 3 priority improvement points. "
            f"1-2 sentences per point. Quantified. Actionable. No repetition."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    # Token budget (FIX-C):
    # System ~200 tokens + data_table ~180 tokens + fence ~80 tokens + instruction ~30 tokens
    # Total input ~490 tokens → num_ctx=1024 sufficient.
    # Output: 3 points × ~35 words = ~105 tokens → num_predict=150 is enough,
    # 180 gives headroom for slightly longer sentences without overflow risk.
    analyst_options = {
        "num_predict": 350,
        "num_ctx":     1024,
        "temperature": 0.25,
        "top_p":       0.9,
    }
    analyst_options_retry = {
        "num_predict": 350,
        "num_ctx":     1024,
        "temperature": 0.0,   # greedy — most-probable token, highest compliance rate
        "top_p":       1.0,
    }

    # ── FIX: SHARED DEADLINE ACROSS BOTH RETRY ATTEMPTS ────────────────────────────
    # Previous buggy code: two independent 300s timeouts = 0–600s worst case
    # But .NET HttpClient has only 300s (5 min) timeout → causes double-timeout bug.
    # 
    # Fix: Allocate a shared 280s budget (20s safety margin before .NET timeout).
    # Each attempt uses only the remaining time, preventing total > 280s.
    #
    import time
    attempt_start_time = time.time()
    deadline = attempt_start_time + 280.0  # Shared budget: 280s (20s before .NET's 300s)

    async def _attempt(options: dict, attempt_label: str) -> Optional[str]:
        """Single call attempt with remaining-time budget (FIX: shared deadline)."""
        try:
            now = time.time()
            remaining = deadline - now
            
            # Abort if less than 5s remaining (not enough for Mistral to answer)
            if remaining <= 5:
                print(f"[ANALYST] {attempt_label}: Skipped (only {remaining:.1f}s remaining before shared deadline)")
                return None
            
            # Use remaining time as timeout for THIS attempt
            # Leave 2s overhead for HTTP round-trip + async context switches
            # Cap at 200s per attempt to ensure attempt-2 gets a fair chance
            timeout_for_attempt = int(min(remaining - 2, 200))
            
            t0    = time.time()
            reply = await _call_ollama_streaming(messages, options, timeout=timeout_for_attempt)
            elapsed = time.time() - t0
            n_words = len(reply.split())
            total_elapsed = time.time() - attempt_start_time
            print(f"[ANALYST] {attempt_label}: Mistral answered in {elapsed:.1f}s - ~{n_words} words (total: {total_elapsed:.1f}s)")

            if n_words < 18:
                print(f"[ANALYST] {attempt_label}: Reply too short ({n_words} words) - skipping")
                return None

            validated = _validate_llm_output_v2(reply, context, lang)
            # _validate_llm_output_v2 returns either the reply or the deterministic fallback.
            # We need to distinguish between the two so we know whether to retry.
            # Strategy: if the returned string starts with the deterministic fallback intro
            # prefix, it means validation failed. Otherwise the reply passed.
            fallback_intro_fr = "Analyse du planning - points d'action prioritaires"
            fallback_intro_en = "Planning analysis - priority actions"
            if validated.startswith(fallback_intro_fr) or validated.startswith(fallback_intro_en):
                print(f"[ANALYST] {attempt_label}: Validation failed (fallback returned) - will retry")
                return None  # signal: try again
            return validated

        except Exception as e:
            print(f"[ANALYST] {attempt_label}: Exception ({e})")
            return None

    # ── Fast-path: skip Mistral when incompatible machines are present ──────────
    # Empirical observation (v5.9 + v5.10 logs): when incompatible_unused is
    # non-empty, Mistral ALWAYS recommends those machines (Tupesa 1/2 etc.),
    # causing Check-6 to fire on every attempt.  Both attempts fail, and we
    # spend ~280s reaching the deterministic fallback anyway.
    #
    # Root cause: Mistral's attention on machine names overrides the fence block.
    # A 7B parameter model on CPU cannot reliably suppress named entities that
    # appear multiple times in the data table ("MACHINE_INCOMPATIBLE nom=Tupesa 2
    # -> NE PAS RECOMMANDER" is read as a mention, not a prohibition).
    #
    # Decision: if incompatible machines are listed, go directly to the
    # deterministic fallback.  It is now binding-constraint-aware (v5.10) and
    # produces correct, grounded output in <1ms instead of ~280s.
    #
    # The LLM path is still used when incompatible_unused is empty, because in
    # that case Check-6 can never fire, and Mistral produces better-phrased
    # (more varied, more natural) output than the deterministic fallback.
    if incompatible_str:
        print(
            f"[ANALYST] Incompatible machines present ({incompatible_str}) — "
            "skipping Mistral (Check-6 would fire). Using deterministic fallback directly."
        )
        return _deterministic_fallback(context, lang)

    # Attempt 1 — temperature=0.25
    result = await _attempt(analyst_options, "attempt-1")
    if result is not None:
        elapsed = time.time() - attempt_start_time
        print(f"[ANALYST] attempt-1 passed validation — returning Mistral response (total: {elapsed:.1f}s).")
        return result

    # Attempt 2 — temperature=0 (greedy, highest compliance rate)
    # Guard: only retry if we have sufficient time remaining
    remaining = deadline - time.time()
    if remaining > 10:  # Need >10s to make a meaningful attempt
        print(f"[ANALYST] Retrying at temperature=0 before falling back to deterministic... ({remaining:.1f}s remaining)")
        result = await _attempt(analyst_options_retry, "attempt-2")
        if result is not None:
            elapsed = time.time() - attempt_start_time
            print(f"[ANALYST] attempt-2 passed validation — returning Mistral response (total: {elapsed:.1f}s).")
            return result
    else:
        print(f"[ANALYST] Insufficient time remaining ({remaining:.1f}s) — skipping attempt-2, using deterministic fallback")

    # Both attempts failed (or timed out) — use deterministic fallback as true last resort
    elapsed = time.time() - attempt_start_time
    print(f"[ANALYST] Both attempts failed validation — using deterministic fallback (total: {elapsed:.1f}s).")
    return _deterministic_fallback(context, lang)


async def analyze(
    planning_id: int,
    question:    str,
    db_rows:     dict,
) -> str:
    """
    Main entry point - v5.1 Mistral-primary architecture.

    IMPROVEMENT questions (v5.0 new path, v5.1 fixes applied):
      _build_improvement_context() -> _build_improvement_prompt()
      -> _call_ollama_improvement() [300s timeout, FIX-3]
      -> _validate_llm_output_v2() [FIX-1, FIX-2, FIX-4, FIX-6]
      -> [on failure/timeout] _deterministic_fallback()

    All other questions (unchanged from v4.x):
      Hard facts + compact SQL context + tight instruction -> LLM answers.
    """

    db_rows = _assert_db_rows_safe(db_rows)

    if _is_gibberish(question):
        lang = _detect_language(question)
        return (
            "I did not understand your question. Could you please rephrase it?"
            if lang == "en" else
            "Je n'ai pas compris votre question. Pouvez-vous la reformuler ?"
        )

    lang = _detect_language(question)
    lang_directive = (
        "⚡ RESPOND ENTIRELY IN ENGLISH. No French."
        if lang == "en" else
        "⚡ RÉPONDRE ENTIÈREMENT EN FRANÇAIS. Pas d'anglais."
    )

    qtype = _classify_question(question)
    print(f"[RAG] qtype={qtype!r} lang={lang!r} planning_id={planning_id}")

    if _is_out_of_scope(question):
        print(f"[RAG] OUT-OF-SCOPE detected: {question!r}")
        if lang == "en":
            return (
                "I can only answer questions about this production planning: "
                "schedules, machine load, delays, makespan, fragmentation, and improvement suggestions. "
                "Your question seems to be about something outside the planning data I have access to."
            )
        return (
            "Je peux uniquement répondre aux questions sur ce planning de production : "
            "séquencement, charge machines, retards, makespan, fragmentation et pistes d'amélioration. "
            "Votre question porte sur un sujet qui n'est pas dans les données de planning dont je dispose."
        )

    # ═══════════════════════════════════════════════════════════════════════
    # MACHINE LOAD PATH - FIX-23: deterministic, no Mistral, no timeout
    # ═══════════════════════════════════════════════════════════════════════
    if qtype == _QTYPE_MACHINE_LOAD:
        print("[MACHINE_LOAD] Building deterministic answer from Query C/F...")
        answer = _answer_machine_load_deterministically(db_rows, lang)
        print(f"[MACHINE_LOAD] Done ({len(answer)} chars).")
        return answer

    # ═══════════════════════════════════════════════════════════════════════
    # IMPROVEMENT PATH - v5.1 Mistral-Primary
    # ═══════════════════════════════════════════════════════════════════════
    if qtype == _QTYPE_IMPROVEMENT:
        # v5.8 grounded-reasoning architecture:
        # Step 1 - Python extracts 100% accurate raw facts from SQL (no LLM, no prose).
        # Step 2 - Mistral reads raw facts and composes its own analysis from scratch.
        # If step 2 fails, _deterministic_fallback(context) is returned — already correct.
        print("[IMPROVEMENT] Building grounded analysis from SQL context...")
        ctx, refusal = _build_improvement_context(db_rows)

        if refusal:
            print(f"[RAG] IMPROVEMENT REFUSAL: {refusal}")
            return refusal

        data_table = _build_data_table(ctx, lang)
        print(f"[IMPROVEMENT] Data table built ({len(data_table)} chars)")

        print("[IMPROVEMENT] Calling Mistral for primary reasoning...")
        result = await _reason_with_mistral(data_table, ctx, lang, question)
        print("[RAG] IMPROVEMENT: analysis complete.")
        return result

    # ═══════════════════════════════════════════════════════════════════════
    # ALL OTHER PATHS - v5.8: streaming, question-aware, human token budget
    # ═══════════════════════════════════════════════════════════════════════
    # Previous versions used LLM_OPTIONS_LOOKUP (num_predict=80) and
    # LLM_OPTIONS_ANALYSIS (num_predict=120) with a non-streaming call.
    # At ~2s/token on CPU, 80 tokens = ~160s worst-case and answers were
    # always clipped mid-sentence. Streaming + 250 tokens gives natural,
    # complete answers and exits early once Mistral stops generating.

    hard_facts  = _build_hard_facts(db_rows, lang)
    sql_context = await _build_sql_context(planning_id, db_rows, question)

    # Token budget per question type:
    #   LOOKUP    — single factual answer, 1-2 sentences → 80 tokens
    #   MAKESPAN  — one number + explanation → 100 tokens
    #   SEQUENCE  — list of ops/lots, can be longer → 200 tokens
    #   SUMMARY   — planning overview → 200 tokens
    #   ANALYSIS  — multi-facet answer → 250 tokens
    if qtype == _QTYPE_LOOKUP:
        num_predict = 80
    elif qtype == _QTYPE_MAKESPAN:
        num_predict = 100
    elif qtype in (_QTYPE_SEQUENCE, _QTYPE_SUMMARY):
        num_predict = 200
    else:
        num_predict = 250

    stream_options = {
        "num_predict": num_predict,
        "num_ctx":     1024,
        "temperature": 0.2,   # slight warmth for natural phrasing; 0 = robotic
        "top_p":       0.9,
    }

    # System prompt: question-type-aware, plain-language, no column names
    if lang == "fr":
        system_prompt = (
            "Tu es un expert en ordonnancement industriel pour un atelier de lavage denim. "
            "Réponds UNIQUEMENT à partir des données fournies. "
            "Si une valeur est absente, dis-le en une phrase. "
            "Ne jamais inventer de machines, commandes, durées ou opérations. "
            "Parle comme un chef d'atelier : pas de noms de colonnes SQL "
            "(jamais QuantiteLot, CapaciteMax, NbLots, StartPM, EndPM, MakespanPM, TauxChargePct). "
            "Utilise : taille de lot, capacité machine, nombre de lots, durée totale, taux de charge. "
            "Langue : français uniquement. Réponse concise et directe."
        )
    else:
        system_prompt = (
            "You are an industrial scheduling expert for a denim washing workshop. "
            "Answer ONLY from the data provided. "
            "If a value is absent, say so in one sentence. "
            "Never invent machines, orders, durations, or operations. "
            "Speak like a production manager: no SQL column names "
            "(never QuantiteLot, CapaciteMax, NbLots, StartPM, EndPM, MakespanPM, TauxChargePct). "
            "Use: batch size, machine capacity, number of batches, total duration, utilisation rate. "
            "Language: English only. Answer concisely and directly."
        )

    user_prompt = (
        f"{hard_facts}\n\n"
        f"[DONNÉES PLANNING #{planning_id}]\n"
        f"{sql_context}\n\n"
        f"[QUESTION]\n{question}"
        if lang == "fr" else
        f"{hard_facts}\n\n"
        f"[PLANNING #{planning_id} DATA]\n"
        f"{sql_context}\n\n"
        f"[QUESTION]\n{question}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    try:
        import time
        t0 = time.time()
        reply = await _call_ollama_streaming(messages, stream_options, timeout=300)
        elapsed = time.time() - t0
        print(f"[{qtype.upper()}] Mistral answered in {elapsed:.1f}s - ~{len(reply.split())} words")
        if not reply.strip():
            return (
                "Aucune donnée disponible pour répondre à cette question."
                if lang == "fr" else
                "No data available to answer this question."
            )
        return reply.strip()
    except httpx.TimeoutException:
        if lang == "en":
            return (
                "Mistral took too long to respond (CPU-only mode). "
                "Try a more specific question, e.g. 'which machines are overloaded?'"
            )
        return (
            "Mistral a mis trop de temps à répondre (mode CPU). "
            "Essayez une question plus ciblée, ex : 'quelles machines sont surchargées ?'"
        )
    except Exception as e:
        return f"[RAG ERROR] {e}"


# ---------------------------------------------------------------------------
# Domain knowledge indexing (startup)
# ---------------------------------------------------------------------------

async def ensure_domain_knowledge_indexed():
    """
    Called at startup. Loads chunks from the DOCX knowledge base and indexes
    them into FAISS if not already done.
    """
    if _faiss_index.index is not None and _faiss_index.index.ntotal > 0:
        print(f"[RAG] FAISS already has {_faiss_index.index.ntotal} vectors - skipping re-index")
        return

    from rag.docx_chunker import load_chunks
    chunks = load_chunks()
    if not chunks:
        print("[RAG] No chunks loaded - FAISS index empty")
        return

    texts = [c["text"] if isinstance(c, dict) else c for c in chunks]

    print(f"[RAG] Embedding {len(texts)} chunks...")
    try:
        embeddings = await embed(texts)
        _faiss_index.build(texts, embeddings)
        if _FAISS_AVAILABLE:
            faiss.write_index(_faiss_index.index, str(FAISS_INDEX_PATH))
            with open(FAISS_META_PATH, "wb") as f:
                pickle.dump(_faiss_index.texts, f)
        print(f"[RAG] FAISS index built: {_faiss_index.index.ntotal} vectors")
    except Exception as e:
        print(f"[RAG] FAISS build failed: {e}")