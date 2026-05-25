
import re
import time
import httpx
from typing import Optional, List

import chatbot.sql_fetcher as sf
from chatbot.rag.faiss_index import retrieve as faiss_retrieve

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "mistral"

# ── System prompt

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

# ── Question routing 

_PATTERNS = {
    "retard":        r"retard|en retard|late|delay|dépasse|dépassé|export",
    "charge":        r"charge|utilisa|load|surchargé|sous.utilis|bottleneck|goulot",
    "fragmentation": r"fragment|split|morcel|découp|fragmentation des lots",
    "commande":      r"commande|order|CMD|numéro",
    "operation":     r"opération|operation|étape|step",
    "machine":       r"machine|équipement|appareil",
    "makespan":      r"makespan|durée totale|total duration|combien de temps|temps total",
    "alerte":        r"alerte|alert|warning|problème|probleme",
    "résumé":        r"résumé|résume|summary|overview|bilan|rapport",
    "amélioration":  r"amélior|amelior|optim|improve|better|r[eé]duire.*makespan|makespan.*r[eé]duire|comment.*planif|make.*optimal|how.*improv|comment.*am[eé]lior|comment.*r[eé]duire|comment.*reduire",
    "panne":         r"panne|tombe.*en panne|en panne|breakdown|fail|hors.service|arrêt.*machine|machine.*arrêt|si.*tombe|impact.*machine|machine.*impact|affecté.*machine",
    "séquence":      r"apr[eè]s|avant|toujours.*apr[eè]s|apr[eè]s.*toujours|ordre.*op[eé]ration|op[eé]ration.*ordre|d'abord|ensuite|avant.*de|commence.*apr[eè]s|fin.*avant|séquence|sequen|chevauchement|overlap|en m[eê]me temps|simultan",
    "définition":    r"c'est quoi|qu'est.ce que|what is|what's|définition|expliqu|signif|veut dire|mean",
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
    _GENERIC = {
        "LES", "DES", "EST", "PAR", "SUR", "POUR", "QUE", "QUI", "OU", "ET",
        "UNE", "DU", "AU", "EN", "SE", "LA", "LE", "UN", "DE", "CE", "IL",
        "SQL", "RAG", "API", "OK",
    }
    pattern = r'\b([A-Z]{2,}[\-A-Z0-9]*[0-9][A-Z0-9\-]*)\b'
    candidates = [m for m in re.findall(pattern, question.upper()) if m not in _GENERIC]
    return candidates[0] if candidates else None


def _extract_machine_name(question: str) -> Optional[str]:
    match = re.search(r'\b(Brongo|Tupesa)\s*(\d+)\b', question, re.IGNORECASE)
    if match:
        return f"{match.group(1).capitalize()} {match.group(2)}"
    return None


# ── SQL context builder

def _build_sql_context(question: str, planning_id: Optional[int]) -> str:
    intents = _detect_intent(question)
    blocks = []

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
                blocks.append(sf.fetch_machine_impact(planning_id, machine_name))
            else:
                # Machine name not parseable — fall back to full load overview
                blocks.append(sf.fetch_machine_load(planning_id))

    if "amélioration" in intents:
        if planning_id:
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
            blocks.append(sf.fetch_operation_sequence(planning_id))

    if ("commande" in intents or "operation" in intents) and "panne" not in intents:
        if planning_id and "séquence" not in intents:
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


# ── num_ctx selection

def _num_ctx(intents: set) -> int:
   
    if intents & {"définition"} and not intents & {"amélioration", "fragmentation"}:
        return 1024
    if intents & {"panne"}:
        return 1536
    if intents & {"séquence"}:
        return 1024
    if intents & {"amélioration"}:
        return 1536
    if intents & {"commande", "operation"}:
        return 2048
    if intents & {"résumé"}:
        return 1024
    if intents & {"fragmentation", "recette"}:
        return 2048
    return 1536


def _temperature(intents: set) -> float:
    if intents & {"amélioration"}:
        return 0.4
    return 0.2

# ── Mistral streaming call 
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
    if intents & {"définition"} and not intents & {"amélioration", "fragmentation"}:
        return 150
    if intents & {"panne"}:
        return 400
    if intents & {"séquence"}:
        return 250
    if intents & {"amélioration"}:
        return 600
    if intents & {"résumé"}:
        return 200
    if intents & {"fragmentation"}:
        return 900
    if intents & {"recette"}:
        return 500
    if intents & {"alerte", "commande", "operation"}:
        return 400
    if intents & {"charge", "retard", "machine"}:
        return 300
    return 200

# ── Main analyze function 

def _build_amelioration_answer(planning_id: int, question: str) -> Optional[str]:
   
    MAX_SUGGESTIONS = 3

    raw = sf.fetch_valid_transfers(planning_id)
    if not raw or "Aucun transfert possible" in raw:
        return None

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

    if "amélioration" in intents and planning_id:
        t0 = time.time()
        amelio_result = _build_amelioration_answer(planning_id, question)
        print(f"[TIMING] SQL fetch (amélioration): {time.time() - t0:.2f}s")

        if amelio_result is not None:
            intro_prompt, bullet_block, footer, top = amelio_result

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