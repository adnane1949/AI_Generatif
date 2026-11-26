"""Rapport d'évaluation : reranker (nDCG@5, MRR@10, 3 configurations) et tests d'accès documentaire (ACL)."""
import json
import math
from datetime import date

from maintidoc import reranker
from maintidoc.config import settings
from maintidoc.store import charger_acl, index


def _fragment(id_: str) -> int:
    return int(id_.rsplit(":", 1)[1])


def pertinent(hit: dict, positif: dict) -> int:
    """Même PDF, même page et fragment voisin (les fragments se chevauchent)."""
    m = hit["metadata"]
    return int(m["pdf"] == positif["pdf"] and m["page"] == positif["page"] and abs(m["fragment"] - _fragment(positif["id"])) <= 1)


def ndcg(rels: list[int], k: int = 5) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, sum(rels))))
    return dcg / ideal if ideal else 0.0


def mrr(rels: list[int], k: int = 10) -> float:
    return next((1 / (i + 1) for i, r in enumerate(rels[:k]) if r), 0.0)


def evaluer_reranker() -> tuple[dict, int]:
    idx = index()
    tests = reranker.charger_dataset("test")
    configs = {"sans_reranker": None, "reranker_initial": settings.reranker_base}
    if settings.reranker_adapted.exists():
        configs["reranker_adapte"] = str(settings.reranker_adapted)
    acc = {c: {"nDCG@5": [], "MRR@10": []} for c in configs}
    for t in tests:
        candidats = idx.rechercher(idx.encoder_textes([t["question"]])[0], "admin", 20, "texte")
        for c, modele in configs.items():
            ordre = candidats if modele is None else reranker.reordonner(t["question"], [dict(h) for h in candidats], modele)
            rels = [pertinent(h, t["positif"]) for h in ordre]
            acc[c]["nDCG@5"].append(ndcg(rels))
            acc[c]["MRR@10"].append(mrr(rels))
    return {c: {m: round(sum(v) / max(len(v), 1), 4) for m, v in d.items()} for c, d in acc.items()}, len(tests)


def tester_acces(questions: list[str]) -> dict:
    """Pour chaque rôle : aucun résultat ne doit provenir d'un PDF non autorisé."""
    idx, acl = index(), charger_acl()
    rapport = {}
    for role in acl["roles"]:
        autorises = {pdf for pdf, info in acl["documents"].items() if role in info["roles"]}
        if role in acl["defaut"]["roles"]:
            autorises |= {d["pdf"] for d in idx.documents("admin")} - set(acl["documents"])
        violations, n = [], 0
        for q in questions:
            for h in idx.rechercher(idx.encoder_textes([q])[0], role, 20):
                n += 1
                if h["metadata"]["pdf"] not in autorises:
                    violations.append((q, h["metadata"]["pdf"]))
        rapport[role] = {"documents_visibles": sorted(d["pdf"] for d in idx.documents(role)), "resultats_controles": n, "violations": violations}
    return rapport


def main() -> None:
    settings.resultats_dir.mkdir(exist_ok=True)
    tests = reranker.charger_dataset("test") if reranker.DATASET.exists() else []
    questions = [t["question"] for t in tests[:10]] or ["procédure de maintenance", "sécurité", "schéma de câblage"]
    acces = tester_acces(questions)
    metriques, n = evaluer_reranker() if tests else ({}, 0)
    json.dump({"date": date.today().isoformat(), "reranker": metriques, "n_test": n, "acces": acces},
              (settings.resultats_dir / "rapport_evaluation.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)

    md = [f"# Rapport d'évaluation MAINTiDOC ({date.today().isoformat()})", "",
          "Résultats observés sur ce corpus et cette machine ; ce ne sont pas des garanties.", "",
          f"## Reranker ({n} questions de test, candidats = top-20 CLIP)", "", "| Configuration | nDCG@5 | MRR@10 |", "|---|---|---|"]
    md += [f"| {c} | {m['nDCG@5']} | {m['MRR@10']} |" for c, m in metriques.items()] or ["| (pas de dataset : lancer maintidoc-dataset) | - | - |"]
    md += ["", "## Tests d'accès (filtrage par rôle)", "", "| Rôle | PDF visibles | Résultats contrôlés | Violations |", "|---|---|---|---|"]
    md += [f"| {r} | {len(a['documents_visibles'])} | {a['resultats_controles']} | {len(a['violations'])} |" for r, a in acces.items()]
    (settings.resultats_dir / "rapport_evaluation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
