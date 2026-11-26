"""Reranker cross-encoder : réordonnancement, construction du jeu de données, fine-tuning versionné."""
import argparse
import json
import random
from datetime import date
from functools import lru_cache

import ollama
from pydantic import BaseModel

from maintidoc.config import settings
from maintidoc.store import index

DATASET = settings.reranker_dir / "dataset.jsonl"


# ---------- Réordonnancement --------------------------------------------------

@lru_cache(maxsize=4)
def charger(chemin: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(chemin, max_length=256)


def modele_actif() -> str:
    """Le reranker adapté s'il a été entraîné, sinon le modèle de base."""
    return str(settings.reranker_adapted) if settings.reranker_adapted.exists() else settings.reranker_base


def reordonner(requete: str, hits: list[dict], modele: str | None = None, top_n: int | None = None) -> list[dict]:
    """Recalcule le score (requête, passage) avec le cross-encoder et trie."""
    if not hits:
        return hits
    scores = charger(modele or modele_actif()).predict([(requete, h["document"]) for h in hits])
    for h, s in zip(hits, scores):
        h["score"] = float(s)
    hits = sorted(hits, key=lambda h: -h["score"])
    return hits[:top_n] if top_n else hits


# ---------- Jeu de données (questions générées par le LLM, vérifiées) -------

class QuestionGeneree(BaseModel):
    question: str


class Verification(BaseModel):
    le_passage_repond: bool


PROMPT_QUESTION = """Tu prépares des données pour un moteur de recherche documentaire destiné à des techniciens de maintenance.
Passage extrait de "{pdf}" (page {page}) :
---
{passage}
---
Rédige UNE question en {langue} qu'un technicien pourrait poser et à laquelle ce passage répond précisément.
Ne mentionne jamais "le passage" ou "le texte"."""

PROMPT_VERIF = """Question : {question}
Passage : {passage}
Le passage contient-il réellement l'information qui répond à la question ? Réponds en JSON."""


def _llm(prompt: str, schema: type[BaseModel]) -> BaseModel:
    rep = ollama.Client(host=settings.ollama_url).chat(
        model=settings.llm_text, messages=[{"role": "user", "content": prompt}],
        format=schema.model_json_schema(), options={"temperature": 0.3, "num_ctx": 2048},
    )
    return schema.model_validate_json(rep.message.content)


def construire_dataset(n_questions: int, n_negatifs: int = 3, seed: int = 42) -> int:
    """Pour n passages : question générée -> vérification indépendante (seul filtre) -> négatifs difficiles (voisins CLIP)."""
    idx = index()
    rng = random.Random(seed)
    tout = idx.collection.get(where={"type": "texte"}, include=["documents", "metadatas"])
    textes = dict(zip(tout["ids"], tout["documents"]))
    passages = [{"id": i, "texte": d, **m} for i, d, m in zip(tout["ids"], tout["documents"], tout["metadatas"]) if len(d) >= 120]
    for p in passages:  # fenêtre = fragment + voisins (les fragments de 60 tokens sont trop courts seuls)
        prefixe, i = p["id"].rsplit(":", 1)
        p["fenetre"] = " ".join(textes[k] for k in (f"{prefixe}:{int(i) + d}" for d in (-1, 0, 1)) if k in textes)
    rng.shuffle(passages)
    # Équilibrage entre PDF : tour de rôle
    par_pdf: dict[str, list] = {}
    for p in passages:
        par_pdf.setdefault(p["pdf"], []).append(p)
    ordre = []
    while any(par_pdf.values()):
        for liste in par_pdf.values():
            if liste:
                ordre.append(liste.pop())
    settings.reranker_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with DATASET.open("w", encoding="utf-8") as f:
        for p in ordre:
            if n >= n_questions:
                break
            langue = "français" if rng.random() < 0.7 else "anglais"
            try:
                q = _llm(PROMPT_QUESTION.format(pdf=p["pdf"], page=p["page"], passage=p["fenetre"], langue=langue), QuestionGeneree)
                if len(q.question) < 10:
                    continue
                if not _llm(PROMPT_VERIF.format(question=q.question, passage=p["fenetre"]), Verification).le_passage_repond:
                    continue
            except Exception as e:
                print(f"  ! LLM : {e}", flush=True)
                continue
            voisins = idx.rechercher(idx.encoder_textes([q.question])[0], "admin", 12, "texte")
            negatifs = [v for v in voisins if not (v["metadata"]["pdf"] == p["pdf"] and v["metadata"]["page"] == p["page"])][:n_negatifs]
            ligne = {
                "question": q.question, "langue": langue, "split": "test" if rng.random() < 0.2 else "train", "verifie": True,
                "positif": {"id": p["id"], "texte": p["fenetre"], "pdf": p["pdf"], "page": p["page"]},
                "negatifs": [{"id": v["id"], "texte": v["document"], "pdf": v["metadata"]["pdf"], "page": v["metadata"]["page"]} for v in negatifs],
            }
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
            n += 1
            print(f"  [{n}/{n_questions}] {q.question[:80]}", flush=True)
    return n


def charger_dataset(split: str | None = None) -> list[dict]:
    lignes = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [l for l in lignes if l.get("verifie", True) and (split is None or l["split"] == split)]


# ---------- Fine-tuning ---------------------------------------------------------

def entrainer(epochs: int = 2, batch: int = 16, lr: float = 2e-5) -> dict:
    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    exemples = []
    for l in charger_dataset("train"):
        exemples.append(InputExample(texts=[l["question"], l["positif"]["texte"]], label=1.0))
        exemples += [InputExample(texts=[l["question"], n["texte"]], label=0.0) for n in l["negatifs"]]
    ce = CrossEncoder(settings.reranker_base, num_labels=1, max_length=256)
    dl = DataLoader(exemples, shuffle=True, batch_size=batch)
    ce.fit(train_dataloader=dl, epochs=epochs, warmup_steps=int(0.1 * len(dl) * epochs),
           optimizer_params={"lr": lr}, use_amp=torch.cuda.is_available(), show_progress_bar=True)
    ce.save(str(settings.reranker_adapted))
    version = {"version": f"{date.today().isoformat()}-{len(exemples)}p", "base": settings.reranker_base,
               "paires_entrainement": len(exemples), "epochs": epochs, "batch": batch, "lr": lr}
    (settings.reranker_adapted / "version.json").write_text(json.dumps(version, indent=2, ensure_ascii=False), encoding="utf-8")
    charger.cache_clear()
    return version


def main_dataset() -> None:
    ap = argparse.ArgumentParser(description="Générer le jeu de données du reranker (data/reranker/dataset.jsonl)")
    ap.add_argument("--n", type=int, default=300, help="nombre de questions (x (1 positif + 3 négatifs) = paires)")
    args = ap.parse_args()
    n = construire_dataset(args.n)
    print(f"{n} questions écrites dans {DATASET} (~{n * 4} paires)")


def main_train() -> None:
    ap = argparse.ArgumentParser(description="Fine-tuner le reranker sur dataset.jsonl")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    print(entrainer(**vars(ap.parse_args())))
