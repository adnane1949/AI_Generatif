"""Index vectoriel : droits d'accès (ACL), encodeur CLIP et collection ChromaDB."""
import json
from functools import lru_cache
from pathlib import Path

import chromadb
import torch
from PIL import Image

from maintidoc.config import settings


# ---------- Droits d'accès ----------------------------------------------------

def charger_acl() -> dict:
    """data/acl.json : {"roles": [...], "documents": {pdf: {famille, version, roles}}, "defaut": {...}}"""
    if not settings.acl_file.exists():
        return {"roles": ["admin"], "documents": {}, "defaut": {"famille": "inconnue", "version": "n/a", "roles": ["admin"]}}
    return json.loads(settings.acl_file.read_text(encoding="utf-8"))


def metadonnees_acces(nom_pdf: str, acl: dict) -> dict:
    """Famille, version et un booléen acces_<role> par rôle (Chroma ne filtre pas sur des listes)."""
    info = acl["documents"].get(nom_pdf, acl["defaut"])
    meta = {"famille": info.get("famille", "inconnue"), "version": str(info.get("version", "n/a"))}
    for role in acl["roles"]:
        meta[f"acces_{role}"] = role in info.get("roles", [])
    return meta


def filtre_where(role: str, type_resultat: str | None = None) -> dict:
    """Filtre ChromaDB combinant le rôle et, si demandé, le type (texte / image)."""
    clauses = [{f"acces_{role}": True}]
    if type_resultat:
        clauses.append({"type": type_resultat})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# ---------- Encodeur CLIP + ChromaDB -----------------------------------------

class Index:
    """Un seul espace vectoriel CLIP pour les passages de texte et les figures."""

    def __init__(self) -> None:
        import open_clip  # import lourd, gardé local

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model, pretrained=settings.clip_pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(settings.clip_model)
        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            settings.collection, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    @torch.inference_mode()
    def encoder_textes(self, textes: list[str]) -> list[list[float]]:
        v = self.model.encode_text(self.tokenizer(textes).to(self.device)).float()
        return (v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().tolist()

    @torch.inference_mode()
    def encoder_image(self, image: str | Path | Image.Image) -> list[float]:
        img = image if isinstance(image, Image.Image) else Image.open(image)
        t = self.preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        v = self.model.encode_image(t).float()
        return (v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu().tolist()[0]

    def rechercher(self, vecteur: list[float], role: str, k: int, type_resultat: str | None = None) -> list[dict]:
        """Retourne [{id, document, metadata, score}] triés par similarité cosinus décroissante."""
        if self.collection.count() == 0:
            return []
        res = self.collection.query(
            query_embeddings=[vecteur], n_results=min(k, self.collection.count()),
            where=filtre_where(role, type_resultat), include=["documents", "metadatas", "distances"],
        )
        return [
            {"id": i, "document": d, "metadata": m, "score": 1 - dist}
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def documents(self, role: str) -> list[dict]:
        """Liste des PDF accessibles au rôle (un enregistrement par PDF)."""
        res = self.collection.get(where=filtre_where(role), include=["metadatas"])
        vus = {}
        for m in res["metadatas"]:
            vus.setdefault(m["pdf"], {"pdf": m["pdf"], "famille": m["famille"], "version": m["version"], "document_id": m["document_id"]})
        return sorted(vus.values(), key=lambda d: d["pdf"])


@lru_cache(maxsize=1)
def index() -> Index:
    """Instance unique, chargée paresseusement (les modèles sont lourds)."""
    return Index()
