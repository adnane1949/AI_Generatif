"""Pipeline d'ingestion : PDF -> texte par page (+ OCR si vide) + figures -> fragments -> index CLIP/ChromaDB."""
import argparse
import hashlib
import sys
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from maintidoc.config import settings
from maintidoc.store import charger_acl, index, metadonnees_acces

EXT_IMAGES = {"png", "jpg", "jpeg", "webp", "bmp", "tiff", "jp2"}


def decouper(texte: str, tokenizer) -> list[str]:
    """Fragments courts (<= chunk_tokens tokens CLIP) coupés sur des limites naturelles."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_tokens, chunk_overlap=settings.chunk_overlap,
        length_function=lambda t: len(tokenizer.encode(t)), separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(texte) if c.strip()]


def texte_page(page: pymupdf.Page) -> tuple[str, bool]:
    """Texte natif ; si la page est vide (scan), tentative d'OCR local Tesseract. Retourne (texte, ocr_utilise)."""
    texte = page.get_text("text", sort=True)
    if texte.strip():
        return texte, False
    try:
        return page.get_text("text", textpage=page.get_textpage_ocr(language="fra+eng", full=True)), True
    except Exception as e:  # Tesseract absent -> la page reste vide
        print(f"    OCR impossible page {page.number + 1} : {e}", flush=True)
        return "", False


def extraire_images(doc: pymupdf.Document, page: pymupdf.Page, dossier: Path) -> list[Path]:
    """Images intégrées de la page, filtrées (taille mini), sauvegardées sur disque."""
    chemins, vus = [], set()
    for info in page.get_images(full=True):
        xref = info[0]
        if xref in vus:
            continue
        vus.add(xref)
        img = doc.extract_image(xref)
        if not img or img["width"] < settings.min_image_px or img["height"] < settings.min_image_px:
            continue
        ext = img["ext"] if img["ext"] in EXT_IMAGES else "png"
        chemin = dossier / f"page_{page.number + 1:03d}_image_{xref}.{ext}"
        chemin.write_bytes(img["image"])
        chemins.append(chemin)
    return chemins


def ingerer_pdf(pdf: Path, acl: dict, force: bool = False) -> dict:
    idx = index()
    document_id = hashlib.sha256(pdf.read_bytes()).hexdigest()[:24]
    deja = idx.collection.get(where={"document_id": document_id}, limit=1)["ids"]
    if deja and not force:
        return {"pdf": pdf.name, "document_id": document_id, "statut": "déjà indexé"}

    # Une nouvelle version du même fichier remplace l'ancienne (pas de passages obsolètes)
    idx.collection.delete(where={"pdf": pdf.name})
    dossier = settings.extracted_dir / document_id
    (dossier / "images").mkdir(parents=True, exist_ok=True)
    base = {"document_id": document_id, "pdf": pdf.name, **metadonnees_acces(pdf.name, acl)}

    ids, docs, metas, embs, n_img, n_ocr = [], [], [], [], 0, 0
    with pymupdf.open(pdf) as doc:
        n_pages = len(doc)
        for page in doc:
            num = page.number + 1
            texte, ocr = texte_page(page)
            n_ocr += ocr
            (dossier / f"page_{num:03d}.txt").write_text(texte, encoding="utf-8")
            fragments = decouper(texte, idx.tokenizer)
            for i, frag in enumerate(fragments):
                ids.append(f"{document_id}:texte:{num}:{i}")
                docs.append(frag)
                metas.append({**base, "type": "texte", "page": num, "fragment": i, "ocr": ocr})
            if fragments:
                embs.extend(idx.encoder_textes(fragments))
            for chemin in extraire_images(doc, page, dossier / "images"):
                try:
                    embs.append(idx.encoder_image(chemin))
                except Exception as e:
                    print(f"    image ignorée {chemin.name} : {e}", flush=True)
                    continue
                ids.append(f"{document_id}:image:{chemin.name}")
                docs.append(f"Figure page {num} de {pdf.name}")
                metas.append({**base, "type": "image", "page": num, "fragment": 0, "ocr": False, "source": str(chemin.resolve())})
                n_img += 1

    for d in range(0, len(ids), 256):  # upsert par lots
        idx.collection.upsert(ids=ids[d:d + 256], documents=docs[d:d + 256], metadatas=metas[d:d + 256], embeddings=embs[d:d + 256])
    return {"pdf": pdf.name, "document_id": document_id, "pages": n_pages, "fragments": len(ids) - n_img,
            "images": n_img, "pages_ocr": n_ocr, "roles": [r for r in acl["roles"] if base[f"acces_{r}"]], "statut": "indexé"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Indexer les PDF de data/pdfs dans ChromaDB")
    ap.add_argument("pdfs", nargs="*", type=Path, help="fichiers précis (défaut : tout le dossier)")
    ap.add_argument("--force", action="store_true", help="réindexer même si déjà présent")
    args = ap.parse_args()
    acl = charger_acl()
    pdfs = args.pdfs or sorted(settings.pdfs_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"Aucun PDF dans {settings.pdfs_dir}")
    for pdf in pdfs:
        if pdf.name not in acl["documents"]:
            print(f"  ! {pdf.name} absent de {settings.acl_file} -> droits par défaut {acl['defaut']['roles']}", flush=True)
        print(ingerer_pdf(pdf, acl, args.force), flush=True)
    print(f"Total dans l'index : {index().collection.count()} éléments", flush=True)


if __name__ == "__main__":
    main()
