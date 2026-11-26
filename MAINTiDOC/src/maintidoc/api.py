"""API FastAPI : /ask (question), /analyze (photo), /documents, /image/{id}. Le rôle est transmis dans l'en-tête X-Role."""
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from maintidoc import rag
from maintidoc.schemas import ReponseRAG, RequeteQuestion
from maintidoc.store import charger_acl, filtre_where, index

app = FastAPI(title="MAINTiDOC API", version="0.2.0")


def verifier_role(x_role: Annotated[str, Header()] = "technicien") -> str:
    if x_role not in charger_acl()["roles"]:
        raise HTTPException(403, f"Rôle inconnu : {x_role}")
    return x_role


@app.get("/health")
def health() -> dict:
    return {"statut": "ok", "elements_indexes": index().collection.count(), "roles": charger_acl()["roles"]}


@app.get("/documents")
def documents(x_role: Annotated[str, Header()] = "technicien") -> list[dict]:
    return index().documents(verifier_role(x_role))


@app.post("/ask", response_model=ReponseRAG)
def ask(req: RequeteQuestion, x_role: Annotated[str, Header()] = "technicien") -> ReponseRAG:
    return rag.repondre(req.question, None, verifier_role(x_role), req.k, req.type_resultat)


@app.post("/analyze", response_model=ReponseRAG)
async def analyze(image: UploadFile = File(...), question: str = Form(""), k: int = Form(6),
                  type_resultat: str = Form("tous"), x_role: Annotated[str, Header()] = "technicien") -> ReponseRAG:
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(415, "Fichier image attendu")
    return rag.repondre(question or None, await image.read(), verifier_role(x_role), k, type_resultat)


@app.get("/image/{id}")
def image(id: str, x_role: Annotated[str, Header()] = "technicien") -> FileResponse:
    res = index().collection.get(ids=[id], where=filtre_where(verifier_role(x_role), "image"), include=["metadatas"])
    if not res["ids"]:
        raise HTTPException(404, "Image inconnue ou accès refusé")
    return FileResponse(res["metadatas"][0]["source"])


def main() -> None:
    uvicorn.run("maintidoc.api:app", host="0.0.0.0", port=8000)
