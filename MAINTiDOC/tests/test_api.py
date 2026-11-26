from fastapi.testclient import TestClient

from maintidoc import api
from maintidoc.schemas import Reponse, ReponseRAG, Source


def _faux_repondre(question, image, role, k, type_resultat):
    src = Source(reference="S1", id="d:texte:1:0", type="texte", pdf="a.pdf", page=1, score=0.9, extrait="passage")
    return ReponseRAG(question=question or "(photo)", reponse=Reponse(reponse=f"[{role}] ok", citations=["S1"]), sources=[src])


def test_ask_et_role(monkeypatch):
    monkeypatch.setattr(api.rag, "repondre", _faux_repondre)
    c = TestClient(api.app)
    r = c.post("/ask", json={"question": "test"}, headers={"X-Role": "superviseur"})
    assert r.status_code == 200 and r.json()["reponse"]["reponse"] == "[superviseur] ok"
    assert r.json()["sources"][0]["reference"] == "S1"


def test_role_inconnu_refuse(monkeypatch):
    monkeypatch.setattr(api.rag, "repondre", _faux_repondre)
    assert TestClient(api.app).post("/ask", json={"question": "test"}, headers={"X-Role": "pirate"}).status_code == 403


def test_analyze_exige_une_image(monkeypatch):
    monkeypatch.setattr(api.rag, "repondre", _faux_repondre)
    r = TestClient(api.app).post("/analyze", files={"image": ("x.txt", b"abc", "text/plain")})
    assert r.status_code == 415
