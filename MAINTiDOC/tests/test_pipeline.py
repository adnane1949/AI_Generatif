from open_clip.tokenizer import SimpleTokenizer

from maintidoc.config import settings
from maintidoc.evaluation import mrr, ndcg
from maintidoc.ingest import decouper
from maintidoc.rag import rrf


def test_decouper_respecte_la_limite_clip():
    tok = SimpleTokenizer()
    texte = ("La pompe centrifuge doit être arrêtée avant toute intervention. " * 40) + "\n\nVérifier la pression."
    fragments = decouper(texte, tok)
    assert len(fragments) > 1
    assert all(len(tok.encode(f)) <= settings.chunk_tokens for f in fragments)


def test_rrf_fusionne_et_classe():
    a = [{"id": "x", "document": "", "metadata": {}, "score": 0.9}, {"id": "y", "document": "", "metadata": {}, "score": 0.8}]
    b = [{"id": "y", "document": "", "metadata": {}, "score": 0.7}, {"id": "z", "document": "", "metadata": {}, "score": 0.6}]
    ordre = [h["id"] for h in rrf([a, b])]
    assert ordre[0] == "y" and set(ordre) == {"x", "y", "z"}


def test_metriques():
    assert ndcg([1, 0, 0, 0, 0]) == 1.0 and ndcg([0, 0, 0, 0, 0]) == 0.0
    assert mrr([0, 0, 1]) == 1 / 3 and mrr([0] * 10 + [1]) == 0.0
