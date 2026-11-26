# MAINTiDOC — assistant documentaire multimodal pour la maintenance

Étapes 1 et 2 de la feuille de route (V1 assistant textuel, V2 assistant multimodal + reranker adapté).
Tout tourne en local : PyMuPDF, CLIP, ChromaDB, Ollama (Qwen2.5 / Qwen2.5-VL), FastAPI, Streamlit.
Les concepts et les choix d'implémentation sont expliqués dans [explication.md](explication.md).

## 1. Prérequis à installer hors du projet (dans cet ordre)

| # | Outil | Pourquoi | Comment vérifier |
|---|---|---|---|
| 1 | **Pilote NVIDIA** récent (facultatif mais fortement conseillé) | fine-tuning du reranker sur GPU, CLIP plus rapide | `nvidia-smi` affiche la carte |
| 2 | **Python 3.11** | version imposée par `pyproject.toml` | `py -3.11 --version` |
| 3 | **uv** (gestionnaire d'environnement) | installe les dépendances, dont PyTorch CUDA | `uv --version` |
| 4 | **Ollama** (serveur de modèles locaux) | génération des réponses et analyse des photos | `ollama --version` |
| 5 | Modèles Ollama : `ollama pull qwen2.5:7b` puis `ollama pull qwen2.5vl:7b` | LLM texte (≈4,7 Go) et VLM vision (≈6 Go) | `ollama list` |
| 6 | **Tesseract OCR** (facultatif) avec les langues `fra` et `eng`, variable `TESSDATA_PREFIX` pointant vers `tessdata` | OCR des pages scannées sans couche texte | `tesseract --version` |

Installation sous Windows : uv → `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` ; Ollama → installeur sur ollama.com ;
Tesseract → installeur UB-Mannheim, puis `setx TESSDATA_PREFIX "C:\Program Files\Tesseract-OCR\tessdata"`.
Sans Tesseract, les pages scannées sont simplement signalées et laissées vides.

Ollama doit être **démarré** (icône dans la barre des tâches, ou `ollama serve`) avant d'utiliser l'API.

## 2. Installation du projet

```powershell
cd MAINTiDOC
uv sync --group dev          # crée .venv, installe PyTorch CUDA 12.8, CLIP, ChromaDB, FastAPI, Streamlit, sentence-transformers, pytest
uv run pytest                # 9 tests unitaires, sans modèle lourd ni Ollama
```

Les paramètres (modèles, chemins, taille des fragments) sont dans `src/maintidoc/config.py` et se surchargent
par variable d'environnement ou fichier `.env` (ex. `LLM_TEXT=qwen2.5:3b` pour une machine moins puissante).

## 3. Préparer le corpus et les droits d'accès

1. Copier les PDF autorisés (30 à 50, deux familles d'équipements) dans `data/pdfs/`.
2. Déclarer chaque PDF dans `data/acl.json` : famille, version et rôles autorisés.
   Un PDF absent du fichier reçoit les droits `defaut` (admin seulement) et un avertissement à l'ingestion.

```json
{
  "roles": ["technicien", "superviseur", "admin"],
  "documents": {
    "pompe_P200_manuel_v2.pdf": {"famille": "pompes", "version": "2.0", "roles": ["technicien", "superviseur", "admin"]},
    "cta_securite_interne.pdf": {"famille": "cta",    "version": "1.3", "roles": ["superviseur", "admin"]}
  },
  "defaut": {"famille": "inconnue", "version": "n/a", "roles": ["admin"]}
}
```

## 4. Indexer les documents (L1.2, L1.3, L2.2)

```powershell
uv run maintidoc-ingest              # tous les PDF de data/pdfs (ceux déjà indexés sont ignorés)
uv run maintidoc-ingest --force      # réindexer
uv run maintidoc-ingest data/pdfs/un_seul.pdf
```

Produit `data/extracted/<document_id>/page_NNN.txt` + `images/`, et l'index ChromaDB dans `data/chroma/`.
Un fichier modifié change de `document_id` et remplace l'ancienne version dans l'index.

## 5. Lancer l'assistant (L1.4, L1.5, L2.6)

Dans deux terminaux :

```powershell
uv run maintidoc-api                              # API FastAPI sur http://localhost:8000 (doc : /docs)
uv run streamlit run src/ui/streamlit_app.py      # interface sur http://localhost:8501
```

Dans l'interface : choisir le **rôle** (filtre les documents), poser une **question** ou envoyer une **photographie**.
La réponse affiche les citations [S1]…, le niveau de confiance, les incertitudes, une éventuelle demande de
clarification et les sources (passage, PDF, page, version ; figures affichées).

Appel direct de l'API :

```powershell
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -H "X-Role: technicien" ^
     -d "{\"question\": \"procédure de remplacement du filtre\", \"k\": 6, \"type_resultat\": \"tous\"}"
curl -X POST http://localhost:8000/analyze -H "X-Role: technicien" -F "image=@photo.jpg" -F "question=que signifie ce voyant ?"
```

Premier appel lent (chargement de CLIP, du reranker et du modèle Ollama) ; ensuite la latence dépend surtout du LLM.
Sur un GPU de 4 Go, un modèle 7B déborde sur le CPU : compter 1 à 3 minutes par réponse. Pour accélérer, utiliser
`qwen2.5:3b` / `qwen2.5vl:3b` via `.env`.

## 6. Reranker adapté (L2.3, L2.4, L2.5)

À faire **après** l'indexation du corpus complet, Ollama démarré.

```powershell
uv run maintidoc-dataset --n 300     # 300 questions vérifiées x (1 positif + 3 négatifs) ≈ 1200 paires -> data/reranker/dataset.jsonl
uv run maintidoc-train --epochs 2    # fine-tuning du cross-encoder sur GPU -> data/reranker/modele_adapte/ (+ version.json)
uv run maintidoc-eval                # nDCG@5 / MRR@10 : sans reranker, reranker initial, reranker adapté + tests d'accès -> resultats/rapport_evaluation.md
```

Vérification humaine du jeu de données : ouvrir `dataset.jsonl`, mettre `"verifie": false` sur une ligne douteuse ; elle est
alors ignorée à l'entraînement et à l'évaluation. Dès que `data/reranker/modele_adapte/` existe, l'API l'utilise automatiquement.
Chaque passage tenté coûte deux appels LLM (≈15 à 20 s sur cette machine) et la vérification indépendante n'en retient
qu'une partie (≈20 à 50 % selon le corpus) : compter plusieurs heures pour 300 questions ; lancer la commande et la laisser
tourner. Le nombre de passages disponibles (30 à 50 PDF) doit être nettement supérieur au nombre de questions visé.

## 7. Tests (L1.6)

```powershell
uv run pytest            # découpage, ACL, fusion, métriques, API (rôle inconnu refusé, image obligatoire)
uv run maintidoc-eval    # rapport documentaire et d'accès : aucun résultat hors des PDF autorisés pour chaque rôle
```

## Arborescence

```
data/pdfs/            PDF sources          data/acl.json         droits d'accès
data/extracted/       textes et figures    data/chroma/          index vectoriel
data/reranker/        dataset + modèle     resultats/            rapports d'évaluation
src/maintidoc/        config, schemas, store, ingest, rag, reranker, evaluation, api
src/ui/streamlit_app.py                    tests/
rag_multimodal_agentique.ipynb             notebook de référence du cours (inchangé)
```
