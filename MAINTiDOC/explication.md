# MAINTiDOC — note de cadrage, concepts et implémentation (étapes 1 et 2)

Ce document tient lieu de **note de cadrage et d'architecture (L1.1)**. Il explique les concepts mobilisés par les
deux premières étapes de la feuille de route et la façon dont chacun est implémenté dans ce dépôt, avec le code le
plus court possible (environ 800 lignes de Python, 8 modules et une interface).

## 1. Objectif et périmètre

Un technicien de maintenance pose une question en français (ou envoie une photographie d'un équipement) ; l'assistant
répond **uniquement à partir de la documentation indexée**, cite les documents et les pages, signale ce qu'il ne sait
pas et demande une clarification quand la demande est ambiguë. Il ne filtre que les documents que le **rôle** de
l'utilisateur autorise. Il ne pilote aucun équipement et ne garantit aucun diagnostic.

Périmètre retenu : étapes 1 (socle documentaire, RAG textuel) et 2 (multimodalité, reranker adapté). Les étapes 3
(LangGraph, MCP, tickets) et 4 (bot, déploiement) ne sont pas implémentées.

Choix de pile, décidés avec l'auteur du projet :

| Besoin | Feuille de route | Choix ici | Raison |
|---|---|---|---|
| Extraction PDF | PyMuPDF + OCR local | PyMuPDF + OCR Tesseract si page vide | identique |
| Embeddings texte | BGE-M3 | **CLIP ViT-B/32** (même espace pour texte et images) | pile du notebook de cours, zéro infrastructure, un seul encodeur |
| Base vectorielle | Qdrant | **ChromaDB** embarqué (fichier local) | pas de Docker nécessaire |
| LLM / VLM | Qwen2.5-VL à poids ouverts | **Ollama** : `qwen2.5:7b` (texte) et `qwen2.5vl:7b` (vision) | local, gratuit, conforme |
| Chaîne RAG | LangChain | fonctions Python + `langchain-text-splitters` pour le découpage | moins de code, chaque étape visible |
| Reranker | petit reranker multilingue adapté | `mmarco-mMiniLMv2-L12-H384` fine-tuné avec sentence-transformers | 118 M paramètres, tient sur un GPU de 4 Go |

Conséquence à connaître : CLIP n'accepte que **77 tokens** par texte. Les passages sont donc courts (60 tokens), ce
qui convient à la recherche fine mais oblige à transmettre plusieurs passages au LLM. C'est le reranker qui compense
la faiblesse relative de CLIP sur le texte français long.

## 2. Architecture

```
            PDF (data/pdfs) + acl.json
                      │  ingest.py : PyMuPDF (texte, OCR) + figures → fragments (60 tokens) → CLIP → ChromaDB
                      ▼
   ┌───────────────── data/chroma (collection unique : passages + figures, métadonnées ACL) ─────────────────┐
   │                                                                                                          │
   │   rag.py : question ──CLIP──► passages + figures                                                         │
   │            photo ─────CLIP──► figures similaires + passages (cross-modal)                                │
   │            photo ─────VLM───► description ──CLIP──► passages                                             │
   │                     ╰──► fusion RRF ──► reranker cross-encoder ──► top-k ──► Ollama (JSON Pydantic)      │
   └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                      ▲                                               ▲
          api.py (FastAPI, en-tête X-Role)                 streamlit_app.py (démo : question / photo)

   reranker.py : dataset.jsonl (questions LLM vérifiées + négatifs difficiles) → fine-tuning → modele_adapte/
   evaluation.py : nDCG@5, MRR@10 (3 configurations) + tests d'accès par rôle → resultats/rapport_evaluation.md
```

Modules (`src/maintidoc/`) :

| Module | Rôle | Livrables |
|---|---|---|
| `config.py` | paramètres (chemins, modèles, tailles) surchargeables par `.env` | — |
| `schemas.py` | modèles Pydantic : `Source`, `Reponse` (schéma imposé au LLM), `ReponseRAG` | L2 sorties structurées |
| `store.py` | ACL, encodeur CLIP, collection ChromaDB, recherche filtrée | L1.3 |
| `ingest.py` | pipeline PDF → index | L1.2 |
| `rag.py` | recherche, fusion, reranking, génération, VLM | L1.4, L2.1, L2.2 |
| `reranker.py` | réordonnancement, génération du jeu de données, fine-tuning versionné | L2.3, L2.4 |
| `evaluation.py` | métriques et tests d'accès | L1.6, L2.5 |
| `api.py` | API FastAPI | L1.4 |
| `ui/streamlit_app.py` | interface de démonstration | L1.5, L2.6 |

## 3. Concepts et implémentation

### 3.1 Ingestion documentaire (`ingest.py`)

**Concept.** Un système RAG (*Retrieval-Augmented Generation*) ne fait pas apprendre les documents au modèle ; il les
découpe en passages, les transforme en vecteurs, puis retrouve les passages proches d'une question pour les donner au
LLM comme contexte. La qualité du découpage et des métadonnées conditionne tout le reste.

**Implémentation.**
- `pymupdf.open(pdf)` puis `page.get_text("text", sort=True)` pour chaque page. Si la page n'a pas de couche texte
  (scan), `page.get_textpage_ocr(language="fra+eng")` déclenche Tesseract ; en son absence, la page est signalée et
  laissée vide (`texte_page`).
- Les images intégrées sont extraites avec `page.get_images` / `doc.extract_image`, filtrées sur leur taille
  (icônes et filets ignorés) et sauvegardées dans `data/extracted/<document_id>/images/`.
- Le découpage utilise `RecursiveCharacterTextSplitter` (LangChain) avec une **fonction de longueur en tokens CLIP**
  : fragments de 60 tokens, chevauchement de 10, coupés de préférence aux paragraphes puis aux phrases.
- Chaque élément reçoit des métadonnées : `document_id` (SHA-256 du fichier → change si le fichier change, ce qui
  gère les **versions**), `pdf`, `page`, `fragment`, `type` (texte / image), `famille`, `version`, `ocr`, et un booléen
  `acces_<role>` par rôle. Identifiant stable : `<document_id>:texte:<page>:<fragment>`.
- Réindexer un fichier supprime d'abord ses anciens passages (`collection.delete(where={"pdf": ...})`).

### 3.2 Embeddings et index vectoriel (`store.py`)

**Concept.** Un *embedding* est un vecteur qui représente le sens d'un texte ou d'une image ; deux contenus proches ont
des vecteurs proches (similarité cosinus). CLIP a été entraîné à placer une image et sa légende au même endroit : on
peut donc chercher des figures avec du texte, et des passages avec une photo.

**Implémentation.** `open_clip` (ViT-B/32, poids laion2b) encode textes et images, vecteurs normalisés. ChromaDB
(`PersistentClient`, espace cosinus) stocke tout dans **une seule collection** : les passages et les figures sont
comparables entre eux. `Index.rechercher` prend un vecteur, un rôle et éventuellement un type, et renvoie les k
voisins avec leur score `1 - distance`. L'instance est unique et chargée paresseusement (`lru_cache`).

### 3.3 Droits d'accès (`store.py`, `api.py`)

**Concept.** La feuille de route impose que les réponses ne s'appuient que sur des passages **accessibles** à
l'utilisateur ; le filtrage doit se faire à la recherche, pas après coup, sinon le LLM pourrait lire un document
interdit.

**Implémentation.** `data/acl.json` déclare les rôles et, pour chaque PDF, famille, version et rôles autorisés. Comme
ChromaDB ne filtre pas sur des listes, l'ingestion écrit un booléen par rôle (`acces_technicien: true`…) et la
recherche applique `where={"acces_<role>": True}` (combiné au type par `$and`). Le rôle arrive par l'en-tête HTTP
`X-Role` ; un rôle inconnu est refusé (403). L'endpoint `/image/{id}` revérifie le droit avant de servir une figure.
Un PDF non déclaré est restreint à `admin` par défaut (sécurité par défaut). Le rapport `maintidoc-eval` vérifie
qu'aucun résultat ne provient d'un PDF non autorisé pour chaque rôle.

### 3.4 Chaîne RAG : recherche, fusion, reranking (`rag.py`)

**Concept.** Une seule recherche vectorielle est rarement suffisante. On interroge plusieurs **canaux** (texte →
passages, texte → figures, photo → figures, photo → passages, description de la photo → passages), on **fusionne** les
classements, puis un **reranker** (cross-encoder) relit chaque couple (question, passage) et donne un score beaucoup
plus fiable que la simple proximité des vecteurs, au prix d'un calcul par candidat.

**Implémentation.**
- `rechercher_sources` construit les canaux selon ce qui est fourni (question, photo, ou les deux), chacun avec `3k`
  candidats texte et `k` candidats image.
- La fusion est un **RRF** (*Reciprocal Rank Fusion*) : score = Σ 1/(60 + rang). Simple, sans normalisation des
  scores hétérogènes entre canaux.
- Le reranking (`reranker.reordonner`) utilise le modèle adapté s'il existe (`data/reranker/modele_adapte/`), sinon
  le modèle de base. La requête du reranker est la question concaténée à la description VLM de la photo.
- On garde `k` passages et 3 figures au plus, numérotés S1…Sn (`vers_sources`).

### 3.5 Multimodalité : VLM et photographies (`rag.py`)

**Concept.** Un VLM (*vision-language model*) lit des images. Il sert deux fois : (1) **décrire** la photo pour en faire
une requête textuelle (ce que CLIP seul ne sait pas faire finement : étiquettes, références, anomalies) ; (2)
**répondre** en voyant à la fois la photo, les figures retrouvées et les passages.

**Implémentation.** `decrire_image` appelle `qwen2.5vl:7b` via l'API Ollama avec l'image (redimensionnée à 1024 px,
PNG, transmise en octets). `generer` choisit le modèle vision dès qu'une image est transmise, sinon le modèle texte.
Les images sont numérotées dans le prompt (« image jointe n°2 = figure de X, page 4 ») pour que le modèle relie
chaque figure à sa référence [Sx]. Au plus 3 figures sont transmises pour contenir la taille du contexte.

### 3.6 Génération et sorties structurées (`rag.py`, `schemas.py`)

**Concept.** Pour que l'interface et les tests puissent exploiter la réponse, le LLM doit produire un objet typé et non
un texte libre : réponse, citations, incertitudes, besoin de clarification, niveau de confiance. Les instructions
rappellent que les sources sont des **données non fiables** (défense contre l'injection de prompt contenue dans un
document) et que l'assistant ne pilote rien et ne garantit aucun diagnostic.

**Implémentation.** Ollama accepte un schéma JSON dans `format=` ; on lui passe `Reponse.model_json_schema()` et on
valide la sortie avec `Reponse.model_validate_json`. `temperature=0` pour la reproductibilité. Après génération,
`repondre` fusionne les références écrites dans le texte (`[S1]`) avec le champ `citations`, retire celles qui ne
correspondent à aucune source transmise (et le signale dans `avertissements`), et mesure la latence. Sans aucune
source accessible, la réponse est une demande de clarification explicite, sans appel au LLM.

### 3.7 API et interface (`api.py`, `streamlit_app.py`)

FastAPI expose `/health`, `/documents` (PDF visibles par le rôle), `/ask` (JSON), `/analyze` (multipart : image +
question facultative) et `/image/{id}`. Toutes les réponses sont des `ReponseRAG` validées par Pydantic ; la
documentation interactive est sur `/docs`. Streamlit ne contient aucune logique métier : il appelle l'API avec le
rôle choisi, affiche la réponse, les citations, les incertitudes, la clarification éventuelle et les sources
(passages dépliables, figures servies par l'API).

### 3.8 Jeu de données d'adaptation (`reranker.py`)

**Concept.** Fine-tuner un reranker demande des couples (question, passage) étiquetés pertinents ou non. Sans
annotateurs, on les **synthétise** avec le LLM local à partir des passages réels du corpus, et on les **vérifie** : un
second appel indépendant confirme que le passage répond bien à la question, et un humain peut invalider une ligne
(`"verifie": false`). Les **négatifs difficiles** (passages proches selon CLIP mais d'une autre page) sont ce qui
apprend vraiment au modèle à discriminer.

**Implémentation.** `construire_dataset(n)` : échantillonnage équilibré entre PDF, question en français (70 %) ou en
anglais (30 %) via Ollama avec schéma JSON (`QuestionGeneree`), vérification (`Verification`), 3 négatifs difficiles
issus des 12 voisins CLIP hors même page, partage train/test à 80/20 fixé par graine. Sortie : `data/reranker/dataset.jsonl`,
une ligne par question. 300 questions ≈ 1 200 paires, dans la fourchette 800–1 500 demandée.

### 3.9 Fine-tuning versionné du reranker (`reranker.py`)

**Concept.** Un cross-encoder prend la question et le passage **ensemble** et sort un score de pertinence ; il est lent
mais précis, d'où son usage sur quelques dizaines de candidats seulement. L'adapter au vocabulaire du corpus (termes
de maintenance, références d'équipements) améliore le classement final.

**Implémentation.** `entrainer` construit des `InputExample` (label 1 pour le positif, 0 pour chaque négatif), entraîne
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` avec `CrossEncoder.fit` (AdamW, warmup 10 %, AMP si CUDA), et sauvegarde
dans `data/reranker/modele_adapte/` avec un `version.json` (date, modèle de base, nombre de paires, hyperparamètres).
L'API bascule automatiquement sur ce modèle.

### 3.10 Évaluation (`evaluation.py`)

**Concept.** Deux métriques de classement classiques : **nDCG@5** (gain cumulé actualisé : récompense les passages
pertinents placés haut dans les 5 premiers) et **MRR@10** (inverse du rang du premier passage pertinent dans les 10
premiers). On compare trois configurations sur les mêmes candidats : ordre CLIP brut, reranker initial, reranker adapté.

**Implémentation.** Pour chaque question de test, 20 candidats CLIP ; un candidat est pertinent s'il vient de la
même page que le passage source et d'un fragment voisin (les fragments se chevauchent). Le rapport ajoute les tests
d'accès : pour chaque rôle, 10 questions, 20 résultats chacune, zéro résultat hors PDF autorisés attendu. Résultat dans
`resultats/rapport_evaluation.md` et `.json`, présentés comme résultats observés, jamais comme garanties.

## 4. Contraintes transversales et limites

- **Citations vérifiables** : chaque source porte PDF, page, version et le texte du passage ; les figures sont
  affichées. Les citations non fondées sont retirées.
- **Entrées non fiables** : les documents et la photo sont présentés au LLM comme des données ; les instructions le
  lui rappellent. Aucune exécution d'outil n'est possible à ce stade (pas d'agent).
- **Secrets** : aucun service externe, aucune clé ; `.env` est ignoré par git.
- **Journalisation minimale** : rien n'est enregistré hors la latence dans la réponse.
- **Limites** : CLIP limite les passages à 60 tokens et reste moins bon que BGE-M3 sur le français long ; un modèle
  7B sur un GPU de 4 Go répond en 1 à 3 minutes ; la qualité des questions synthétiques dépend du LLM local, d'où la
  relecture humaine ; l'OCR requiert Tesseract ; les dessins vectoriels des PDF ne sont pas extraits comme figures.
- **Non couvert** : historique via MCP, tickets, approbation humaine, bot, Docker Compose (étapes 3 et 4).
