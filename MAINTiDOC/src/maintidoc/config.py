"""Paramètres globaux (surchargeables via .env ou variables d'environnement)."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Chemins
    pdfs_dir: Path = Path("data/pdfs")
    extracted_dir: Path = Path("data/extracted")
    acl_file: Path = Path("data/acl.json")
    chroma_dir: Path = Path("data/chroma")
    reranker_dir: Path = Path("data/reranker")
    resultats_dir: Path = Path("resultats")
    collection: str = "maintidoc_clip_vit_b32"

    # Encodeur multimodal CLIP (texte et images dans le même espace)
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    chunk_tokens: int = 60      # CLIP accepte 77 tokens max
    chunk_overlap: int = 10
    min_image_px: int = 80      # ignorer les icônes et filets

    # Modèles génératifs locaux (Ollama)
    ollama_url: str = "http://localhost:11434"
    llm_text: str = "qwen2.5:7b"
    llm_vision: str = "qwen2.5vl:7b"
    num_ctx: int = 8192

    # Reranker (cross-encoder multilingue, petit)
    reranker_base: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_adapted: Path = Path("data/reranker/modele_adapte")

    # Interface
    api_url: str = "http://localhost:8000"


settings = Settings()
