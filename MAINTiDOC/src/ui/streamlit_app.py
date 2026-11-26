"""Interface de démonstration Streamlit : question texte ou photographie, réponse sourcée, images retrouvées."""
import httpx
import streamlit as st

from maintidoc.config import settings

API = settings.api_url
st.set_page_config(page_title="MAINTiDOC", layout="wide")
st.title("MAINTiDOC – assistant documentaire de maintenance")
st.caption("Réponses fondées sur la documentation indexée. Aucun pilotage d'équipement, aucun diagnostic garanti.")

with st.sidebar:
    try:
        sante = httpx.get(f"{API}/health", timeout=10).json()
    except Exception as e:
        st.error(f"API injoignable ({API}) : {e}")
        st.stop()
    role = st.selectbox("Rôle", sante["roles"])
    k = st.slider("Passages (k)", 1, 12, 6)
    type_resultat = st.radio("Type de sources", ["tous", "texte", "image"], horizontal=True)
    entetes = {"X-Role": role}
    st.markdown(f"**Index :** {sante['elements_indexes']} éléments")
    st.markdown("**Documents accessibles :**")
    for d in httpx.get(f"{API}/documents", headers=entetes, timeout=30).json():
        st.markdown(f"- {d['pdf']} ({d['famille']}, v{d['version']})")


def afficher(rep: dict) -> None:
    r = rep["reponse"]
    if r["besoin_clarification"]:
        st.warning(f"Clarification demandée : {r['question_clarification']}")
    st.markdown(r["reponse"])
    st.markdown(f"**Confiance :** {r['niveau_confiance']} · **Citations :** {', '.join(r['citations']) or 'aucune'} · {rep['latence_ms']} ms")
    if r["incertitudes"]:
        st.markdown("**Incertitudes :** " + " ; ".join(r["incertitudes"]))
    for a in rep["avertissements"]:
        st.info(a)
    st.subheader("Sources")
    for s in rep["sources"]:
        with st.expander(f"[{s['reference']}] {s['type']} · {s['pdf']} · page {s['page']} · v{s['version']} · score {s['score']}"):
            if s["type"] == "texte":
                st.write(s["extrait"])
            else:
                st.image(httpx.get(f"{API}/image/{s['id']}", headers=entetes, timeout=30).content, width=500)


tab_q, tab_photo = st.tabs(["Question", "Photographie"])

with tab_q:
    question = st.text_area("Question technique", placeholder="Ex. : quelle est la procédure de remplacement du filtre ?")
    if st.button("Interroger") and question.strip():
        with st.spinner("Recherche et génération en cours..."):
            r = httpx.post(f"{API}/ask", json={"question": question, "k": k, "type_resultat": type_resultat}, headers=entetes, timeout=600)
        afficher(r.json()) if r.status_code == 200 else st.error(r.text)

with tab_photo:
    photo = st.file_uploader("Photographie de l'équipement", type=["png", "jpg", "jpeg", "webp"])
    question_photo = st.text_input("Question (facultatif)", placeholder="Ex. : que signifie ce voyant ?")
    if photo:
        st.image(photo, width=400)
    if st.button("Analyser") and photo:
        with st.spinner("Analyse de la photo, recherche et génération..."):
            r = httpx.post(f"{API}/analyze", files={"image": (photo.name, photo.getvalue(), photo.type)},
                           data={"question": question_photo, "k": k, "type_resultat": type_resultat}, headers=entetes, timeout=900)
        afficher(r.json()) if r.status_code == 200 else st.error(r.text)
