from maintidoc.store import filtre_where, metadonnees_acces

ACL = {"roles": ["technicien", "admin"], "documents": {"a.pdf": {"famille": "pompes", "version": "2", "roles": ["technicien", "admin"]}},
       "defaut": {"famille": "inconnue", "version": "n/a", "roles": ["admin"]}}


def test_metadonnees_acces_document_connu():
    m = metadonnees_acces("a.pdf", ACL)
    assert m == {"famille": "pompes", "version": "2", "acces_technicien": True, "acces_admin": True}


def test_metadonnees_acces_document_inconnu_restreint_par_defaut():
    m = metadonnees_acces("x.pdf", ACL)
    assert m["acces_technicien"] is False and m["acces_admin"] is True


def test_filtre_where():
    assert filtre_where("technicien") == {"acces_technicien": True}
    assert filtre_where("admin", "image") == {"$and": [{"acces_admin": True}, {"type": "image"}]}
