import streamlit as st
import cv2
import numpy as np
import ezdxf
import io
import zipfile
from skimage.morphology import skeletonize

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Convertisseur Image vers DXF", layout="centered")

# --- TITRE ET EXPLICATIONS ---
st.title("🪄 Convertisseur Magique : Image ➡️ DXF")
st.write("Bienvenue ! Glissez-déposez vos images (plans, croquis, photos) ci-dessous. Le système va détecter les lignes centrales et créer un fichier lisible par AutoCAD.")

# --- OPTIONS SIMPLIFIÉES (Optionnelles mais utiles) ---
st.write("---")
col1, col2 = st.columns(2)
with col1:
    # Pour lisser les courbes "en escalier"
    lissage = st.slider("Niveau de lissage des courbes", min_value=1.0, max_value=10.0, value=3.0, step=0.5, help="Plus la valeur est grande, plus les lignes droites et les courbes seront douces, mais vous perdrez un peu de détails fins.")
with col2:
    # Pour ignorer les petites poussières
    taille_min = st.slider("Ignorer les petits points (pixels)", min_value=0, max_value=500, value=50, step=10, help="Supprime les petits artefacts ou la poussière sur l'image.")
st.write("---")


# --- ZONE DE TÉLÉCHARGEMENT ---
fichiers_uploades = st.file_uploader(
    "Déposez vos images ici (formats acceptés : JPG, PNG)", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True
)

# --- TRAITEMENT DES IMAGES ---
if fichiers_uploades:
    st.write("⏳ *Traitement en cours, veuillez patienter...*")
    
    dxf_generes = {}
    barre_progression = st.progress(0)
    total_fichiers = len(fichiers_uploades)

    for index, fichier in enumerate(fichiers_uploades):
        # 1. LIRE L'IMAGE
        bytes_data = np.asarray(bytearray(fichier.read()), dtype=np.uint8)
        image = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error(f"Oups, impossible de lire l'image {fichier.name}.")
            continue
            
        hauteur, largeur = image.shape[:2]
        
        # 2. PRÉPARATION (Noir et Blanc strict)
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # On inverse souvent les couleurs car la squelettisation travaille sur le blanc (les traits à garder) 
        # sur fond noir. On utilise un seuil automatique (Otsu) pour séparer parfaitement le trait du fond.
        _, binaire = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. SQUELETTISATION (Le secret pour éviter les lignes doubles)
        # On transforme l'image binaire (0 ou 255) en valeurs True/False (0 ou 1) pour l'algorithme
        image_bool = binaire > 0
        # On ronge les traits jusqu'à 1 pixel d'épaisseur
        squelette = skeletonize(image_bool)
        
        # On repasse en format image (0 ou 255) pour OpenCV
        squelette_cv8u = (squelette * 255).astype(np.uint8)
        
        # 4. EXTRACTION ET LISSAGE DES LIGNES
        contours_maths, _ = cv2.findContours(squelette_cv8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. CRÉATION DU FICHIER DXF
        doc_dxf = ezdxf.new("R2010")
        espace_modele = doc_dxf.modelspace()
        
        for contour in contours_maths:
            # On ignore les contours trop petits (la "poussière")
            if cv2.contourArea(contour) < taille_min and len(contour) < taille_min:
                continue
                
            # LISSAGE : On simplifie le contour pour enlever l'effet "escalier" (Aliasing)
            # L'algorithme de Douglas-Peucker (approxPolyDP) réduit le nombre de points d'une courbe
            epsilon = lissage * cv2.arcLength(contour, True) / 1000.0 # Plus lissage est grand, plus on lisse
            contour_lisse = cv2.approxPolyDP(contour, epsilon, False)

            if len(contour_lisse) >= 2:
                # On inverse la hauteur pour AutoCAD
                points = [(point[0][0], hauteur - point[0][1]) for point in contour_lisse]
                espace_modele.add_lwpolyline(points)
                
        # 6. SAUVEGARDE EN MÉMOIRE
        flux_texte = io.StringIO()
        doc_dxf.write(flux_texte)
        
        nom_base = fichier.name.rsplit('.', 1)[0]
        nom_dxf = f"{nom_base}.dxf"
        dxf_generes[nom_dxf] = flux_texte.getvalue()
        
        barre_progression.progress((index + 1) / total_fichiers)

    st.success("✅ Traitement terminé avec succès !")

    # --- TÉLÉCHARGEMENT ---
    st.write("### 📥 Récupérez vos fichiers")
    
    if len(dxf_generes) == 1:
        nom_fichier, contenu = list(dxf_generes.items())[0]
        st.download_button(
            label=f"Télécharger {nom_fichier}",
            data=contenu,
            file_name=nom_fichier,
            mime="application/dxf"
        )
        
    elif len(dxf_generes) > 1:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as archive_zip:
            for nom_fichier, contenu in dxf_generes.items():
                archive_zip.writestr(nom_fichier, contenu)
                
        st.download_button(
            label="Télécharger tous les DXF (Dossier ZIP)",
            data=zip_buffer.getvalue(),
            file_name="fichiers_convertis_dxf.zip",
            mime="application/zip"
        )


