import streamlit as st
import cv2
import numpy as np
import ezdxf
import io
import zipfile
from skimage.morphology import skeletonize

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Convertisseur Image vers DXF", layout="centered")

st.title("🪄 Convertisseur Magique : Image ➡️ DXF")
st.write("Glissez-déposez vos images (plans, croquis, PNG transparents). Le système détectera tous les détails intérieurs et extérieurs !")

# --- OPTIONS ---
st.write("---")
col1, col2 = st.columns(2)

with col1:
    type_trace = st.radio(
        "Type de tracé dans AutoCAD :",
        options=["📐 Lignes brisées (Polyligne - Idéal plans)", "〰️ Vraies courbes (Spline - Idéal dessins)"],
        index=0
    )

with col2:
    lissage = st.slider("Simplification des traits", min_value=1.0, max_value=10.0, value=2.0, step=0.5, help="Réduit le nombre de points d'ancrage.")
    taille_min = st.slider("Ignorer la poussière", min_value=0, max_value=500, value=50, step=10, help="Efface les micro-détails indésirables.")
st.write("---")

# --- ZONE DE TÉLÉCHARGEMENT ---
fichiers_uploades = st.file_uploader(
    "Déposez vos images ici (JPG, PNG même transparents)", 
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
        image = cv2.imdecode(bytes_data, cv2.IMREAD_UNCHANGED) 
        
        if image is None:
            continue
            
        hauteur, largeur = image.shape[:2]
        
        # 2. GESTION DE LA TRANSPARENCE (PNG)
        if len(image.shape) == 3 and image.shape[2] == 4:
            fond_blanc = np.ones_like(image[:, :, :3], dtype=np.uint8) * 255
            alpha = image[:, :, 3].astype(float) / 255.0
            for couleur in range(3):
                fond_blanc[:, :, couleur] = image[:, :, couleur] * alpha + fond_blanc[:, :, couleur] * (1 - alpha)
            image = fond_blanc 

        if len(image.shape) == 3:
            gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gris = image
        
        # 3. PRÉPARATION MATHÉMATIQUE
        _, binaire = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 4. SQUELETTISATION 
        image_bool = binaire > 0
        squelette = skeletonize(image_bool)
        squelette_cv8u = (squelette * 255).astype(np.uint8)
        
        # ON GARDE TOUS LES POINTS (CHAIN_APPROX_NONE) POUR FAIRE DE BELLES COURBES
        contours_maths, _ = cv2.findContours(squelette_cv8u, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # 5. CRÉATION DU FICHIER DXF
        doc_dxf = ezdxf.new("R2010")
        espace_modele = doc_dxf.modelspace()
        
        est_mode_courbe = "Courbes" in type_trace
        
        for contour in contours_maths:
            if cv2.contourArea(contour) < taille_min and len(contour) < taille_min:
                continue
                
            # L'ASTUCE EST ICI : 
            if est_mode_courbe:
                # Si on veut des courbes, on garde beaucoup de points (on divise l'epsilon par 5)
                epsilon = (lissage / 5.0) * cv2.arcLength(contour, True) / 1000.0
            else:
                # Si on veut des traits droits, on simplifie normalement
                epsilon = lissage * cv2.arcLength(contour, True) / 1000.0
                
            contour_lisse = cv2.approxPolyDP(contour, epsilon, False)

            # Extraction des coordonnées pour AutoCAD
            points = [(float(point[0][0]), float(hauteur - point[0][1])) for point in contour_lisse]
            
            # --- CRÉATION DE LA LIGNE OU DE LA COURBE ---
            if len(points) >= 2:
                # Une vraie Spline a besoin d'au moins 4 points pour exister
                if est_mode_courbe and len(points) >= 4:
                    try:
                        espace_modele.add_spline(fit_points=points)
                    except Exception:
                        # Sécurité si la forme est trop bizarre mathématiquement
                        espace_modele.add_lwpolyline(points)
                else:
                    # Pour les tout petits traits (2 ou 3 points), on fait une polyligne
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
    if len(dxf_generes) == 1:
        nom_fichier, contenu = list(dxf_generes.items())[0]
        st.download_button(
            label=f"📥 Télécharger {nom_fichier}",
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
            label="📥 Télécharger tous les DXF (Dossier ZIP)",
            data=zip_buffer.getvalue(),
            file_name="fichiers_convertis_dxf.zip",
            mime="application/zip"
        )


