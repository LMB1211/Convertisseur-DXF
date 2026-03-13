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

# --- NOUVELLES OPTIONS ---
st.write("---")
col1, col2 = st.columns(2)

with col1:
    # L'utilisateur peut maintenant choisir le type de trait AutoCAD
    type_trace = st.radio(
        "Type de tracé dans AutoCAD :",
        options=["📐 Lignes brisées (Polyligne - Idéal plans)", "〰️ Vraies courbes (Spline - Idéal dessins)"],
        index=0
    )

with col2:
    # Les réglages précédents, toujours très utiles
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
        # 1. LIRE L'IMAGE (En gardant la transparence s'il y en a)
        bytes_data = np.asarray(bytearray(fichier.read()), dtype=np.uint8)
        image = cv2.imdecode(bytes_data, cv2.IMREAD_UNCHANGED) # UNCHANGED garde le canal Alpha (transparence)
        
        if image is None:
            st.error(f"Oups, impossible de lire l'image {fichier.name}.")
            continue
            
        hauteur, largeur = image.shape[:2]
        
        # 2. GESTION DE LA TRANSPARENCE (PNG)
        # Si l'image a 4 canaux (Rouge, Vert, Bleu, et Alpha/Transparence)
        if len(image.shape) == 3 and image.shape[2] == 4:
            # On crée une feuille blanche de la même taille
            fond_blanc = np.ones_like(image[:, :, :3], dtype=np.uint8) * 255
            alpha = image[:, :, 3].astype(float) / 255.0
            # On "colle" notre image sur la feuille blanche
            for couleur in range(3):
                fond_blanc[:, :, couleur] = image[:, :, couleur] * alpha + fond_blanc[:, :, couleur] * (1 - alpha)
            image = fond_blanc # L'image n'a plus de transparence, elle a un fond blanc

        # S'assurer qu'on a bien une image couleur ou nuance de gris classique
        if len(image.shape) == 3:
            gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gris = image
        
        # 3. PRÉPARATION MATHÉMATIQUE
        # On sépare fermement les traits noirs du fond blanc
        _, binaire = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 4. SQUELETTISATION (Évite les lignes doubles)
        image_bool = binaire > 0
        squelette = skeletonize(image_bool)
        squelette_cv8u = (squelette * 255).astype(np.uint8)
        
        # 5. EXTRACTION DES LIGNES (RETR_LIST prend tout, intérieur comme extérieur)
        contours_maths, _ = cv2.findContours(squelette_cv8u, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 6. CRÉATION DU FICHIER DXF
        doc_dxf = ezdxf.new("R2010")
        espace_modele = doc_dxf.modelspace()
        
        for contour in contours_maths:
            if cv2.contourArea(contour) < taille_min and len(contour) < taille_min:
                continue
                
            # On simplifie un peu le tracé selon le curseur
            epsilon = lissage * cv2.arcLength(contour, True) / 1000.0
            contour_lisse = cv2.approxPolyDP(contour, epsilon, False)

            if len(contour_lisse) >= 2:
                # Formatage des points pour AutoCAD (inversion de la hauteur)
                points = [(float(point[0][0]), float(hauteur - point[0][1])) for point in contour_lisse]
                
                # --- LE CHOIX MAGIQUE : POLYLIGNE OU SPLINE ---
                if "Courbes" in type_trace and len(points) >= 3:
                    # On crée une vraie courbe (Spline) qui passe par ces points
                    espace_modele.add_spline(fit_points=points)
                else:
                    # On crée des segments droits (Polyligne classique)
                    espace_modele.add_lwpolyline(points)
                
        # 7. SAUVEGARDE EN MÉMOIRE
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


