import streamlit as st
import cv2
import numpy as np
import ezdxf
import io
import zipfile
from skimage.morphology import skeletonize

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Convertisseur Image vers DXF", layout="wide")

st.title("🪄 Convertisseur Magique : Image ➡️ DXF")
st.write("Glissez-déposez vos images. Les lignes droites restent droites, les courbes restent fluides !")

# --- OPTIONS DU PANNEAU DE CONTRÔLE ---
st.write("---")
col1, col2 = st.columns(2)

with col1:
    type_trace = st.radio(
        "1. Type de tracé dans AutoCAD :",
        options=["〰️ Vraies courbes (Splines - Parfait pour les dessins)", "📐 Lignes brisées (Polylignes - Parfait pour les plans)"],
        index=0
    )
    
    mode_analyse = st.radio(
        "2. Méthode d'analyse de l'image :",
        options=[
            "🎯 Trait central unique (Idéal Plans : force une ligne simple sans doublon)",
            "⭕ Contours exacts (Idéal Dessins/Logos : garde l'épaisseur de votre coup de crayon)"
        ],
        index=0
    )

with col2:
    lissage = st.slider("Simplification des traits", min_value=1.0, max_value=10.0, value=2.0, step=0.5, help="Réduit le nombre de points. Gardez bas pour plus de détails.")
    taille_min = st.slider("Ignorer la poussière", min_value=0, max_value=500, value=50, step=10, help="Efface les petits points indésirables.")
st.write("---")

# --- ZONE DE TÉLÉCHARGEMENT ---
fichiers_uploades = st.file_uploader(
    "Déposez vos images ici (JPG, PNG même transparents)", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True
)

# --- TRAITEMENT DES IMAGES ---
if fichiers_uploades:
    st.write("⏳ *Traitement en cours, un fer à repasser mathématique lisse vos traits...*")
    dxf_generes = {}
    barre_progression = st.progress(0)
    total_fichiers = len(fichiers_uploades)

    for index, fichier in enumerate(fichiers_uploades):
        # 1. LIRE L'IMAGE (Transparence incluse)
        bytes_data = np.asarray(bytearray(fichier.read()), dtype=np.uint8)
        image = cv2.imdecode(bytes_data, cv2.IMREAD_UNCHANGED) 
        
        if image is None:
            continue
            
        hauteur, largeur = image.shape[:2]
        
        # Gestion du PNG transparent
        if len(image.shape) == 3 and image.shape[2] == 4:
            fond_blanc = np.ones_like(image[:, :, :3], dtype=np.uint8) * 255
            alpha = image[:, :, 3].astype(float) / 255.0
            for couleur in range(3):
                fond_blanc[:, :, couleur] = image[:, :, couleur] * alpha + fond_blanc[:, :, couleur] * (1 - alpha)
            image = fond_blanc 

        # Conversion en noir et blanc
        if len(image.shape) == 3:
            gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gris = image
        
        # 2. SÉPARATION FOND / TRAIT
        _, binaire = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. ANALYSE SELON LE CHOIX DE L'UTILISATEUR
        if "Trait central" in mode_analyse:
            image_bool = binaire > 0
            squelette = skeletonize(image_bool)
            img_a_traiter = (squelette * 255).astype(np.uint8)
            est_mode_squelette = True
        else:
            img_a_traiter = binaire
            est_mode_squelette = False
        
        contours_maths, _ = cv2.findContours(img_a_traiter, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # 4. CRÉATION DU DXF
        doc_dxf = ezdxf.new("R2010")
        espace_modele = doc_dxf.modelspace()
        
        for contour in contours_maths:
            if cv2.contourArea(contour) < taille_min and len(contour) < taille_min:
                continue
            
            # Anti-lignes doubles
            if est_mode_squelette:
                points_uniques = []
                pixels_vus = set()
                for pt in contour:
                    coord = (int(pt[0][0]), int(pt[0][1]))
                    if coord not in pixels_vus:
                        pixels_vus.add(coord)
                        points_uniques.append([coord])
                contour_propre = np.array(points_uniques, dtype=np.int32)
                contour_ferme = False 
            else:
                contour_propre = contour
                contour_ferme = True

            if len(contour_propre) < 2:
                continue
                
            # --- LE FER À REPASSER MATHÉMATIQUE (Lissage anti-vaguelettes) ---
            # Transforme les escaliers de pixels en lignes droites parfaites
            if len(contour_propre) > 4:
                pts_flottants = contour_propre.astype(np.float32)
                # On applique 4 passages de lissage pour un rendu parfait
                for _ in range(4):
                    temp = np.copy(pts_flottants)
                    # Aligne chaque point entre son voisin de gauche et de droite
                    temp[1:-1] = (pts_flottants[:-2] + pts_flottants[1:-1] + pts_flottants[2:]) / 3.0
                    
                    if contour_ferme: # Raccorde proprement le début et la fin
                        temp[0] = (pts_flottants[-1] + pts_flottants[0] + pts_flottants[1]) / 3.0
                        temp[-1] = (pts_flottants[-2] + pts_flottants[-1] + pts_flottants[0]) / 3.0
                        
                contour_propre = temp # On remplace l'escalier par la courbe lissée

            # Extraction des points de repère importants
            if "Splines" in type_trace:
                # La division par 10 garde beaucoup de points d'ancrage pour forcer la Spline à rester tendue et droite
                epsilon = (lissage / 10.0) * cv2.arcLength(contour_propre, contour_ferme) / 1000.0
            else:
                epsilon = lissage * cv2.arcLength(contour_propre, contour_ferme) / 1000.0
                
            contour_final = cv2.approxPolyDP(contour_propre, epsilon, contour_ferme)

            # Préparation des coordonnées pour AutoCAD (Y inversé)
            points = [(float(p[0][0]), float(hauteur - p[0][1])) for p in contour_final]
            
            # --- DESSIN FINAL ---
            if len(points) >= 2:
                if "Splines" in type_trace and len(points) >= 3:
                    try:
                        espace_modele.add_spline(fit_points=points)
                    except Exception:
                        espace_modele.add_lwpolyline(points)
                else:
                    espace_modele.add_lwpolyline(points)
                
        # 5. FINALISATION
        flux_texte = io.StringIO()
        doc_dxf.write(flux_texte)
        nom_base = fichier.name.rsplit('.', 1)[0]
        nom_dxf = f"{nom_base}.dxf"
        dxf_generes[nom_dxf] = flux_texte.getvalue()
        barre_progression.progress((index + 1) / total_fichiers)

    st.success("✅ Fichiers lissés et générés avec succès !")

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


