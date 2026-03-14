import streamlit as st
import cv2
import numpy as np
import ezdxf
import io
import zipfile
from skimage.morphology import skeletonize

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Convertisseur Image vers DXF", layout="wide")

st.title("🪄 Convertisseur Magique : Image ➡️ DXF (Pro)")
st.write("Le moteur d'ancrage virtuel garantit des lignes parfaitement droites et des virages ultra-fluides, sans aucune vaguelette.")

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
            "🎯 Trait central unique (Force une ligne simple sans doublon)",
            "⭕ Contours exacts (Garde l'épaisseur de votre coup de crayon)"
        ],
        index=0
    )

with col2:
    lissage = st.slider(
        "Tolérance de lissage", 
        min_value=0.5, max_value=15.0, value=2.0, step=0.5, 
        help="Efface le bruit et les escaliers (pixels). Gardez autour de 2.0 ou 3.0."
    )
    
    # NOUVEAU CURSEUR : Le secret pour les Splines
    tension = st.slider(
        "Tension des lignes droites (Ancrage Spline)", 
        min_value=5.0, max_value=50.0, value=10.0, step=1.0, 
        help="Plus la valeur est basse, plus on injecte de points alignés pour forcer la ligne à rester droite avant un virage. (Moyenne recommandée : 10)"
    )
    
    taille_min = st.slider("Ignorer la poussière", min_value=0, max_value=500, value=50, step=10)
st.write("---")

# --- ZONE DE TÉLÉCHARGEMENT ---
fichiers_uploades = st.file_uploader(
    "Déposez vos images ici (JPG, PNG même transparents)", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True
)

# --- TRAITEMENT DES IMAGES ---
if fichiers_uploades:
    st.write("⏳ *Moteur d'ancrage mathématique en cours de calcul...*")
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

        if len(image.shape) == 3:
            gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gris = image
            
        gris_adouci = cv2.GaussianBlur(gris, (3, 3), 0)
        
        # 2. SEPARATION FOND / TRAIT
        _, binaire = cv2.threshold(gris_adouci, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. ANALYSE SELON LE CHOIX
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
                
            # ÉTAPE A : Le lissage absolu (enlève l'effet escalier)
            contour_simplifie = cv2.approxPolyDP(contour_propre, float(lissage), contour_ferme)

            # ÉTAPE B : L'ANCRAGE MATHÉMATIQUE (La solution à ton problème)
            # Si on veut une Spline, on injecte des points pour la tendre
            if "Splines" in type_trace:
                points_denses = []
                nb_points = len(contour_simplifie)
                
                for i in range(nb_points):
                    p1 = contour_simplifie[i][0]
                    
                    if i == nb_points - 1:
                        if not contour_ferme:
                            points_denses.append([[p1[0], p1[1]]])
                            break
                        else:
                            p2 = contour_simplifie[0][0] # On boucle
                    else:
                        p2 = contour_simplifie[i+1][0]
                        
                    points_denses.append([[p1[0], p1[1]]])
                    
                    # On calcule la distance de la ligne droite
                    dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    
                    # Si la ligne droite est plus grande que la tension demandée...
                    if dist > tension:
                        # ... on injecte des points parfaits au milieu !
                        nb_injections = int(dist / tension)
                        for j in range(1, nb_injections):
                            fraction = j / float(nb_injections)
                            nx = p1[0] + (p2[0] - p1[0]) * fraction
                            ny = p1[1] + (p2[1] - p1[1]) * fraction
                            points_denses.append([[nx, ny]])
                            
                contour_final = np.array(points_denses, dtype=np.float32)
            else:
                contour_final = contour_simplifie # Pour les Polylignes, pas besoin d'ancrage

            # Préparation des coordonnées pour AutoCAD (Y inversé)
            points = [(float(p[0][0]), float(hauteur - p[0][1])) for p in contour_final]
            
            # --- DESSIN FINAL ---
            if len(points) >= 2:
                if "Splines" in type_trace and len(points) >= 4:
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

    st.success("✅ Fichiers ancrés et générés avec succès !")

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



