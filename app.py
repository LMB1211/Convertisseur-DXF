import streamlit as st
import cv2
import numpy as np
import ezdxf
import io
import zipfile
from skimage.morphology import skeletonize

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Convertisseur Image vers DXF", layout="centered")

# --- INTERFACE ULTRA-MINIMALISTE ---
st.title("🪄 Image ➡️ DXF")
st.write("Glissez-déposez vos dessins filaires, personnages ou meubles. Le moteur intelligent s'occupe de tout : les lignes droites restent droites, les courbes restent fluides.")
st.write("---")

# --- ZONE DE TÉLÉCHARGEMENT ---
fichiers_uploades = st.file_uploader(
    "Importez vos images (JPG, PNG transparents acceptés)", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# --- MOTEUR DE TRAITEMENT INTELLIGENT ---
if fichiers_uploades:
    st.write("⏳ *Analyse géométrique en cours...*")
    dxf_generes = {}
    barre_progression = st.progress(0)
    total_fichiers = len(fichiers_uploades)

    for index, fichier in enumerate(fichiers_uploades):
        # 1. LECTURE ET GESTION DE LA TRANSPARENCE
        bytes_data = np.asarray(bytearray(fichier.read()), dtype=np.uint8)
        image = cv2.imdecode(bytes_data, cv2.IMREAD_UNCHANGED) 
        
        if image is None:
            continue
            
        hauteur, largeur = image.shape[:2]
        
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
            
        # 2. EXTRACTION DU SQUELETTE (Le trait central parfait)
        gris_adouci = cv2.GaussianBlur(gris, (3, 3), 0)
        _, binaire = cv2.threshold(gris_adouci, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        image_bool = binaire > 0
        squelette = skeletonize(image_bool)
        img_a_traiter = (squelette * 255).astype(np.uint8)
        
        # 3. EXTRACTION DES CONTOURS
        contours_maths, _ = cv2.findContours(img_a_traiter, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        # 4. CRÉATION DU DXF
        doc_dxf = ezdxf.new("R2010")
        espace_modele = doc_dxf.modelspace()
        
        for contour in contours_maths:
            # Filtrage de la poussière
            if len(contour) < 10:
                continue
            
            # Suppression des lignes doubles (aller-retour)
            points_uniques = []
            pixels_vus = set()
            for pt in contour:
                coord = (int(pt[0][0]), int(pt[0][1]))
                if coord not in pixels_vus:
                    pixels_vus.add(coord)
                    points_uniques.append([coord])
            contour_propre = np.array(points_uniques, dtype=np.int32)
            
            if len(contour_propre) < 2:
                continue
                
            # Lissage de base pour retirer l'effet "pixel en escalier" (epsilon = 1.5)
            contour_simplifie = cv2.approxPolyDP(contour_propre, 1.5, False)
            
            # Conversion des coordonnées pour AutoCAD (Y inversé)
            pts = [(float(p[0][0]), float(hauteur - p[0][1])) for p in contour_simplifie]
            
            # --- 🧠 L'INTELLIGENCE MATHÉMATIQUE (Découpage Lignes / Courbes) ---
            chunks = []
            current_chunk = [pts[0]]
            
            # Détection des angles cassés (coins)
            for i in range(1, len(pts) - 1):
                A = np.array(pts[i-1])
                B = np.array(pts[i])
                C = np.array(pts[i+1])
                
                V1 = B - A
                V2 = C - B
                n1 = np.linalg.norm(V1)
                n2 = np.linalg.norm(V2)
                
                is_corner = False
                if n1 > 0 and n2 > 0:
                    cos_theta = np.dot(V1, V2) / (n1 * n2)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_theta))
                    # Si le trait tourne de plus de 25 degrés, c'est un coin net !
                    if angle > 25.0:  
                        is_corner = True
                        
                current_chunk.append(pts[i])
                if is_corner:
                    chunks.append(current_chunk)
                    current_chunk = [pts[i]] # On redémarre un nouveau morceau après le coin
                    
            current_chunk.append(pts[-1])
            chunks.append(current_chunk)
            
            # --- DESSIN INTELLIGENT DANS AUTOCAD ---
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                
                if len(chunk) == 2:
                    # 2 points = Une ligne parfaite
                    espace_modele.add_line(chunk[0], chunk[1])
                else:
                    # Analyse du comportement du trait
                    p_start = np.array(chunk[0])
                    p_end = np.array(chunk[-1])
                    dist_straight = np.linalg.norm(p_end - p_start)
                    
                    dist_path = 0.0
                    for i in range(1, len(chunk)):
                        dist_path += np.linalg.norm(np.array(chunk[i]) - np.array(chunk[i-1]))
                        
                    # Si le chemin parcouru est presque égal à la ligne droite, c'est une ligne droite !
                    if dist_straight > 0 and (dist_path / dist_straight) < 1.02:
                        espace_modele.add_line(chunk[0], chunk[-1])
                    else:
                        # Sinon, c'est une vraie courbe, on lance la Spline !
                        try:
                            espace_modele.add_spline(fit_points=chunk)
                        except Exception:
                            espace_modele.add_lwpolyline(chunk)
                
        # 5. FINALISATION
        flux_texte = io.StringIO()
        doc_dxf.write(flux_texte)
        nom_base = fichier.name.rsplit('.', 1)[0]
        nom_dxf = f"{nom_base}.dxf"
        dxf_generes[nom_dxf] = flux_texte.getvalue()
        barre_progression.progress((index + 1) / total_fichiers)

    st.success("✅ Fichiers générés avec succès !")

    # --- TÉLÉCHARGEMENT ---
    st.write("---")
    if len(dxf_generes) == 1:
        nom_fichier, contenu = list(dxf_generes.items())[0]
        st.download_button(
            label=f"📥 Télécharger {nom_fichier}",
            data=contenu,
            file_name=nom_fichier,
            mime="application/dxf",
            use_container_width=True
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
            mime="application/zip",
            use_container_width=True
        )


