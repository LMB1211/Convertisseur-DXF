import streamlit as st
import cv2
import numpy as np
import ezdxf
import io
import zipfile

# --- CONFIGURATION DE LA PAGE ---
# On donne un titre à l'onglet du navigateur et on choisit une disposition large
st.set_page_config(page_title="Convertisseur Image vers DXF", layout="centered")

# --- TITRE ET EXPLICATIONS ---
st.title("🪄 Convertisseur Magique : Image ➡️ DXF")
st.write("Bienvenue ! Glissez-déposez vos images (plans, croquis, photos) ci-dessous. Le système va détecter les contours automatiquement et créer un fichier lisible par AutoCAD.")

# --- ZONE DE TÉLÉCHARGEMENT (UPLOAD) ---
# accept_multiple_files=True permet le traitement par lots (batch)
fichiers_uploades = st.file_uploader(
    "Déposez vos images ici (formats acceptés : JPG, PNG)", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True
)

# --- TRAITEMENT DES IMAGES ---
# Si l'utilisateur a déposé des fichiers, on commence le travail
if fichiers_uploades:
    st.write("⏳ *Traitement en cours, veuillez patienter...*")
    
    # Ce dictionnaire va stocker nos fichiers finaux (Nom du fichier -> Contenu du fichier)
    dxf_generes = {}
    
    # Barre de progression visuelle pour que ce soit joli
    barre_progression = st.progress(0)
    total_fichiers = len(fichiers_uploades)

    for index, fichier in enumerate(fichiers_uploades):
        # 1. LIRE L'IMAGE DEPUIS LE NAVIGATEUR
        # On convertit le fichier déposé en un format que l'intelligence de vision (OpenCV) peut comprendre
        bytes_data = np.asarray(bytearray(fichier.read()), dtype=np.uint8)
        image = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error(f"Oups, impossible de lire l'image {fichier.name}.")
            continue
            
        hauteur, largeur = image.shape[:2]
        
        # 2. PRÉPARER L'IMAGE (Magie invisible)
        # On passe en noir et blanc (niveaux de gris)
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # On applique un léger flou pour effacer les petits défauts et le bruit (poussières, pixels)
        flou = cv2.GaussianBlur(gris, (5, 5), 0)
        
        # 3. DÉTECTION DES CONTOURS INTELLIGENTE (Auto-Canny)
        # L'algorithme regarde la luminosité moyenne de l'image pour régler ses paramètres tout seul
        mediane = np.median(flou)
        seuil_bas = int(max(0, (1.0 - 0.33) * mediane))
        seuil_haut = int(min(255, (1.0 + 0.33) * mediane))
        # Canny est l'outil qui trace les lignes autour des formes
        contours_image = cv2.Canny(flou, seuil_bas, seuil_haut)
        
        # On extrait mathématiquement ces lignes dessinées
        contours_maths, _ = cv2.findContours(contours_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4. CRÉATION DU FICHIER DXF
        # On crée un nouveau document AutoCAD (format R2010, très standard et compatible)
        doc_dxf = ezdxf.new("R2010")
        espace_modele = doc_dxf.modelspace()
        
        # On dessine chaque ligne trouvée dans le nouveau fichier DXF
        for contour in contours_maths:
            # Un contour doit avoir au moins 2 points pour faire une ligne
            if len(contour) >= 2:
                # Petite astuce mathématique : OpenCV compte de haut en bas, AutoCAD de bas en haut.
                # On inverse la hauteur (hauteur - y) pour que le dessin ne soit pas à l'envers dans AutoCAD !
                points = [(point[0][0], hauteur - point[0][1]) for point in contour]
                espace_modele.add_lwpolyline(points)
                
        # 5. SAUVEGARDE EN MÉMOIRE
        # Au lieu de sauvegarder sur un disque dur physique, on sauvegarde dans la mémoire du site web
        flux_texte = io.StringIO()
        doc_dxf.write(flux_texte)
        
        # On prépare le nom du nouveau fichier (ex: "mon_plan.jpg" devient "mon_plan.dxf")
        nom_base = fichier.name.rsplit('.', 1)[0]
        nom_dxf = f"{nom_base}.dxf"
        
        # On stocke le résultat
        dxf_generes[nom_dxf] = flux_texte.getvalue()
        
        # Mise à jour de la barre de progression
        barre_progression.progress((index + 1) / total_fichiers)

    st.success("✅ Traitement terminé avec succès !")

    # --- TÉLÉCHARGEMENT ---
    st.write("### 📥 Récupérez vos fichiers")
    
    # Cas 1 : Une seule image a été traitée
    if len(dxf_generes) == 1:
        nom_fichier, contenu = list(dxf_generes.items())[0]
        st.download_button(
            label=f"Télécharger {nom_fichier}",
            data=contenu,
            file_name=nom_fichier,
            mime="application/dxf"
        )
        
    # Cas 2 : Plusieurs images ont été traitées (Batch) -> On crée un dossier compressé (ZIP)
    elif len(dxf_generes) > 1:
        # On prépare un fichier ZIP en mémoire
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

