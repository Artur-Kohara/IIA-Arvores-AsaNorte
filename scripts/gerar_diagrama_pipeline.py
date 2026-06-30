import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_pipeline_diagram():
    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    colors = {
        'phase1_bg': '#e1f5fe', 'phase1_border': '#0288d1',
        'phase2_bg': '#fff8e1', 'phase2_border': '#f57c00',
        'phase3_bg': '#e8f5e9', 'phase3_border': '#388e3c',
        'phase4_bg': '#f3e5f5', 'phase4_border': '#7b1fa2',
        'text_dark': '#212121',
        'arrow': '#555555'
    }
    
    # Draw background group boxes (Swimlanes / Phases)
    # Phase 1: Pré-processamento
    p1_rect = patches.FancyBboxPatch(
        (0.2, 5.2), 5.5, 4.4, boxstyle="round,pad=0.2",
        facecolor='#f7fbfd', edgecolor='#b3e5fc', linestyle='--', linewidth=1.5
    )
    ax.add_patch(p1_rect)
    ax.text(0.5, 9.4, "Fase 1: Pré-processamento", fontsize=12, fontweight='bold', color=colors['phase1_border'])
    
    # Phase 2: Curadoria e QA
    p2_rect = patches.FancyBboxPatch(
        (6.1, 5.2), 5.5, 4.4, boxstyle="round,pad=0.2",
        facecolor='#fffdf7', edgecolor='#ffe0b2', linestyle='--', linewidth=1.5
    )
    ax.add_patch(p2_rect)
    ax.text(6.4, 9.4, "Fase 2: Curadoria & Controle de Qualidade", fontsize=12, fontweight='bold', color=colors['phase2_border'])
    
    # Phase 3: Treinamento
    p3_rect = patches.FancyBboxPatch(
        (0.2, 0.4), 7.5, 4.2, boxstyle="round,pad=0.2",
        facecolor='#f7fdf8', edgecolor='#c8e6c9', linestyle='--', linewidth=1.5
    )
    ax.add_patch(p3_rect)
    ax.text(0.5, 4.2, "Fase 3: Modelagem & Treinamento", fontsize=12, fontweight='bold', color=colors['phase3_border'])
    
    # Phase 4: Avaliação
    p4_rect = patches.FancyBboxPatch(
        (8.1, 0.4), 3.5, 4.2, boxstyle="round,pad=0.2",
        facecolor='#fdf7fd', edgecolor='#e1bee7', linestyle='--', linewidth=1.5
    )
    ax.add_patch(p4_rect)
    ax.text(8.4, 4.2, "Fase 4: Avaliação", fontsize=12, fontweight='bold', color=colors['phase4_border'])
    
    # Helper to draw nodes
    def draw_node(x, y, w, h, bg_color, border_color, title, subtitle):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=bg_color, edgecolor=border_color, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, fontsize=9, fontweight='bold', 
                color=colors['text_dark'], ha='center', va='center')
        ax.text(x + w/2, y + h*0.3, subtitle, fontsize=8, 
                color='#555555', ha='center', va='center')
        
    # Helper to draw arrows
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            label, xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor=colors['arrow'], edgecolor=colors['arrow'], 
                            arrowstyle="->", lw=1.5, shrinkA=2, shrinkB=2),
            fontsize=8, color='#444444', ha='center', va='bottom'
        )

    # --- Phase 1 Nodes ---
    # N1: Ortofotos GeoTIFF
    draw_node(0.5, 7.8, 2.2, 1.2, colors['phase1_bg'], colors['phase1_border'], 
              "Ortofotos GeoTIFF", "Geoportal IDE-DF")
    # N2: Slicing Rasterio
    draw_node(3.2, 7.8, 2.2, 1.2, colors['phase1_bg'], colors['phase1_border'], 
              "Fatiamento Espacial", "Recortes 640x640 (Rasterio)")
    # N3: Normalização e BGR
    draw_node(3.2, 5.8, 2.2, 1.2, colors['phase1_bg'], colors['phase1_border'], 
              "Conversão Cromática", "RGB -> BGR (OpenCV)")
    # N4: HDF5 Bruto
    draw_node(0.5, 5.8, 2.2, 1.2, colors['phase1_bg'], colors['phase1_border'], 
              "HDF5 Bruto", "dataset_v1_raw.h5")

    # --- Phase 2 Nodes ---
    # N5: Pseudo-Labelling
    draw_node(6.4, 7.8, 2.2, 1.2, colors['phase2_bg'], colors['phase2_border'], 
              "Pseudo-Labelling", "DeepForest (RetinaNet)")
    # N6: Exportador Roboflow
    draw_node(9.1, 7.8, 2.2, 1.2, colors['phase2_bg'], colors['phase2_border'], 
              "Exportação YOLO", "Geração de ZIP estruturado")
    # N7: Curadoria Visual
    draw_node(9.1, 5.8, 2.2, 1.2, colors['phase2_bg'], colors['phase2_border'], 
              "Curadoria Roboflow", "Revisão Manual de BBoxes")
    # N8: Particionamento Geográfico
    draw_node(6.4, 5.8, 2.2, 1.2, colors['phase2_bg'], colors['phase2_border'], 
              "Divisão Geográfica", "Setorização Treino (80%) / Val (20%)")

    # --- Phase 3 Nodes ---
    # N9: Novos HDF5s
    draw_node(0.5, 2.4, 2.2, 1.2, colors['phase3_bg'], colors['phase3_border'], 
              "Novos HDF5s", "dataset_treino / dataset_val")
    # N10: RAM Disk Loader
    draw_node(3.0, 2.4, 2.2, 1.2, colors['phase3_bg'], colors['phase3_border'], 
              "RAM Disk Loader", "Extração rápida em /dev/shm")
    # N11: YOLOv11m Fine-tuning
    draw_node(5.2, 0.8, 2.2, 1.2, colors['phase3_bg'], colors['phase3_border'], 
              "Fine-Tuning YOLO11m", "Backbone Freeze=10 / SGD")
    
    # --- Phase 4 Nodes ---
    # N12: Avaliação Estatística
    draw_node(8.7, 2.4, 2.2, 1.2, colors['phase4_bg'], colors['phase4_border'], 
              "Métricas Estatísticas", "mAP, Precisão, Recall, PR")
    # N13: Relatório Científico
    draw_node(8.7, 0.8, 2.2, 1.2, colors['phase4_bg'], colors['phase4_border'], 
              "Relatório Final", "LaTeX / Overleaf")

    # --- Arrows ---
    # Phase 1 Flow
    draw_arrow(2.7, 8.4, 3.2, 8.4)  # N1 -> N2
    draw_arrow(4.3, 7.8, 4.3, 7.0)  # N2 -> N3
    draw_arrow(3.2, 6.4, 2.7, 6.4)  # N3 -> N4
    
    # Connection to Phase 2
    draw_arrow(1.6, 7.0, 6.4, 8.4, "dataset_v1_raw.h5") # N4 -> N5
    
    # Phase 2 Flow
    draw_arrow(8.6, 8.4, 9.1, 8.4)  # N5 -> N6
    draw_arrow(10.2, 7.8, 10.2, 7.0) # N6 -> N7
    draw_arrow(9.1, 6.4, 8.6, 6.4)  # N7 -> N8
    
    # Connection to Phase 3
    draw_arrow(6.4, 6.4, 1.6, 3.6, "HDF5s Curados") # N8 -> N9
    
    # Phase 3 Flow
    draw_arrow(2.7, 3.0, 3.0, 3.0)  # N9 -> N10
    draw_arrow(4.1, 2.4, 5.2, 1.4, "YOLO Format") # N10 -> N11
    
    # Connection to Phase 4
    draw_arrow(7.4, 1.4, 8.7, 2.4, "weights / best.pt") # N11 -> N12
    draw_arrow(9.8, 2.4, 9.8, 2.0)  # N12 -> N13
    
    plt.tight_layout()
    os.makedirs('figuras', exist_ok=True)
    plt.savefig('figuras/pipeline.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Pipeline diagram generated successfully at figuras/pipeline.png")

if __name__ == '__main__':
    generate_pipeline_diagram()
