"""Utilitaires partagés par les notebooks du cours.

Pour l'instant : `flowchart()`, qui dessine un organigramme en SORTIE d'une cellule
de code (matplotlib). On évite ainsi Mermaid, qui ne s'affiche que dans certaines
versions de JupyterLab et reste blanc partout ailleurs.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def flowchart(steps, title=None, width=8.5, box_h=0.62, gap=0.45,
              facecolor="#e7f0fb", edgecolor="#2b6cb0", fontsize=11):
    """Dessine un organigramme vertical : une boîte par étape, flèches entre elles.

    `steps` : liste de chaînes (du haut vers le bas).
    Le diagramme est rendu via `plt.show()` -> visible dans tout Jupyter, nbconvert
    et l'aperçu GitHub.
    """
    n = len(steps)
    unit = box_h + gap
    fig, ax = plt.subplots(figsize=(width, n * unit + 0.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(-gap, n * unit)
    ax.axis("off")
    for i, label in enumerate(steps):
        y = (n - 1 - i) * unit
        ax.add_patch(FancyBboxPatch((1, y), 8, box_h,
                     boxstyle="round,pad=0.08", linewidth=1.6,
                     facecolor=facecolor, edgecolor=edgecolor))
        ax.text(5, y + box_h / 2, label, ha="center", va="center", fontsize=fontsize)
        if i < n - 1:
            ax.annotate("", xy=(5, y - gap), xytext=(5, y),
                        arrowprops=dict(arrowstyle="-|>", color=edgecolor, lw=1.8))
    if title:
        ax.set_title(title, fontsize=fontsize + 2, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.show()
