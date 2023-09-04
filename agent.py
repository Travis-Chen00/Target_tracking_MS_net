import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


def draw_grid_world():
    fig, ax = plt.subplots(figsize=(4, 4))

    center_x, center_y = 5, 5

    circle = patches.Circle((center_y, center_x), 0.4, fill=False, edgecolor="black")
    ax.add_patch(circle)

    ax.text(center_y, center_x - 0.1, "G0", ha='center', va='center_baseline', fontsize=10, color='blue')

    ax.plot([center_y, center_y], [center_x, center_x + 0.4], color="black", linewidth=1.5)

    blue_rects = [
        (center_y - 1, center_x, "G3") if center_y - 1 >= 4 else None,
        (center_y + 1, center_x, "G4") if center_y + 1 <= 6 else None,
        (center_y, center_x - 1, "G2") if center_x - 1 >= 4 else None,
        (center_y, center_x + 1, "G1 / S0") if center_x + 1 <= 6 else None
    ]
    for coords in blue_rects:
        if coords:
            bx, by, label = coords
            rect = patches.Rectangle((bx - 0.5, by - 0.5), 1, 1, linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)

            # Add text to the cell
            ax.text(bx, by, label, ha='center', va='center', fontsize=10, color='blue')

    ax.set_xlim(3, 7)
    ax.set_ylim(3, 7)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("agent.jpg", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()


draw_grid_world()
