## rumz 3/25

import matplotlib.pyplot as plt


def graph(df_bubbles, save=False, savepath=""):

    fig, ax = plt.subplots(figsize=(4, 20))
    ax.plot(df_bubbles['Bubble_Count'], df_bubbles['Depth_cm'], color='tab:blue')
    if savepath != "":
        plt.title(f"{savepath.parent.name}")
    ax.set_ylabel("Depth (cm)", fontweight='bold')
    ax.set_xlabel('Num of Bubbles')
    ax.invert_yaxis()  # invert y-axis
    ax.grid(True)
    
    plt.tight_layout()

    
    # axs[1].scatter(df_bubbles['Normalized_Count'], df_bubbles['Depth_cm'], color='tab:purple', marker='x')
    # axs[1].invert_yaxis()
    # axs[1].set_xlabel('Count cm^2')
    # axs[1].grid(True)

    # axs[2].scatter(df_bubbles['Bubble_Number_Density'], df_bubbles['Depth_cm'], color='tab:orange', marker='s')
    # axs[2].invert_yaxis()
    # axs[2].set_xlabel('Bubble Number Density')
    # axs[2].grid(True)
    
    # axs[3].scatter(df_bubbles['Avg_Bubble_Size'], df_bubbles['Depth_cm'], color='tab:blue', marker='o')
    # axs[3].invert_yaxis()
    # axs[3].set_xlabel('Avg Size')
    # axs[3].grid(True)

    # axs[4].scatter(df_bubbles['Avg_Elongation'], df_bubbles['Depth_cm'], color='tab:green', marker='s')
    # axs[4].invert_yaxis()
    # axs[4].set_xlabel('Elongation')
    # axs[4].grid(True)

    # axs[5].scatter(df_bubbles['Avg_Eccentricity'], df_bubbles['Depth_cm'], color='tab:red', marker='o')
    # axs[5].invert_yaxis()
    # axs[5].set_xlabel('Eccentricity')
    # axs[5].grid(True)

    # axs[6].scatter(df_bubbles['Avg_Orientation'], df_bubbles['Depth_cm'], color='tab:purple', marker='x')
    # #axs[6].invert_yaxis()
    # axs[6].set_xlabel('Orientation')
    # axs[6].grid(True)
    
    # === Set common y-axis label ===
    # plt.set_ylabel("Depth (cm)", fontweight='bold')
    
    # plt.tight_layout()
    
    if save == True:
        DSName = savepath.parent.name
        filename = f"{DSName}_bubbleGraph.png"
        plt.savefig(f"{savepath}/{filename}", format='png', dpi=300, bbox_inches='tight')
        # readme_path = savepath / "README.md"
        # with readme_path.open("a") as f:
        #     f.write(f"\n###  Output File: {filename}\n- {DSName}: Graph of bubble stats in core sections\n")

    plt.show()