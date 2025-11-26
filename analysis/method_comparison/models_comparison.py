import pandas as pd #dataframe manipulation
import matplotlib.pyplot as plt #data visualization

def get_accuracy(df):
    """
    Calculate the accuracy for each ontology in the given dataframe.

    Parameters:
        df (DataFrame): The dataframe containing predictions and true values
                         for each ontology. Columns are expected to include
                         pairs of columns for each ontology with the suffixes '_C'
                         (for true values) and '_M' (for model predictions).

    """
    for column in df:
        df[column] = df[column].fillna('unknown')
    
    suffixes = ['CLO', 'CL', 'UBERON', 'BTO']

    accuracies = {}
    for suffix in suffixes: #for each ontology the accuracy is calculated
        true_col = f'{suffix}_C'
        pred_col = f'{suffix}_M'
        true_pos=0
        false_pos = 0
        if true_col in df.columns and pred_col in df.columns:
            for i in range(len(df)):
                true_val = df.iloc[i][true_col]
                pred_val = df.iloc[i][pred_col]
                if true_val == "-" and pred_val == "-":
                    continue
                if true_val == pred_val:
                    true_pos += 1
                else:
                    false_pos += 1
            if (true_pos + false_pos) > 0:
                accuracy = true_pos / (true_pos + false_pos)
            else:
                accuracy = None
            accuracies[suffix] = accuracy
        else:
            accuracies[suffix] = None
    return accuracies

def plot_accuracies(models_data, model_names):
    """
    Plot the accuracies of different models for each ontology.
    """
    accuracies = [get_accuracy(df) for df in models_data]
    ontologies = ['CLO', 'CL', 'UBERON', 'BTO']
    data = {ont: [acc[ont] for acc in accuracies] for ont in ontologies}
    df_plot = pd.DataFrame(data, index=model_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 4
    spacing = 5
    positions = [i * (bar_width * len(ontologies) + spacing) for i in range(len(models_data))]

    for i, ont in enumerate(ontologies):
        bar_positions = [pos + i * bar_width for pos in positions]
        ax.bar(bar_positions, df_plot[ont], width=bar_width, label=ont)

        # 🔹 Aumentamos el tamaño del texto encima de cada barra
        for j, pos in enumerate(bar_positions):
            ax.annotate(f'{df_plot[ont][j]:.3f}',
                        (pos, df_plot[ont][j]),
                        ha='center', va='center', xytext=(0, 10),
                        textcoords='offset points',
                        fontsize=12, fontweight='normal', color='black')

    # 🔹 Aumentamos tamaños de letra en los ejes y el título
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Ontology Precision by Method', fontsize=16, fontweight='bold')

    ax.set_ylim(0, 1)
    ax.set_xticks([pos + (bar_width * (len(ontologies) - 1)) / 2 for pos in positions])
    ax.set_xticklabels(model_names, fontsize=12, fontweight='bold')

    # 🔹 Aumentamos tamaño de los números en eje Y y de la leyenda
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12, title='Ontology', title_fontsize=13)

    plt.tight_layout()
    plt.show()


def main():
    df_comparison_RAG_5mini_reduced_inference_index = pd.read_csv("RAG_reduced_info.csv", header=0) # our RAG approach
    df_comparison_RAG_4o_Bioportal_RAG = pd.read_csv("RAG_with_Bioportal.csv", header=0) #RAG with Biportal
    df_comparison_4o_mini = pd.read_csv("base_model.csv", header=0) #base model
    df_comparison_4o_mini_ft = pd.read_csv("finetuned-model.csv", header=0) #fine-tuned model
    df_RAG_full_ontologies = pd.read_csv("RAG_full_ontologies.csv", header=0)

    models_data = [df_comparison_4o_mini,df_comparison_4o_mini_ft,df_RAG_full_ontologies,df_comparison_RAG_4o_Bioportal_RAG,df_comparison_RAG_5mini_reduced_inference_index]
    model_names = ['Base model', 'Fine-tuned model', 'RAG full ontologies','RAG+Bioportal',"RAG reduced information"]

    plot_accuracies(models_data, model_names)

if __name__ == "__main__":
    main()