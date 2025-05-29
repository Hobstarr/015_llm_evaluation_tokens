import json

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')


def plot_results(infiles=['results_single.jsonl', 'results.jsonl'],
                 labels=['one completion per prompt', 'five completions per prompt'],
                 colours=['forestgreen', 'cornflowerblue'],
                 outfile='all_results.jpg'):
    """Plot histplot of token_per_sec"""

    # Read in result data
    for i, results_file in enumerate(infiles):
        with open(results_file, 'r', encoding="UTF-8") as f:
            result_data = [json.loads(line) for line in f]

    # Plot the histplot and savefig
        sns.histplot([entry['token_per_sec'] for entry in result_data],
                     label=labels[i], color=colours[i], alpha=0.7,
                     binwidth=5, linewidth=0.5)

    plt.title('Evaluating Tokens/Second for Different Numbers of Generations')
    plt.xlabel('Tokens / Second')
    plt.legend()
    plt.xlim(0, 250)
    plt.savefig(outfile)

    # Inform user and tidy up figure
    print(f'Result figure saved as {outfile}')
    plt.clf()


plot_results()
