import json
import time

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')


def plot_results(outfile='results.jpg'):
    """Plot histplot of token_per_sec"""

    # Read in result data
    with open('results.jsonl', 'r', encoding="UTF-8") as f:
        result_data = [json.loads(line) for line in f]

    # Plot the histplot and savefig
    sns.histplot([entry['token_per_sec'] for entry in result_data])
    plt.savefig(outfile)

    # Inform user and tidy up figure
    print(f'Result figure saved as {outfile}')
    plt.clf()

def print_evaluation_summary():
    """Prints evaluation summary details"""

    # Read in result data
    with open('results.jsonl', 'r', encoding="UTF-8") as f:
        result_data = [json.loads(line) for line in f]

    # Get specific data
    time_taken = [entry['time_taken'] for entry in result_data]
    total_tokens = [entry['output_tokens'] for entry in result_data]
    token_per_sec = [entry['token_per_sec'] for entry in result_data]

    # Print evaluation summary
    formatted_time = time.strftime("%Hh %Mm %Ss", time.gmtime(sum(time_taken)))
    print('-------\nEvaluation Summary: \n-------\n' +
          f'Total prompts: {len(time_taken)}\n' +
          f'Total completions: {len(time_taken)*5}\n' +
          f'Time taken: {formatted_time}\n' +
          f'Total tokens produced: {sum(total_tokens)}\n' +
          f'Average tokens per second: {round(sum(token_per_sec)/len(token_per_sec), 2)}')


if __name__ == "__main__":
    plot_results()
    print_evaluation_summary()
