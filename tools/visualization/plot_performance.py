import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

def plot_performance(results_file):
    # Read the CSV file
    df = pd.read_csv(results_file)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Size', y='GFLOPS', hue='Version')
    
    plt.title('GEMM Performance Comparison')
    plt.xlabel('Matrix Size')
    plt.ylabel('Performance (GFLOPS)')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add value labels on top of each bar
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.1f')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(results_file).parent / 'performance_plot.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_performance.py <results_csv_file>")
        sys.exit(1)
    
    plot_performance(sys.argv[1]) 