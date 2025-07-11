
import matplotlib.pyplot as plt
import seaborn as sns

def create_training_plots(train_accuracies, val_accuracies, 
                         train_f1s, val_f1s, fold, model_dir):
    """
    Create and save training progression plots for a specific fold.
    
    Args:
        train_accuracies: List of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch
        train_f1s: List of training F1 scores per epoch
        val_f1s: List of validation F1 scores per epoch
        fold: Current fold number
        model_dir: Directory to save the plots
    """
    # Create visualizations directory
    viz_dir = model_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    epochs = range(1, len(train_f1s) + 1)
    
    # Plot all metrics on the same graph
    ax.plot(epochs, train_f1s, 'tab:blue', linewidth=2, label='Train F1', alpha=0.4)
    ax.plot(epochs, val_f1s, 'tab:blue', linewidth=2, label='Validation F1', alpha=0.85)
    ax.plot(epochs, train_accuracies, 'tab:orange', linewidth=2, label='Train Accuracy', alpha=0.4)
    ax.plot(epochs, val_accuracies, 'tab:orange', linewidth=2, label='Validation Accuracy', alpha=0.85)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Performance (Accuracy/F1)', fontsize=12)
    ax.set_title(f'Training Progression - Fold {fold+1}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(.7, 1)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = viz_dir / f"training_progression_fold_{fold+1}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_plot(fold_metrics, model_dir):
    """
    Create a summary plot showing all folds' final performance.
    
    Args:
        fold_metrics: List of dictionaries containing fold results
        model_dir: Directory to save the plot
    """
    # Create visualizations directory
    viz_dir = model_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    folds = [m['fold'] + 1 for m in fold_metrics]
    f1_scores = [m['val_f1'] for m in fold_metrics]
    accuracies = [m['val_acc'] for m in fold_metrics]
    recalls = [m['val_recall'] for m in fold_metrics]
    precisions = [m['val_precision'] for m in fold_metrics]
    
    # Plot 1: F1 Scores across folds
    bars1 = ax1.bar(folds, f1_scores, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1)
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Scores Across Folds', fontsize=14, fontweight='bold')
    ax1.set_ylim(0.8, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Accuracies across folds
    bars2 = ax2.bar(folds, accuracies, color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1)
    ax2.set_xlabel('Fold', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracies Across Folds', fontsize=14, fontweight='bold')
    ax2.set_ylim(0.8, 1)
    ax2.grid(True, alpha=0.3)
    
    for bar, value in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Recalls across folds
    bars3 = ax3.bar(folds, recalls, color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1)
    ax3.set_xlabel('Fold', fontsize=12)
    ax3.set_ylabel('Recall', fontsize=12)
    ax3.set_title('Recalls Across Folds', fontsize=14, fontweight='bold')
    ax3.set_ylim(0.8, 1)
    ax3.grid(True, alpha=0.3)
    
    for bar, value in zip(bars3, recalls):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Precisions across folds
    bars4 = ax4.bar(folds, precisions, color='gold', alpha=0.7, edgecolor='orange', linewidth=1)
    ax4.set_xlabel('Fold', fontsize=12)
    ax4.set_ylabel('Precision', fontsize=12)
    ax4.set_title('Precisions Across Folds', fontsize=14, fontweight='bold')
    ax4.set_ylim(0.8, 1)
    ax4.grid(True, alpha=0.3)
    
    for bar, value in zip(bars4, precisions):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    summary_path = viz_dir / "cross_validation_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved to: {summary_path}")