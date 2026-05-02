"""
Evaluation Visualization Script for SimpleNet
Generates graphs and visual representations of training results
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
from PIL import Image

import torch

import backbones
import simplenet
import metrics
from datasets import mvtec

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def read_results_csv(results_path):
    """Read results from CSV file."""
    csv_path = os.path.join(results_path, "results.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: results.csv not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    return df


def read_tensorboard_logs(results_path):
    """Try to read TensorBoard logs if available."""
    # Look for tensorboard logs in subdirectories
    tb_logs = {}
    for root, dirs, files in os.walk(results_path):
        if 'tb' in dirs:
            tb_dir = os.path.join(root, 'tb')
            # TensorBoard logs are in event files, but we'll just note the path
            tb_logs[root] = tb_dir
    return tb_logs


def plot_metrics_comparison(df, save_path):
    """Create bar chart comparing metrics across classes."""
    if df is None or len(df) == 0:
        print("No data to plot for metrics comparison")
        return
    
    # Filter out 'Mean' row if present
    df_plot = df[df.iloc[:, 0] != 'Mean'].copy()
    
    if len(df_plot) == 0:
        print("No class data to plot")
        return
    
    # Get class names and metrics
    class_names = df_plot.iloc[:, 0].values
    metrics = df_plot.columns[1:].tolist()
    
    # Create subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 8))
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = df_plot[metric].values
        
        bars = ax.bar(range(len(class_names)), values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} by Class', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    save_file = os.path.join(save_path, "metrics_comparison.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_file}")
    plt.close()


def plot_metrics_summary(df, save_path):
    """Create summary visualization of all metrics."""
    if df is None or len(df) == 0:
        print("No data to plot for summary")
        return
    
    # Filter out 'Mean' row
    df_plot = df[df.iloc[:, 0] != 'Mean'].copy()
    
    if len(df_plot) == 0:
        return
    
    class_names = df_plot.iloc[:, 0].values
    metrics = df_plot.columns[1:].tolist()
    
    # Create grouped bar chart
    x = np.arange(len(class_names))
    width = 0.8 / len(metrics)
    
    fig, ax = plt.subplots(figsize=(max(12, len(class_names)*0.8), 8))
    
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * width
        values = df_plot[metric].values
        bars = ax.bar(x + offset, values, width, label=metric, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if height > 0.05:  # Only show label if bar is tall enough
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('All Metrics Comparison Across Classes', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_file = os.path.join(save_path, "metrics_summary.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_file}")
    plt.close()


def plot_performance_heatmap(df, save_path):
    """Create heatmap of metrics across classes."""
    if df is None or len(df) == 0:
        print("No data for heatmap")
        return
    
    df_plot = df[df.iloc[:, 0] != 'Mean'].copy()
    
    if len(df_plot) == 0:
        return
    
    class_names = df_plot.iloc[:, 0].values
    metrics = df_plot.columns[1:].tolist()
    
    # Prepare data for heatmap
    data = df_plot[metrics].values.T
    
    fig, ax = plt.subplots(figsize=(max(10, len(class_names)*0.7), max(6, len(metrics)*0.8)))
    
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(metrics)
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(class_names)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=9)
    
    ax.set_title("Performance Heatmap: Metrics vs Classes", fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    save_file = os.path.join(save_path, "performance_heatmap.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_file}")
    plt.close()


def plot_statistics_summary(df, save_path):
    """Create statistical summary visualization."""
    if df is None or len(df) == 0:
        print("No data for statistics")
        return
    
    df_plot = df[df.iloc[:, 0] != 'Mean'].copy()
    
    if len(df_plot) == 0:
        return
    
    metrics = df_plot.columns[1:].tolist()
    
    # Calculate statistics
    stats_data = []
    for metric in metrics:
        values = df_plot[metric].values
        stats_data.append({
            'Metric': metric,
            'Mean': np.mean(values),
            'Std': np.std(values),
            'Min': np.min(values),
            'Max': np.max(values),
            'Median': np.median(values)
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Mean scores
    ax = axes[0, 0]
    bars = ax.bar(stats_df['Metric'], stats_df['Mean'], color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_title('Mean Scores by Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Score', fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, stats_df['Mean']):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Standard deviation
    ax = axes[0, 1]
    bars = ax.bar(stats_df['Metric'], stats_df['Std'], color='coral', alpha=0.8, edgecolor='black')
    ax.set_title('Standard Deviation by Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Std Dev', fontsize=11)
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, stats_df['Std']):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.005,
               f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Min-Max range
    ax = axes[1, 0]
    x_pos = np.arange(len(stats_df))
    ax.bar(x_pos - 0.2, stats_df['Min'], 0.4, label='Min', color='lightcoral', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + 0.2, stats_df['Max'], 0.4, label='Max', color='lightgreen', alpha=0.8, edgecolor='black')
    ax.set_title('Min-Max Range by Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stats_df['Metric'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Box plot
    ax = axes[1, 1]
    data_for_box = [df_plot[metric].values for metric in metrics]
    bp = ax.boxplot(data_for_box, labels=metrics, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    ax.set_title('Score Distribution by Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Statistical Summary of Model Performance', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_file = os.path.join(save_path, "statistics_summary.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_file}")
    plt.close()


def create_evaluation_report(df, save_path):
    """Create a text report with summary statistics."""
    if df is None:
        return
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SIMPLENET EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    df_plot = df[df.iloc[:, 0] != 'Mean'].copy()
    
    if len(df_plot) > 0:
        metrics = df_plot.columns[1:].tolist()
        class_names = df_plot.iloc[:, 0].values
        
        report_lines.append(f"Number of Classes Evaluated: {len(class_names)}")
        report_lines.append(f"Metrics Evaluated: {', '.join(metrics)}")
        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append("PER-CLASS RESULTS")
        report_lines.append("-" * 80)
        
        for idx, class_name in enumerate(class_names):
            report_lines.append(f"\n{class_name}:")
            for metric in metrics:
                value = df_plot[metric].iloc[idx]
                report_lines.append(f"  {metric}: {value:.4f}")
        
        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append("OVERALL STATISTICS")
        report_lines.append("-" * 80)
        
        for metric in metrics:
            values = df_plot[metric].values
            report_lines.append(f"\n{metric}:")
            report_lines.append(f"  Mean:   {np.mean(values):.4f}")
            report_lines.append(f"  Std:    {np.std(values):.4f}")
            report_lines.append(f"  Min:    {np.min(values):.4f}")
            report_lines.append(f"  Max:    {np.max(values):.4f}")
            report_lines.append(f"  Median: {np.median(values):.4f}")
    
    # Check for Mean row
    mean_row = df[df.iloc[:, 0] == 'Mean']
    if len(mean_row) > 0:
        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append("MEAN SCORES (from CSV)")
        report_lines.append("-" * 80)
        for metric in metrics:
            value = mean_row[metric].iloc[0]
            report_lines.append(f"  {metric}: {value:.4f}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Save report
    report_file = os.path.join(save_path, "evaluation_report.txt")
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"Saved: {report_file}")
    print("\n" + report_text)


# ------------------------------------------------------------------
# Single-class, model-based visualizations (merged from visualize_bottle_model)
# ------------------------------------------------------------------

def generate_single_class_visuals(results_root, class_name, dataset_path):
    """
    Use a trained SimpleNet checkpoint for a single class to create:
    - Image-wise ROC curve
    - Pixel-wise ROC curve (if masks available)
    - Score histogram
    - Mean anomaly heatmap
    - One example anomaly image with heatmap overlay

    results_root: path like results/Full_Training/simplenet_full/full_run
    """
    print(f"\n[Single-class visuals] Class='{class_name}'  results_root='{results_root}'")

    visualization_dir = os.path.join(results_root, f"{class_name}_visualizations")
    os.makedirs(visualization_dir, exist_ok=True)

    device = torch.device("cpu")

    # ----- Build test dataloader -----
    test_dataset = mvtec.MVTecDataset(
        dataset_path,
        classname=class_name,
        resize=329,
        imagesize=288,
        split=mvtec.DatasetSplit.TEST,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
    )

    # ----- Recreate model & load checkpoint -----
    backbone_name = "wideresnet50"
    layers_to_extract_from = ["layer2", "layer3"]

    backbone = backbones.load(backbone_name)
    backbone.name, backbone.seed = backbone_name, None

    model = simplenet.SimpleNet(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=layers_to_extract_from,
        device=device,
        input_shape=test_dataset.imagesize,
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        patchsize=3,
        embedding_size=256,
        meta_epochs=40,
        gan_epochs=4,
        noise_std=0.015,
        dsc_hidden=1024,
        dsc_layers=2,
        dsc_margin=0.5,
        pre_proj=1,
    )

    models_dir = os.path.join(results_root, "models")
    model_dir = os.path.join(models_dir, "0")
    dataset_name = f"mvtec_{class_name}"
    model.set_model_dir(model_dir, dataset_name)

    ckpt_path = os.path.join(model.ckpt_dir, "ckpt.pth")
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found for class '{class_name}': {ckpt_path}")
        return

    state_dict = torch.load(ckpt_path, map_location=device)
    if "discriminator" in state_dict:
        model.discriminator.load_state_dict(state_dict["discriminator"])
    if hasattr(model, "pre_projection") and "pre_projection" in state_dict:
        model.pre_projection.load_state_dict(state_dict["pre_projection"])

    # NOTE: do not call model.eval(); SimpleNet overrides train(...).

    # ----- Inference -----
    scores, segmentations, features, labels_gt, masks_gt = model.predict(test_loader)

    scores = np.array(scores).reshape(-1)
    labels_gt = np.array(labels_gt).astype(int)

    if len(masks_gt) > 0:
        segmentations_np = np.array(segmentations)
        masks_gt_np = np.array(masks_gt)
    else:
        segmentations_np = None
        masks_gt_np = None

    # ----- Image-level metrics & ROC -----
    img_metrics = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt)
    img_auroc = img_metrics["auroc"]

    plt.figure(figsize=(7, 6))
    plt.plot(img_metrics["fpr"], img_metrics["tpr"], label=f"Image AUROC = {img_auroc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Image-wise ROC Curve ({class_name})")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    roc_path = os.path.join(visualization_dir, "image_roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Score histogram
    plt.figure(figsize=(7, 6))
    normal_scores = scores[labels_gt == 0]
    anomaly_scores = scores[labels_gt == 1]
    bins = 30
    plt.hist(
        normal_scores,
        bins=bins,
        alpha=0.6,
        label="Normal",
        color="steelblue",
        edgecolor="black",
    )
    plt.hist(
        anomaly_scores,
        bins=bins,
        alpha=0.6,
        label="Anomalies",
        color="indianred",
        edgecolor="black",
    )
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.title(f"Score Distribution ({class_name})")
    plt.legend()
    plt.grid(alpha=0.3)
    hist_path = os.path.join(visualization_dir, "score_histogram.png")
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()

    pixel_auroc = None
    pro_auc = None

    # ----- Pixel-level metrics & heatmap -----
    if segmentations_np is not None and masks_gt_np is not None and masks_gt_np.size > 0:
        pix_metrics = metrics.compute_pixelwise_retrieval_metrics(
            segmentations_np, masks_gt_np
        )
        pixel_auroc = pix_metrics["auroc"]
        pro_auc = metrics.compute_pro(
            np.squeeze(np.array(masks_gt_np)), segmentations_np
        )

        plt.figure(figsize=(7, 6))
        plt.plot(
            pix_metrics["fpr"],
            pix_metrics["tpr"],
            label=f"Pixel AUROC = {pixel_auroc:.3f}",
        )
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Pixel-wise ROC Curve ({class_name})")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        pixel_roc_path = os.path.join(visualization_dir, "pixel_roc_curve.png")
        plt.savefig(pixel_roc_path, dpi=300, bbox_inches="tight")
        plt.close()

        anomaly_idxs = np.where(labels_gt == 1)[0]
        if len(anomaly_idxs) > 0:
            mean_map = segmentations_np[anomaly_idxs].mean(axis=0)
            plt.figure(figsize=(6, 6))
            plt.imshow(mean_map, cmap="hot")
            plt.colorbar(label="Anomaly intensity")
            plt.title(f"Mean Anomaly Heatmap ({class_name} anomalies)")
            heatmap_path = os.path.join(visualization_dir, "mean_anomaly_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            plt.close()

    # ----- One example anomaly image with heatmap overlay -----
    anomaly_indices = np.where(labels_gt == 1)[0]
    if len(anomaly_indices) > 0:
        example_idx = anomaly_indices[np.argmax(scores[anomaly_indices])]
    else:
        example_idx = int(np.argmax(scores))

    image_path = test_dataset.data_to_iterate[example_idx][2]
    example_seg = segmentations[example_idx]

    example_path = os.path.join(visualization_dir, "example_anomaly.png")

    orig_img = Image.open(image_path).convert("RGB")
    orig_np = np.array(orig_img)

    seg = example_seg
    if isinstance(seg, torch.Tensor):
        seg = seg.detach().cpu().numpy()
    if seg.ndim > 2:
        seg = np.squeeze(seg)
    seg_min, seg_max = seg.min(), seg.max()
    if seg_max > seg_min:
        seg_norm = (seg - seg_min) / (seg_max - seg_min)
    else:
        seg_norm = seg

    seg_img = Image.fromarray((seg_norm * 255).astype(np.uint8))
    seg_img = seg_img.resize((orig_np.shape[1], orig_np.shape[0]), Image.BILINEAR)
    seg_resized = np.array(seg_img) / 255.0

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(orig_np)
    plt.title(f"{class_name} - Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(orig_np)
    hm = plt.imshow(seg_resized, cmap="jet", alpha=0.5)
    plt.title(f"{class_name} - Anomaly Heatmap")
    plt.axis("off")
    plt.colorbar(hm, fraction=0.046, pad=0.04, label="Anomaly score")

    plt.tight_layout()
    plt.savefig(example_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Summary text
    summary_path = os.path.join(visualization_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Class: {class_name}\n")
        f.write(f"Image-wise AUROC: {img_auroc:.4f}\n")
        if pixel_auroc is not None:
            f.write(f"Pixel-wise AUROC: {pixel_auroc:.4f}\n")
        if pro_auc is not None:
            f.write(f"PRO AUC: {pro_auc:.4f}\n")

    print(f"[Single-class visuals] Saved to: {visualization_dir}")


def main():
    """Main function to generate all evaluations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate evaluation visualizations')
    parser.add_argument('--results_path', type=str, default='results',
                       help='Path to results directory')
    parser.add_argument('--project', type=str, default=None,
                       help='Project name (e.g., Initial_Test, Full_Training)')
    parser.add_argument('--group', type=str, default=None,
                       help='Log group name')
    parser.add_argument('--run', type=str, default=None,
                       help='Run name')
    parser.add_argument('--class_name', type=str, default=None,
                       help='Optional: single MVTec class to generate model-based visuals for (e.g., bottle)')
    parser.add_argument('--dataset_path', type=str, default='datasets/mvtec_anomaly_detection',
                       help='Path to MVTec dataset root (for class-wise visuals)')
    
    args = parser.parse_args()
    
    # Determine results path
    if args.project and args.group and args.run:
        results_path = os.path.join(args.results_path, args.project, args.group, args.run)
    else:
        # Try to find the most recent results
        results_path = args.results_path
        if os.path.exists(results_path):
            # Look for results.csv in subdirectories
            for root, dirs, files in os.walk(results_path):
                if 'results.csv' in files:
                    results_path = root
                    break
    
    if not os.path.exists(results_path):
        print(f"Error: Results path not found: {results_path}")
        print("Usage: python generate_evaluations.py --results_path results --project Initial_Test --group simplenet_test --run bottle_initial")
        return
    
    print(f"Reading results from: {results_path}")
    
    # Create evaluations folder
    eval_folder = os.path.join(results_path, "evaluations")
    os.makedirs(eval_folder, exist_ok=True)
    
    # Read results (may be missing if training was interrupted)
    df = read_results_csv(results_path)
    
    if df is not None:
        print(f"\nFound results for {len(df)} entries")
        print(f"Metrics: {list(df.columns)}")
        print("\nGenerating CSV-based visualizations...")
        
        # Generate CSV-based visualizations
        plot_metrics_comparison(df, eval_folder)
        plot_metrics_summary(df, eval_folder)
        plot_performance_heatmap(df, eval_folder)
        plot_statistics_summary(df, eval_folder)
        create_evaluation_report(df, eval_folder)
    else:
        print("No results.csv found – skipping CSV-based metric plots.")

    # Optional: generate model-based visuals for a single class
    if args.class_name is not None:
        try:
            generate_single_class_visuals(
                results_root=results_path,
                class_name=args.class_name,
                dataset_path=args.dataset_path,
            )
        except Exception as e:
            print(f"[WARN] Single-class visualization failed: {e}")
    
    print(f"\n[SUCCESS] All evaluations saved to: {eval_folder}")
    print("\nGenerated files:")
    if df is not None:
        print("  - metrics_comparison.png (bar charts per metric)")
        print("  - metrics_summary.png (grouped bar chart)")
        print("  - performance_heatmap.png (heatmap visualization)")
        print("  - statistics_summary.png (statistical analysis)")
        print("  - evaluation_report.txt (text summary)")
    if args.class_name is not None:
        print(f"  - {args.class_name}_visualizations/* (class-wise model outputs)")


if __name__ == "__main__":
    main()

