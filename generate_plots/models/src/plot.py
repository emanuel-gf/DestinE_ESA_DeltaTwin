import matplotlib.pyplot as plt
import numpy as np
import os
from loguru import logger
import json
from typing import List

def generate_plot_band(x_np: np.ndarray, gt_np: np.ndarray, pred_np: np.ndarray, bands: list, cmap: str, output_dir: str) -> None:
    """
    Visualize the results with a simple histogram comparison for prediction vs reference.

    Args:
        x_np: Input data (L1C) array with shape [H, W, C]
        gt_np: Ground truth data (L2A) with shape [H, W, C]
        pred_np: Predicted data (L2A) array with shape [H, W, C]
        bands: List of band names
        cmap: Colormap to use for visualization
        output_dir: Directory to save the output images
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        for idx, band in enumerate(bands):
            # Create a figure with images and a simple histogram
            fig = plt.figure(figsize=(20, 12))

            # Define a grid layout - 2 rows, 4 columns with bigger image row
            grid = plt.GridSpec(2, 4, height_ratios=[2, 1], hspace=0.3, wspace=0.3)

            # Top row - images
            ax_img1 = plt.subplot(grid[0, 0])
            ax_img2 = plt.subplot(grid[0, 1])
            ax_img3 = plt.subplot(grid[0, 2])
            ax_img4 = plt.subplot(grid[0, 3])

            # Bottom row - just one histogram comparing prediction and reference
            ax_hist = plt.subplot(grid[1, :])

            # Get the data for this band
            band_x = x_np[:, :, idx]
            band_gt = gt_np[:, :, idx]
            band_pred = pred_np[:, :, idx]

            # Calculate the difference
            diff_target_pred = (np.abs(band_gt - band_pred) / band_gt) * 100

            # Plot images
            im1 = ax_img1.imshow(band_x, cmap=cmap, vmin=0, vmax=1)
            ax_img1.set_title(f"Input L1C - Band: {band}", fontsize=14)
            ax_img1.axis('off')
            plt.colorbar(im1, ax=ax_img1, fraction=0.046, pad=0.04)

            im2 = ax_img2.imshow(band_gt, cmap=cmap, vmin=0, vmax=1)
            ax_img2.set_title(f"Reference L2A Sen2Cor - Band: {band}", fontsize=14)
            ax_img2.axis('off')
            plt.colorbar(im2, ax=ax_img2, fraction=0.046, pad=0.04)

            im3 = ax_img3.imshow(band_pred, cmap=cmap, vmin=0, vmax=1)
            ax_img3.set_title(f"Prediction L2A - Band: {band}", fontsize=14)
            ax_img3.axis('off')
            plt.colorbar(im3, ax=ax_img3, fraction=0.046, pad=0.04)

            im4 = ax_img4.imshow(diff_target_pred, cmap='plasma', vmin=0, vmax=100)
            ax_img4.set_title(f"Relative Error [%] - {band}", fontsize=14)
            ax_img4.axis('off')
            plt.colorbar(im4, ax=ax_img4, fraction=0.046, pad=0.04)

            # Simple histogram comparison
            # Filter out zeros and NaN values
            gt_data = band_gt[band_gt > 0].flatten()
            pred_data = band_pred[band_pred > 0].flatten()
            # Find common x-axis limits
            min_val = min(gt_data.min(), pred_data.min())
            max_val = max(np.percentile(gt_data, 98), np.percentile(pred_data, 98))

            # Create bins
            bins = np.linspace(min_val, max_val, 100)

            # Plot histograms
            ax_hist.hist(gt_data, bins=bins, alpha=0.5, color='green', label='Reference L2A')
            ax_hist.hist(pred_data, bins=bins, alpha=0.5, color='red', label='Prediction L2A')
            ax_hist.set_title(f"Histogram Comparison - Band {band}", fontsize=14)
            ax_hist.set_xlabel("Pixel Value", fontsize=12)
            ax_hist.set_ylabel("Frequency", fontsize=12)
            ax_hist.legend(fontsize=12)
            ax_hist.set_xlim(0,1)

            # Add metrics
            rmse = np.sqrt(np.mean((band_gt - band_pred)**2))
            mae = np.mean(np.abs(band_gt - band_pred))

            # Add metrics as text to the plot
            ax_hist.text(0.99, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}',
                        transform=ax_hist.transAxes, ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                        fontsize=12)
            # Save the figure
            fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
            fig.savefig(f"{output_dir}/{band}.svg", dpi=300, bbox_inches='tight')
            logger.info(f"{output_dir}/{band}.svg")
            plt.close(fig)

        logger.success(f"Visualizations with histograms generated in {output_dir}")
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
        import traceback
        logger.error(traceback.format_exc())


def generate_tci_plot(x_np: np.ndarray, gt_np: np.ndarray, pred_np: np.ndarray, bands: list, output_dir: str) -> None:
    """
    Generate True Color Image (RGB composite) plots for both input and predicted data.

    Args:
        x_np: Input data array with shape [H, W, C]
        gt_np: Ground true L2A data array with shape [H, W, C]
        pred_np: Predicted data array with shape [H, W, C]
        bands: List of band names
        output_dir: Directory to save the output images
    """
    try:
        # Find indices for RGB bands (B04-Red, B03-Green, B02-Blue)
        rgb_indices = []
        for rgb_band in bands:
            if rgb_band in bands:
                rgb_indices.append(bands.index(rgb_band))
            else:
                logger.error(f"Required band {rgb_band} not found in the available bands")
                return

        if len(rgb_indices) != 3:
            logger.error("Could not find all required RGB bands")
            return
        rgb_indices = rgb_indices[::-1]
        # Extract RGB bands
        rgb_x = x_np[:, :, rgb_indices].copy()  # Make a copy to avoid modifying the original data
        rgb_pred = pred_np[:, :, rgb_indices].copy()
        gt_np = gt_np[:, :, rgb_indices].copy()
        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))

        # Plot input TCI
        axs[0].imshow(rgb_x)
        axs[0].set_title(f"Input L1C - True Color Index {bands}", fontsize=16)
        axs[0].axis('off')


        # Plot gt  TCI
        axs[1].imshow(rgb_pred)
        axs[1].set_title(f"Reference L2A Sen2Cor- True Color Index {bands}", fontsize=16)
        axs[1].axis('off')

        # Plot predicted TCI
        axs[2].imshow(rgb_pred)
        axs[2].set_title(f"Predicted L2A - True Color Index {bands}", fontsize=16)
        axs[2].axis('off')

        # Save figure
        fig.tight_layout()
        fig.savefig(f"./TCI.svg", dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.success("TCI RGB composite visualization generated")
    except Exception as e:
        logger.error(f"Failed to generate TCI RGB composite: {e}")


def extract_metrics_to_json(x_np: np.ndarray, gt_np: np.ndarray, pred_np: np.ndarray, 
                           bands: List[str], output_dir: str) -> None:
    """
    Extract all metrics from the prediction results and save them to a JSON file.
    
    Args:
        x_np: Input data (L1C) array with shape [H, W, C]
        gt_np: Ground truth data (L2A) with shape [H, W, C]
        pred_np: Predicted data (L2A) array with shape [H, W, C]
        bands: List of band names
        output_dir: Directory to save the output JSON file
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics dictionary
        metrics = {
            "overall_metrics": {},
            "band_metrics": {},
            "summary_statistics": {}
        }
        
        # Calculate metrics for each band
        band_metrics = {}
        all_rmse_values = []
        all_mae_values = []
        all_relative_errors = []
        
        for idx, band in enumerate(bands):
            # Get the data for this band
            band_x = x_np[:, :, idx]
            band_gt = gt_np[:, :, idx]
            band_pred = pred_np[:, :, idx]
            
            # Calculate basic metrics
            rmse = np.sqrt(np.mean((band_gt - band_pred)**2))
            mae = np.mean(np.abs(band_gt - band_pred))
            
            # Calculate relative error
            mask = band_gt > 0
            relative_error = np.abs(band_gt - band_pred) / np.maximum(band_gt, 1e-8) * 100
            mean_relative_error = np.mean(relative_error[mask]) if np.any(mask) else 0
            
            # mse
            mse = np.mean((band_gt - band_pred)**2)
            
            # Correlation coefficient
            correlation = np.corrcoef(band_gt.flatten(), band_pred.flatten())[0, 1]
            
            # R-squared
            ss_res = np.sum((band_gt - band_pred)**2)
            ss_tot = np.sum((band_gt - np.mean(band_gt))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Bias (mean error)
            bias = np.mean(band_pred - band_gt)
            
            # Standard deviation of errors
            error_std = np.std(band_pred - band_gt)
            
            # Min and max values for input, ground truth, and prediction
            band_stats = {
                "input_stats": {
                    "min": float(np.min(band_x)),
                    "max": float(np.max(band_x)),
                    "mean": float(np.mean(band_x)),
                    "std": float(np.std(band_x))
                },
                "ground_truth_stats": {
                    "min": float(np.min(band_gt)),
                    "max": float(np.max(band_gt)),
                    "mean": float(np.mean(band_gt)),
                    "std": float(np.std(band_gt))
                },
                "prediction_stats": {
                    "min": float(np.min(band_pred)),
                    "max": float(np.max(band_pred)),
                    "mean": float(np.mean(band_pred)),
                    "std": float(np.std(band_pred))
                }
            }
            
            # Store band metrics
            band_metrics[band] = {
                "rmse": float(rmse),
                "mae": float(mae),
                "mse": float(mse),
                "mean_relative_error_percent": float(mean_relative_error),
                "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
                "r_squared": float(r_squared),
                "bias": float(bias),
                "error_std": float(error_std),
                **band_stats
            }
            
            # Collect for overall statistics
            all_rmse_values.append(rmse)
            all_mae_values.append(mae)
            all_relative_errors.append(mean_relative_error)
        
        # Calculate overall metrics across all bands
        overall_rmse = np.sqrt(np.mean((gt_np - pred_np)**2))
        overall_mae = np.mean(np.abs(gt_np - pred_np))
        overall_mse = np.mean((gt_np - pred_np)**2)
        
        # Overall correlation
        overall_correlation = np.corrcoef(gt_np.flatten(), pred_np.flatten())[0, 1]
        
        # Overall R-squared
        overall_ss_res = np.sum((gt_np - pred_np)**2)
        overall_ss_tot = np.sum((gt_np - np.mean(gt_np))**2)
        overall_r_squared = 1 - (overall_ss_res / overall_ss_tot) if overall_ss_tot > 0 else 0
        
        # Overall bias
        overall_bias = np.mean(pred_np - gt_np)
        
        # Store overall metrics
        metrics["overall_metrics"] = {
            "overall_rmse": float(overall_rmse),
            "overall_mae": float(overall_mae),
            "overall_mse": float(overall_mse),
            "overall_correlation": float(overall_correlation) if not np.isnan(overall_correlation) else 0.0,
            "overall_r_squared": float(overall_r_squared),
            "overall_bias": float(overall_bias),
            "mean_rmse_across_bands": float(np.mean(all_rmse_values)),
            "mean_mae_across_bands": float(np.mean(all_mae_values)),
            "mean_relative_error_across_bands": float(np.mean(all_relative_errors))
        }
        
        # Store band metrics
        metrics["band_metrics"] = band_metrics
        
        # Summary statistics
        metrics["summary_statistics"] = {
            "total_bands": len(bands),
            "bands_list": bands,
            "data_shape": {
                "height": int(gt_np.shape[0]),
                "width": int(gt_np.shape[1]),
                "channels": int(gt_np.shape[2])
            },
            "best_band_by_rmse": min(band_metrics.keys(), key=lambda x: band_metrics[x]["rmse"]),
            "worst_band_by_rmse": max(band_metrics.keys(), key=lambda x: band_metrics[x]["rmse"]),
            "best_band_by_mae": min(band_metrics.keys(), key=lambda x: band_metrics[x]["mae"]),
            "worst_band_by_mae": max(band_metrics.keys(), key=lambda x: band_metrics[x]["mae"]),
            "rmse_statistics": {
                "min": float(np.min(all_rmse_values)),
                "max": float(np.max(all_rmse_values)),
                "mean": float(np.mean(all_rmse_values)),
                "std": float(np.std(all_rmse_values))
            },
            "mae_statistics": {
                "min": float(np.min(all_mae_values)),
                "max": float(np.max(all_mae_values)),
                "mean": float(np.mean(all_mae_values)),
                "std": float(np.std(all_mae_values))
            }
        }
        
        # Save metrics to JSON file
        json_path = os.path.join(output_dir, "metrics.json")
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metrics saved to: {json_path}")
        
        return metrics
        
    except Exception as e:
        print(f"Failed to extract metrics: {e}")
        import traceback
        print(traceback.format_exc())
        return None
