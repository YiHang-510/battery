import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm


def analyze_and_plot_individual(folder_path='.'):
    """
    Analyzes battery data from CSV files and creates an individual plot for each file.

    Args:
        folder_path (str): The path to the folder containing the CSV files.
    """
    # Create a directory to save the individual plots
    output_plot_dir = f'{folder_path}/相关性/individual_plots'
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)
        print(f"Created directory: {output_plot_dir}")

    # Find all relevant CSV files
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and '结果' in f]

    if not csv_files:
        print("No suitable CSV files were found in the directory.")
        return

    results_per_battery = []
    print(f"Found {len(csv_files)} CSV files. Starting processing and plotting...")

    for file in csv_files:
        try:
            battery_name = file.split('_')[0]
            df = pd.read_csv(os.path.join(folder_path, file))

            # --- 1. Data Preprocessing ---
            df['c'] = df['累计放电容量(Ah)'] / df['累计放电容量(Ah)'].max()
            df['SOH'] = df['最大容量(Ah)'] / 3.5

            c = df['c'].values
            soh = df['SOH'].values

            # --- 2. Calculations and Model Fitting for the current battery ---
            spearman_rho, _ = spearmanr(c, soh)
            kendall_tau, _ = kendalltau(c, soh)

            c_reshaped = c.reshape(-1, 1)

            # Sort values for smooth curve plotting
            sort_indices = np.argsort(c)
            c_sorted = c[sort_indices]
            soh_sorted = soh[sort_indices]
            c_sorted_reshaped = c_sorted.reshape(-1, 1)

            # Linear Regression
            linear_model = LinearRegression()
            linear_model.fit(c_reshaped, soh)
            soh_pred_linear = linear_model.predict(c_reshaped)
            rmse_linear = np.sqrt(mean_squared_error(soh, soh_pred_linear))
            r2_linear = r2_score(soh, soh_pred_linear)

            # LOESS Regression
            lowess_results = sm.nonparametric.lowess(soh, c, frac=0.3)
            rmse_loess = np.sqrt(mean_squared_error(soh, lowess_results[:, 1]))
            r2_loess = r2_score(soh, lowess_results[:, 1])

            # Isotonic Regression
            iso_reg = IsotonicRegression(out_of_bounds="clip",  increasing=False)
            soh_pred_iso = iso_reg.fit_transform(c_sorted, soh_sorted)

            # --- 3. Store results ---
            results_per_battery.append({
                'Battery': battery_name,
                'Spearman_rho': spearman_rho,
                'Kendall_tau': kendall_tau,
                'RMSE_Linear': rmse_linear,
                'R2_Linear': r2_linear,
                'RMSE_LOESS': rmse_loess,
                'R2_LOESS': r2_loess
            })

            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = 'Times New Roman'

            plt.figure(figsize=(6, 4))

            # Scatter plot
            plt.scatter(c, soh, alpha=0.8, label='SOH', color='royalblue', s=10)

            # Regression curves
            # plt.plot(c_sorted, soh_pred_iso, color='red', linewidth=2.5, label='Isotonic Regression', alpha=0.5)
            plt.plot(lowess_results[:, 0], lowess_results[:, 1], color='#ff7f00', linewidth=2.5, label='LOESS (span=0.3)', alpha=1)
            plt.plot(c_sorted, linear_model.predict(c_sorted_reshaped), color='#4daf4a', linewidth=2.5, label='Linear Regression', alpha=1)

            # Annotation with metrics for this battery
            text_str = (
                f'Spearman $\\rho_s$: {spearman_rho:.4f}\n'
                f'Kendall $\\tau$: {kendall_tau:.4f}\n\n'
                f'Linear RMSE: {rmse_linear:.4f}\n'
                f'Linear $R^2$: {r2_linear:.4f}\n\n'
                f'LOESS RMSE: {rmse_loess:.4f}\n'
                f'LOESS $R^2$: {r2_loess:.4f}'
            )
            plt.text(0.97, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

            # Labels, Title, and Legend
            plt.xlabel('Normalized Accumulated Discharge Capacity', fontsize=16)
            plt.ylabel('SOH', fontsize=16)
            plt.title(f'{battery_name.title()}', fontsize=16)
            # plt.title(f'SOH vs. Normalized Accumulated Discharge Capacity for {battery_name.title()}', fontsize=14)
            plt.legend(loc='lower left', fontsize=12)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(False)
            plt.ylim(0.65, 1)  # Adjust Y-axis to fit data
            plt.xlim(0, 1)

            # Save the plot to the dedicated directory
            plot_filename = os.path.join(output_plot_dir, f'{battery_name}_soh_plot.pdf')
            plt.savefig(plot_filename, dpi=600, bbox_inches='tight')
            plt.close()  # Close the figure to free up memory

            print(f"Processed and plotted data for: {battery_name}")

        except Exception as e:
            print(f"Could not process file {file}: {e}")

    # --- 5. Save the summary results to a CSV file ---
    if results_per_battery:
        results_df = pd.DataFrame(results_per_battery)
        output_csv_path = f'{folder_path}/相关性/battery_analysis_results.csv'
        results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\nSaved summary of metrics for all batteries to '{output_csv_path}'")

    print("\nAll tasks completed.")


if __name__ == '__main__':
    analyze_and_plot_individual(r'E:\code\Battery\test-7.3\累计放电量\相关性分析\statistic')

