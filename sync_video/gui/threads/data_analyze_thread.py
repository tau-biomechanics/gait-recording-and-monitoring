"""
Thread for analyzing datasets for ML
"""

try:
    from PyQt5.QtCore import QThread, pyqtSignal
except ImportError as e:
    print(f"ERROR importing PyQt5: {str(e)}")
    print("This may cause linter errors, but the app should still run")

import numpy as np
import sys
import time
import traceback
from ..utils import debug_log


class DataAnalyzeThread(QThread):
    """Thread for analyzing datasets for ML"""

    # Progress signals
    progress_signal = pyqtSignal(int, str, str)  # (progress_value, message, detail)
    finished_signal = pyqtSignal(object)  # Results dict
    error_signal = pyqtSignal(str)  # Error message

    def __init__(self, dataset):
        debug_log("Creating DataAnalyzeThread")
        super(DataAnalyzeThread, self).__init__()
        self.dataset = dataset
        debug_log(f"DataAnalyzeThread created for dataset shape {dataset.shape}")

    def run(self):
        """Run the analysis process"""
        debug_log("Starting dataset analysis thread")
        try:
            print("**** ANALYSIS THREAD STARTED ****")
            print(f"Dataset shape: {self.dataset.shape}")
            sys.stdout.flush()

            results = {}

            # Check if the dataset is too large
            debug_log(f"Dataset shape: {self.dataset.shape}")
            if self.dataset.shape[0] > 10000 or self.dataset.shape[1] > 100:
                debug_log("WARNING: Large dataset detected - this may take longer")

            # Basic dataset info - minimal operation
            debug_log("Getting dataset size")
            results["dataset_size"] = len(self.dataset)
            self.progress_signal.emit(
                10,
                "Starting analysis...",
                f"Dataset size: {results['dataset_size']} rows",
            )

            # Just capture column names first - very fast operation
            debug_log("Getting column names")
            column_names = list(self.dataset.columns)
            debug_log(f"Found {len(column_names)} columns")
            results["dataset_cols"] = column_names

            # Use first column as time
            debug_log(f"First column (time): {column_names[0]}")
            results["time_col"] = column_names[0]

            # Fast string operations to categorize columns
            debug_log("Categorizing columns")
            self.progress_signal.emit(20, "Categorizing columns...", "")

            # Use simple list comprehensions for speed
            debug_log("Finding insole columns")
            results["insole_cols"] = [
                col for col in column_names if "insole" in col.lower()
            ]
            debug_log(f"Found {len(results['insole_cols'])} insole columns")

            debug_log("Finding OpenCap columns")
            results["opencap_cols"] = [
                col for col in column_names if "opencap" in col.lower()
            ]
            debug_log(f"Found {len(results['opencap_cols'])} OpenCap columns")

            debug_log("Finding QTM force columns")
            results["qtm_force_cols"] = [
                col
                for col in column_names
                if "force" in col.lower() and "qtm" in col.lower()
            ]
            debug_log(f"Found {len(results['qtm_force_cols'])} QTM force columns")

            debug_log("Finding QTM kinematic columns")
            results["qtm_kine_cols"] = [
                col
                for col in column_names
                if "qtm" in col.lower()
                and "moment" not in col.lower()
                and "force" not in col.lower()
            ]
            debug_log(f"Found {len(results['qtm_kine_cols'])} QTM kinematic columns")

            debug_log("Finding moment columns")
            results["moment_cols"] = [
                col for col in column_names if "moment" in col.lower()
            ]
            debug_log(f"Found {len(results['moment_cols'])} moment columns")

            # Update progress
            self.progress_signal.emit(
                40,
                "Finding column types...",
                f"Insole: {len(results['insole_cols'])}, "
                f"OpenCap: {len(results['opencap_cols'])}, "
                f"QTM Force: {len(results['qtm_force_cols'])}, "
                f"QTM Kine: {len(results['qtm_kine_cols'])}, "
                f"Moments: {len(results['moment_cols'])}",
            )
            print("Analysis in progress... 40% complete")
            sys.stdout.flush()

            # Simple validation - no data operations
            debug_log("Validating columns")
            self.progress_signal.emit(60, "Validating columns...", "")
            if not results["moment_cols"]:
                debug_log("ERROR: No moment columns found")
                self.error_signal.emit(
                    "No joint moment columns found in dataset. Cannot train a model."
                )
                return

            if not (
                results["insole_cols"]
                or results["opencap_cols"]
                or results["qtm_force_cols"]
                or results["qtm_kine_cols"]
            ):
                debug_log("ERROR: No valid input features found")
                self.error_signal.emit("No valid input features found in dataset.")
                return

            print("Analysis in progress... 60% complete")
            sys.stdout.flush()

            # Now check for NaN values - this can be an expensive operation
            debug_log("Checking for NaN values - THIS MAY TAKE A WHILE")
            self.progress_signal.emit(
                70,
                "Checking for NaN values...",
                "This may take a while for large datasets",
            )

            # Use a more efficient way to count NaNs
            nan_count = 0
            try:
                # Try optimized approach first
                debug_log("Using fast NaN counting method")
                start_time = time.time()
                nan_count = self.dataset.isna().sum().sum()
                elapsed = time.time() - start_time
                debug_log(f"Counted {nan_count} NaN values in {elapsed:.2f} seconds")

            except Exception as e:
                debug_log(f"Fast NaN counting failed: {str(e)}")
                # Fallback to a slower but safer approach
                debug_log("Using slower column-by-column NaN counting")
                for col in self.dataset.columns:
                    try:
                        col_nan_count = self.dataset[col].isna().sum()
                        nan_count += col_nan_count
                        debug_log(f"Column {col}: {col_nan_count} NaNs")
                    except Exception as e:
                        debug_log(f"Error counting NaNs in column {col}: {str(e)}")

            results["nan_count"] = nan_count

            print("Analysis in progress... 80% complete")
            sys.stdout.flush()

            # Check for other data issues
            debug_log("Checking for data quality issues")
            self.progress_signal.emit(
                85, "Checking data quality...", f"Found {nan_count} NaN values"
            )

            # Additional debug checks
            try:
                debug_log("Checking for inf values")
                inf_test = np.isinf(self.dataset.select_dtypes(include=[np.number]))
                inf_count = inf_test.sum().sum()
                debug_log(f"Found {inf_count} infinity values")

                # Check column ranges
                debug_log("Checking column ranges")
                numeric_cols = self.dataset.select_dtypes(include=[np.number])
                if not numeric_cols.empty:
                    col_max = numeric_cols.max()
                    col_min = numeric_cols.min()
                    for col in col_max.index:
                        max_val = col_max[col]
                        min_val = col_min[col]
                        if np.abs(max_val) > 1e6 or np.abs(min_val) > 1e6:
                            debug_log(
                                f"WARNING: Column {col} has extreme values: min={min_val}, max={max_val}"
                            )
            except Exception as e:
                debug_log(f"Error during data quality check: {str(e)}")

            print("Analysis in progress... 90% complete")
            sys.stdout.flush()

            debug_log("Analysis complete")
            self.progress_signal.emit(100, "Analysis complete", "")

            # Emit results
            print("**** ANALYSIS COMPLETE - EMITTING RESULTS ****")
            sys.stdout.flush()
            self.finished_signal.emit(results)
            print("**** RESULTS EMITTED ****")
            sys.stdout.flush()

        except Exception as e:
            debug_log(f"ERROR in analysis thread: {str(e)}")
            traceback.print_exc()
            sys.stdout.flush()
            print(f"**** ANALYSIS THREAD FAILED: {str(e)} ****")
            sys.stdout.flush()
            self.error_signal.emit(f"Error analyzing dataset: {str(e)}")

            # Still return partial results if we have them
            if 'results' in locals() and results and "dataset_size" in results:
                debug_log("Returning partial results")
                self.finished_signal.emit(results)