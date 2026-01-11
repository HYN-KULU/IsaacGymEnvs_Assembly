import json
with open("eval_vis/results.json","r") as f: log_result = json.load(f)
import numpy as np

summary = {}

for task_id, records in log_result.items():
    success_rates = []
    mean_errors = []
    median_errors = []

    for r in records:
        if r.get("num_valid_frames", 0) == 0:
            continue

        if r.get("success_rate") is not None:
            success_rates.append(r["success_rate"])

        if r.get("mean_angle_error_deg") is not None:
            mean_errors.append(r["mean_angle_error_deg"])

        if r.get("median_angle_error_deg") is not None:
            median_errors.append(r["median_angle_error_deg"])

    summary[task_id] = {
        "avg_success_rate": np.mean(success_rates) if success_rates else None,
        "avg_mean_angle_error_deg": np.mean(mean_errors) if mean_errors else None,
        "avg_median_angle_error_deg": np.mean(median_errors) if median_errors else None,
        "num_valid_runs": len(success_rates),
    }

# -------- SAFE PRINT --------
for task_id in sorted(summary.keys()):
    s = summary[task_id]

    if s["avg_median_angle_error_deg"] is None:
        median_str = "N/A"
    else:
        median_str = "%.2f°" % s["avg_median_angle_error_deg"]

    if s["avg_mean_angle_error_deg"] is None:
        mean_str = "N/A"
    else:
        mean_str = "%.2f°" % s["avg_mean_angle_error_deg"]

    if s["avg_success_rate"] is None:
        success_str = "N/A"
    else:
        success_str = "%.3f" % s["avg_success_rate"]

    print(
        "Task %s | Success: %s | Mean Err: %s | Median Err: %s"
        % (
            task_id,
            success_str,
            mean_str,
            median_str,
        )
    )
with open("flow_eval_results.md", "w") as f:
    # Header
    f.write("| Task ID | Success | Mean Error (°) | Median Error (°) |\n")
    f.write("|--------|---------|----------------|------------------|\n")

    # Rows (sorted for readability)
    for task_id in sorted(summary.keys()):
        s = summary[task_id]

        success = (
            "N/A" if s["avg_success_rate"] is None
            else f"{s['avg_success_rate']:.3f}"
        )
        mean_err = (
            "N/A" if s["avg_mean_angle_error_deg"] is None
            else f"{s['avg_mean_angle_error_deg']:.2f}"
        )
        median_err = (
            "N/A" if s["avg_median_angle_error_deg"] is None
            else f"{s['avg_median_angle_error_deg']:.2f}"
        )

        f.write(
            f"| {task_id} | {success} | {mean_err} | {median_err} |\n"
        )
# Collect valid per-task values
overall_success = []
overall_mean_err = []
overall_median_err = []

for s in summary.values():
    if s["avg_success_rate"] is not None:
        overall_success.append(s["avg_success_rate"])

    if s["avg_mean_angle_error_deg"] is not None:
        overall_mean_err.append(s["avg_mean_angle_error_deg"])

    if s["avg_median_angle_error_deg"] is not None:
        overall_median_err.append(s["avg_median_angle_error_deg"])

# Compute overall averages
overall_avg_success = np.mean(overall_success)
overall_avg_mean_err = np.mean(overall_mean_err)
overall_avg_median_err = np.mean(overall_median_err)

print("=== Overall (Task-wise Average) ===")
print(f"Overall Success Rate: {overall_avg_success:.3f}")
print(f"Overall Mean Error:   {overall_avg_mean_err:.2f}°")
print(f"Overall Median Error: {overall_avg_median_err:.2f}°")
