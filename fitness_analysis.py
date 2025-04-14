import sys
import subprocess
import json
import os
import re
import plotly.express as px
import pandas as pd
from datetime import datetime

def run_experiment(base_config, image_path, fitness_mode, run_label, timestamp, results_subfolder):
    """
    1) Takes the base_config dict, plus a single image path and a target fitness_mode.
    2) Overrides config['fitness_mode'] with the given mode.
    3) Sets config['results_folder'] = results_subfolder so main.py saves images there.
    4) Writes a temporary config file in that subfolder.
    5) Calls main.py with that config and the specified image.
    6) Parses console output for lines about avg fitness.
    7) Returns a list of (generation, avg_fitness).
    """
    # Make a copy of the config so we don't mutate the original
    config = dict(base_config)

    # Override the 'fitness_mode' (e.g. "default" or "interpolated")
    config["fitness_mode"] = fitness_mode

    # Override the results folder so main.py saves images in image-specific subfolder
    config["results_folder"] = results_subfolder

    # Also adjust the output image name to avoid collisions
    # Example: composite_guyana_default_20250414_190210.png
    config["output_image_name"] = f"composite_{run_label}_{timestamp}.png"

    # Create a temp config file inside the subfolder
    temp_config_filename = f"temp_config_{run_label}_{timestamp}.json"
    temp_config_path = os.path.join(results_subfolder, temp_config_filename)
    with open(temp_config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Run main.py
    cmd = ["python", "main.py", temp_config_path, image_path]
    print(f"==> Running: {cmd}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    generations_data = []
    for line in process.stdout:
        print(line, end="")  # Echo to console

        # Look for lines like:
        # [Main] Saved generation 50 to ... avg fitness = 0.123456
        match = re.search(r"Saved generation\s+(\d+).*avg fitness = ([0-9.]+)", line)
        if match:
            gen = int(match.group(1))
            avg_fitness = float(match.group(2))
            generations_data.append((gen, avg_fitness))

    process.wait()

    return generations_data


def main():
    # 1) Parse command line arguments: which config to load?
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <analysis_config.json>")
        sys.exit(1)
    analysis_config_path = sys.argv[1]

    # 2) Load the base config (GA params + list of images, etc.)
    with open(analysis_config_path, "r") as f:
        base_config = json.load(f)

    # The images to process
    image_paths = base_config["images"]

    # Fitness modes to compare
    fitness_modes = [
        ("default", "default"),
        ("interpolated", "interp")
    ]

    # 3) Create a single top-level timestamped folder for everything
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    top_results_folder = f"analysis_results_{timestamp}"
    os.makedirs(top_results_folder, exist_ok=True)

    # 4) For each image, create a subfolder, run each mode, and gather data
    for img_path in image_paths:
        # Derive a label from filename (e.g., "guyana" from "guyana.png")
        img_basename = os.path.basename(img_path)
        image_label = os.path.splitext(img_basename)[0]

        # Create a subfolder for this image
        # e.g.: analysis_results_20250414_190210/guyana/
        results_subfolder = os.path.join(top_results_folder, image_label)
        os.makedirs(results_subfolder, exist_ok=True)

        # We'll accumulate run data for just this image
        runs_for_image = []

        # Run both fitness modes for this image
        for (mode_value, mode_label) in fitness_modes:
            run_label = f"{image_label}_{mode_label}"
            print(f"\n=== RUNNING EXPERIMENT: {run_label} ===\n")

            # Run the GA
            generations_data = run_experiment(
                base_config=base_config,
                image_path=img_path,
                fitness_mode=mode_value,
                run_label=run_label,
                timestamp=timestamp,
                results_subfolder=results_subfolder
            )

            # Collect data for plotting
            for (gen, avg_fit) in generations_data:
                runs_for_image.append({
                    "run_label": run_label,
                    "image_label": image_label,
                    "fitness_mode": mode_value,
                    "generation": gen,
                    "avg_fitness": avg_fit
                })

        # 5) Plot the results for this image
        if len(runs_for_image) == 0:
            # Possibly no successful runs (e.g., missing image file)
            continue

        df_image = pd.DataFrame(runs_for_image)

        # (a) A line chart comparing default vs. interpolated for *this* image
        fig_line = px.line(
            df_image,
            x="generation",
            y="avg_fitness",
            color="run_label",
            markers=True,
            title=f"Average Fitness by Generation - {image_label}"
        )
        line_chart_path = os.path.join(results_subfolder, "analysis_fitness_line.html")
        fig_line.write_html(line_chart_path)
        print(f"[Analysis] Wrote line chart to {line_chart_path}")

        # (b) A bar chart of final fitness for each run (default vs. interpolation)
        final_df = df_image.sort_values("generation").groupby("run_label").tail(1)
        fig_bar = px.bar(
            final_df,
            x="run_label",
            y="avg_fitness",
            title=f"Final Average Fitness - {image_label}",
            text="avg_fitness"
        )
        fig_bar.update_traces(textposition="outside")
        bar_chart_path = os.path.join(results_subfolder, "analysis_fitness_final.html")
        fig_bar.write_html(bar_chart_path)
        print(f"[Analysis] Wrote final fitness bar chart to {bar_chart_path}")

    print("\n=== Analysis complete! ===")
    print(f"All results saved under: {top_results_folder}")


if __name__ == "__main__":
    main()