import sys
import subprocess
import json
import os
import re
import plotly.express as px
import pandas as pd
from datetime import datetime
from utils.image_utils import create_montage

def run_experiment(base_config, image_path, fitness_mode, run_label, timestamp, results_subfolder):
    config = dict(base_config)
    config["fitness_mode"] = fitness_mode
    config["results_folder"] = results_subfolder
    config["output_image_name"] = f"composite_{run_label}_{timestamp}.png"

    temp_config_filename = f"temp_config_{run_label}_{timestamp}.json"
    temp_config_path = os.path.join(results_subfolder, temp_config_filename)
    with open(temp_config_path, "w") as f:
        json.dump(config, f, indent=2)

    cmd = ["python", "main.py", temp_config_path, image_path]
    print(f"==> Running: {cmd}", flush=True)

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    generations_data = []
    snapshot_paths = []

    # We'll read lines in a loop (instead of a simple 'for line in process.stdout')
    pattern = re.compile(r"Saved generation\s+(\d+)\s+to\s+(\S+),\s+avg fitness\s*=\s*([0-9.]+)")

    while True:
        line = process.stdout.readline()
        if not line:  
            # Subprocess is done (or pipe closed)
            break

        print(line, end="", flush=True)  # Force immediate output
        match = pattern.search(line)
        if match:
            gen = int(match.group(1))
            img_path = match.group(2)
            avg_fitness = float(match.group(3))

            generations_data.append((gen, avg_fitness))
            snapshot_paths.append(img_path)

    process.wait()
    return generations_data, snapshot_paths

def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <analysis_config.json>")
        sys.exit(1)
    analysis_config_path = sys.argv[1]

    with open(analysis_config_path, "r") as f:
        base_config = json.load(f)

    image_paths = base_config["images"]
    fitness_modes = [("default", "default"), ("interpolated", "interp")]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    top_results_folder = os.path.join("results", f"fitness_analysis_results_{timestamp}")
    os.makedirs(top_results_folder, exist_ok=True)

    for img_path in image_paths:
        img_basename = os.path.basename(img_path)
        image_label = os.path.splitext(img_basename)[0]
        results_subfolder = os.path.join(top_results_folder, image_label)
        os.makedirs(results_subfolder, exist_ok=True)

        runs_for_image = []
        run_snapshots_map = {}

        for (mode_value, mode_label) in fitness_modes:
            run_label = f"{image_label}_{mode_label}"
            print(f"\n=== RUNNING EXPERIMENT: {run_label} ===\n", flush=True)

            gens_data, snapshot_paths = run_experiment(
                base_config=base_config,
                image_path=img_path,
                fitness_mode=mode_value,
                run_label=run_label,
                timestamp=timestamp,
                results_subfolder=results_subfolder
            )

            for (gen, avg_fit) in gens_data:
                runs_for_image.append({
                    "run_label": run_label,
                    "image_label": image_label,
                    "fitness_mode": mode_value,
                    "generation": gen,
                    "avg_fitness": avg_fit
                })

            run_snapshots_map[run_label] = snapshot_paths

        # If no runs worked, skip
        if not runs_for_image:
            continue

        # Create the Plotly charts
        df_image = pd.DataFrame(runs_for_image)

        fig_line = px.line(
            df_image,
            x="generation",
            y="avg_fitness",
            color="run_label",
            markers=True,
            title=f"Average Fitness by Generation - {image_label}"
        )
        fig_line_path = os.path.join(results_subfolder, "analysis_fitness_line.html")
        fig_line.write_html(fig_line_path)
        print(f"[Analysis] Wrote line chart to {fig_line_path}", flush=True)

        final_df = df_image.sort_values("generation").groupby("run_label").tail(1)
        fig_bar = px.bar(
            final_df,
            x="run_label",
            y="avg_fitness",
            title=f"Final Average Fitness - {image_label}",
            text="avg_fitness"
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar_path = os.path.join(results_subfolder, "analysis_fitness_final.html")
        fig_bar.write_html(fig_bar_path)
        print(f"[Analysis] Wrote final fitness bar chart to {fig_bar_path}", flush=True)

        # Make the montage for each run
        for run_label, snaps in run_snapshots_map.items():
            montage_inputs = [img_path] + snaps
            montage_title = run_label
            montage_filename = f"montage_{run_label}.png"
            montage_path = os.path.join(results_subfolder, montage_filename)
            create_montage(
                run_label=run_label, 
                image_paths=montage_inputs, 
                output_path=montage_path,
                big_title_height=50,       # optional
                tile_label_height=30       # optional
            )

    print("\n=== Analysis complete! ===", flush=True)
    print(f"All results saved under: {top_results_folder}", flush=True)

if __name__ == "__main__":
    main()