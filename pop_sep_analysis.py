import sys
import subprocess
import json
import os
import re
import plotly.express as px
import pandas as pd
from datetime import datetime
from utils.image_utils import create_montage

def run_experiment(base_config, image_path, sep_method, sel_method, run_label, timestamp, results_subfolder):
    """
    1) Copies base_config and overrides:
       - separation_method = sep_method
       - selection_method  = sel_method
       - results_folder    = results_subfolder
       - output_image_name = composite_<run_label>_<timestamp>.png

    2) Runs main.py in a subprocess, parsing lines to gather:
       - generation & avg fitness
       - generation & variance
       - snapshot image paths.

    3) Returns:
       - A list of tuples: (generation, avg_fitness, variance)
       - A list of snapshot paths in order of generation.
    """

    # Create a config specific to this run
    config = dict(base_config)
    config["separation_method"] = sep_method
    config["selection_method"] = sel_method
    config["results_folder"] = results_subfolder
    config["output_image_name"] = f"composite_{run_label}_{timestamp}.png"

    # Write this config to a temp file in the subfolder
    temp_config_filename = f"temp_config_{run_label}_{timestamp}.json"
    temp_config_path = os.path.join(results_subfolder, temp_config_filename)
    with open(temp_config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Subprocess command
    cmd = ["python", "main.py", temp_config_path, image_path]
    print(f"==> Running: {cmd}", flush=True)

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    gens_data = {}  # generation -> {"avg_fitness": float, "variance": float}
    snapshot_paths = []

    # Regex for average fitness
    # e.g. "[Main] Saved generation 50 to ... gen50.png, avg fitness = 0.123456"
    pat_avg = re.compile(r"Saved generation\s+(\d+)\s+to\s+(\S+),\s+avg fitness\s*=\s*([0-9.]+)")

    # Regex for variance
    # e.g. "[Main] Fitness variance across patches for generation 50: 0.000123"
    pat_var = re.compile(r"Fitness variance.*generation\s+(\d+):\s*([0-9.]+)")

    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(line, end="", flush=True)

        # Check for average fitness line
        match_avg = pat_avg.search(line)
        if match_avg:
            gen = int(match_avg.group(1))
            snapshot_path = match_avg.group(2)
            avg_fit = float(match_avg.group(3))

            # Store the avg_fitness
            if gen not in gens_data:
                gens_data[gen] = {}
            gens_data[gen]["avg_fitness"] = avg_fit

            # Keep track of the snapshot image path
            snapshot_paths.append(snapshot_path)

        # Check for variance line
        match_var = pat_var.search(line)
        if match_var:
            gen2 = int(match_var.group(1))
            var_val = float(match_var.group(2))
            if gen2 not in gens_data:
                gens_data[gen2] = {}
            gens_data[gen2]["variance"] = var_val

    process.wait()

    # Convert gens_data into a sorted list of (gen, avg_f, variance)
    all_tuples = []
    for gen, record in sorted(gens_data.items()):
        avg_f = record.get("avg_fitness", None)
        var_ = record.get("variance", None)
        all_tuples.append((gen, avg_f, var_))

    return all_tuples, snapshot_paths

def main():
    if len(sys.argv) < 2:
        print("Usage: python pop_sep_analysis.py <base_config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        base_config = json.load(f)

    # The images you want to test (e.g., flags + photos)
    image_paths = base_config["images"]

    # We'll compare two separation methods + two selection methods
    sep_methods = ["traditional", "young_bias"]
    sel_methods = ["elite", "roulette"]

    # Create a top-level results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    top_folder = os.path.join("results", f"pop_sep_analysis_{timestamp}")
    os.makedirs(top_folder, exist_ok=True)

    # We'll store run data for each image in a single DataFrame
    all_records = []

    for img_path in image_paths:
        image_basename = os.path.basename(img_path)
        image_label = os.path.splitext(image_basename)[0]

        # Subfolder for this image
        image_subfolder = os.path.join(top_folder, image_label)
        os.makedirs(image_subfolder, exist_ok=True)

        # We also track snapshots for creating montages
        # run_label -> list of snapshot paths
        run_snapshots_map = []

        # We'll accumulate data for each run (line/variance/final bar charts)
        runs_for_image = []

        # Loop over each combo of (sep_method, sel_method)
        for sep_m in sep_methods:
            for sel_m in sel_methods:
                # e.g. run_label = "guyana_traditional_elite"
                run_label = f"{image_label}_{sep_m}_{sel_m}"
                print(f"\n=== RUNNING: {run_label} ===\n", flush=True)

                data_tuples, snapshot_paths = run_experiment(
                    base_config=base_config,
                    image_path=img_path,
                    sep_method=sep_m,
                    sel_method=sel_m,
                    run_label=run_label,
                    timestamp=timestamp,
                    results_subfolder=image_subfolder
                )

                # data_tuples is a list of (gen, avg_fitness, variance)
                # build up "runs_for_image" for Plotly
                for (gen, avg_f, var_) in data_tuples:
                    runs_for_image.append({
                        "run_label": run_label,
                        "image_label": image_label,
                        "separation_method": sep_m,
                        "selection_method": sel_m,
                        "generation": gen,
                        "avg_fitness": avg_f,
                        "variance": var_
                    })

                # Store for montage creation
                # We'll include the original image first, then the snapshot paths
                montage_inputs = [img_path] + snapshot_paths
                run_snapshots_map.append((run_label, montage_inputs))

        # If we have no data for this image, skip
        if not runs_for_image:
            continue

        # Convert to a DataFrame
        df_image = pd.DataFrame(runs_for_image)

        # 1) Fitness vs Generation (line chart)
        fig_fit = px.line(
            df_image,
            x="generation",
            y="avg_fitness",
            color="run_label",
            title=f"Pop. Separation - Avg Fitness by Gen - {image_label}",
            markers=True
        )
        fitness_line_path = os.path.join(image_subfolder, f"analysis_fitness_line.html")
        fig_fit.write_html(fitness_line_path)
        print(f"[Analysis] Wrote fitness line chart to {fitness_line_path}")

        # 2) Variance vs Generation (line chart)
        df_image["variance"] = df_image["variance"].fillna(0.0)
        fig_var = px.line(
            df_image,
            x="generation",
            y="variance",
            color="run_label",
            title=f"Pop. Separation - Variance by Gen - {image_label}",
            markers=True
        )
        variance_line_path = os.path.join(image_subfolder, f"analysis_variance_line.html")
        fig_var.write_html(variance_line_path)
        print(f"[Analysis] Wrote variance line chart to {variance_line_path}")

        # 3) Final fitness bar chart
        # group by run_label, take the largest generation
        final_gen_df = df_image.groupby("run_label", as_index=False)["generation"].max()
        merged = pd.merge(df_image, final_gen_df, on=["run_label", "generation"], how="inner")

        fig_final = px.bar(
            merged,
            x="run_label",
            y="avg_fitness",
            color="separation_method",  # or "selection_method"
            title=f"Pop. Separation - Final Fitness - {image_label}",
            text="avg_fitness"
        )
        fig_final.update_traces(textposition="outside")
        final_bar_path = os.path.join(image_subfolder, f"analysis_fitness_final.html")
        fig_final.write_html(final_bar_path)
        print(f"[Analysis] Wrote final fitness bar chart to {final_bar_path}")

        # 4) Montages for each run
        # We do the same approach as in your prior script
        for (run_label, montage_inputs) in run_snapshots_map:
            from utils.image_utils import create_montage
            montage_filename = f"montage_{run_label}.png"
            montage_path = os.path.join(image_subfolder, montage_filename)
            # The "run_label" is used as the big title across the top
            create_montage(
                run_label=run_label,
                image_paths=montage_inputs,
                output_path=montage_path,
                big_title_height=50,
                tile_label_height=30
            )

    print("\n=== Population Separation Analysis Complete! ===")
    print(f"All results saved under: {top_folder}")

if __name__ == "__main__":
    main()