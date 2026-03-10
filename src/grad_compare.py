# import os
# import pandas as pd
# import glob
# import re
# import numpy as np

# def parse_folder_metadata(folder_name):
#     """Extracts BS, Nodes, Loss Rate, and Timestamp from folder name."""
#     # Pattern to match: ...loss-rate0_20260310-004407_gradcmp_bs8_nodes1
#     pattern = r"loss-rate(?P<loss>[\d\.]+)_.*_bs(?P<bs>\d+)_nodes(?P<nodes>\d+)"
#     match = re.search(pattern, folder_name)
#     if match:
#         return match.groupdict()
#     return None

# def process_layer_files(root_dir):
#     all_rows = []
    
#     # Define Cluster Edge pairs for 16 GPUs (4 nodes)
#     edge_pairs = [(0, 15), (1, 14), (2, 13), (3, 12), (15, 0), (14, 1), (13, 2), (12, 3)]

#     # Walk through the sanity_check_logs directory
#     for folder in os.listdir(root_dir):
#         folder_path = os.path.join(root_dir, folder)
#         if not os.path.isdir(folder_path):
#             continue
            
#         meta = parse_folder_metadata(folder)
#         if not meta:
#             continue

#         print(f"Processing Run: BS={meta['bs']}, Loss={meta['loss']}, Nodes={meta['nodes']}")

#         # Get all layer.csv files in this folder
#         layer_files = glob.glob(os.path.join(folder_path, "layer*.csv"))
        
#         for file_path in layer_files:
#             layer_id = re.search(r"layer(\d+)\.csv", file_path).group(1)
#             df = pd.read_csv(file_path)
            
#             # Clean data: drop NaNs in critical columns
#             df = df.dropna(subset=['cosine_similarity', 'rel_magnitude_diff'])

#             # Helper to calculate averages for specific filters
#             def get_metrics(subset):
#                 if subset.empty:
#                     return [None, None, None]
#                 return [
#                     subset['cosine_similarity'].mean(),
#                     subset['rel_magnitude_diff'].mean(),
#                     subset['correlation'].mean()
#                 ]

#             # 1. Intra-Node
#             intra = df[df['comparison_type'] == 'intra_node']
            
#             # 2. Inter-Node
#             inter = df[df['comparison_type'] == 'inter_node']
            
#             # 3. Cluster Edge (Furthest Ranks)
#             edge = df[df.apply(lambda row: (row['rank1'], row['rank2']) in edge_pairs, axis=1)]
            
#             # 4. Global Diversity (All comparisons)
#             global_avg = df

#             # Define the comparisons to save
#             comparisons = [
#                 ("Intra-Node", "All Intra-Server", "Same Server", intra),
#                 ("Inter-Node", "Position-wise", "Diff Server", inter),
#                 ("Cluster Edge", "Max Distance Ranks", "Max Distance", edge),
#                 ("Global Diversity", "All Ranks", "Fleet Avg", global_avg)
#             ]

#             for comp_name, gpu_comp, dist, data in comparisons:
#                 cos, mag, corr = get_metrics(data)
#                 if cos is not None:
#                     all_rows.append({
#                         "Nodes": meta['nodes'],
#                         "Loss Rate": meta['loss'],
#                         "BS": meta['bs'],
#                         "Layer": int(layer_id),
#                         "Comparison Type": comp_name,
#                         "GPU Rank Comparison": gpu_comp,
#                         "Logical Distance": dist,
#                         "Cosine": round(cos, 4),
#                         "Mag Diff": round(mag, 4),
#                         "Corr": round(corr, 4),
#                         "Folder": folder # For tracking
#                     })

#     return pd.DataFrame(all_rows)

# # --- EXECUTION ---
# # Change 'sanity_check_logs' to your actual path if different
# root_directory = 'sanity_check_logs' 
# master_df = process_layer_files(root_directory)

# # Sort by Nodes, BS, and Layer for a clean table
# master_df = master_df.sort_values(by=['Nodes', 'BS', 'Layer', 'Comparison Type'])

# # Save to CSV
# master_df.to_csv("1_node_grad_cmp_0_loss_BS_32.csv", index=False)
# print("Processing complete! File saved as '1_node_grad_cmp_0_loss_BS_32.csv'")




import os
import pandas as pd
import glob
import re

# --- CONFIGURATION ---
ROOT_DIR = 'sanity_check_logs'

# LIST YOUR FOLDERS HERE (Copy-paste the folder names from your directory)
FOLDERS_TO_PROCESS = [
    "4gpus_piqa_seed10_loss-rate_high_persistence_low_intensity_1_20260310-002049_gradcmp_bs8_nodes4",
    # "another_folder_name_here",
]
# ---------------------

def parse_metadata(folder_name):
    """Extracts metadata using simple keyword searches to handle various naming styles."""
    meta = {}
    # Extract Loss Rate (looks for 'loss-rate' followed by digits/dots)
    loss_match = re.search(r"loss-rate([\d\.]+)", folder_name)
    meta['Loss Rate'] = loss_match.group(1) if loss_match else "0"
    
    # Extract Batch Size (looks for 'bs' followed by digits)
    bs_match = re.search(r"bs(\d+)", folder_name)
    meta['BS'] = bs_match.group(1) if bs_match else "Unknown"
    
    # Extract Nodes (looks for 'nodes' followed by digits)
    nodes_match = re.search(r"nodes(\d+)", folder_name)
    meta['Nodes'] = nodes_match.group(1) if nodes_match else "Unknown"
    
    return meta

def process_single_folder(folder_name):
    folder_path = os.path.join(ROOT_DIR, folder_name)
    if not os.path.exists(folder_path):
        print(f"!! Skipping: {folder_name} (Folder not found)")
        return pd.DataFrame()

    print(f"Processing: {folder_name}...")
    meta = parse_metadata(folder_name)
    all_layer_rows = []
    
    # Define Cluster Edge pairs for 16 GPUs (4 nodes)
    edge_pairs = [(0, 15), (1, 14), (2, 13), (3, 12), (15, 0), (14, 1), (13, 2), (12, 3)]

    # Get all layer files and sort them numerically (0, 1, 2... 21)
    layer_files = glob.glob(os.path.join(folder_path, "layer*.csv"))
    layer_files.sort(key=lambda x: int(re.search(r'layer(\d+)', x).group(1)))
    
    for file_path in layer_files:
        layer_id = re.search(r"layer(\d+)\.csv", file_path).group(1)
        df = pd.read_csv(file_path)
        
        # Remove empty/error rows
        df = df.dropna(subset=['cosine_similarity'])

        # Categorize comparisons
        comparisons = [
            ("Intra-Node", df[df['comparison_type'] == 'intra_node']),
            ("Inter-Node", df[df['comparison_type'] == 'inter_node']),
            ("Cluster Edge", df[df.apply(lambda r: (r['rank1'], r['rank2']) in edge_pairs, axis=1)]),
            ("Global Diversity", df) # All ranks
        ]

        for comp_name, subset in comparisons:
            if not subset.empty:
                all_layer_rows.append({
                    "Nodes": meta['Nodes'],
                    "Loss Rate": meta['Loss Rate'],
                    "BS": meta['BS'],
                    "Layer": int(layer_id),
                    "Comparison Type": comp_name,
                    "Cosine": round(subset['cosine_similarity'].mean(), 6),
                    "Mag Diff": round(subset['rel_magnitude_diff'].mean(), 6),
                    "Corr": round(subset['correlation'].mean(), 6)
                })

    return pd.DataFrame(all_layer_rows)

# --- EXECUTION ---
final_dfs = []
for folder in FOLDERS_TO_PROCESS:
    folder_df = process_single_folder(folder)
    final_dfs.append(folder_df)

if final_dfs:
    master_df = pd.concat(final_dfs, ignore_index=True)
    # Save a specific file for these folders
    output_name = "4_nodes_grad_cmp_1%_det_loss_BS_8.csv"
    master_df.to_csv(output_name, index=False)
    print(f"\nSuccess! Results saved to: {output_name}")
    print(master_df.head(10))
else:
    print("No data processed.")