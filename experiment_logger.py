import csv
from pathlib import Path

def log_experiment(result_dict, file_path="experiments_results_3_Best.csv"):
    file_path = Path(file_path)
    file_exists = file_path.exists()

    with open(file_path, "a", newline='', encoding='utf-8') as csvfile:
        fieldnames = list(result_dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        
        writer.writerow(result_dict)
