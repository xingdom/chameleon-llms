import json
import glob

# Get all JSON files in the current directory
json_files = glob.glob("*.json")

# Dictionary to store summed values and count for averaging
trait_sums = {}
trait_counts = {}

# Process each JSON file
for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for trait, scores in data.items():
            if trait not in trait_sums:
                trait_sums[trait] = {key: 0 for key in scores}
                trait_counts[trait] = 0
            for key, value in scores.items():
                trait_sums[trait][key] += value
            trait_counts[trait] += 1

# Calculate the average scores
trait_averages = {
    trait: {key: round(value / trait_counts[trait], 2) for key, value in scores.items()}
    for trait, scores in trait_sums.items()
}

# Save results to a new JSON file
output_filename = "averaged_results.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(trait_averages, f, indent=4)

print(f"Averaged results saved to {output_filename}")