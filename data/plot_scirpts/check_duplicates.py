
import numpy as np
import matplotlib.pyplot as plt

from os.path import dirname, join

from data.utils import calculate_duplicates, load_packaged_data
raw_data_path = join(
    dirname(__file__), "store", "embed-bert_inst-scn-1_se-11_prev-true.npz"
)
raw_data = load_packaged_data(raw_data_path)
indice_pairs = raw_data.map_pair_indices[..., None]  
# indice_pairs = np.concat(
#     [
#         raw_data.map_pair_indices[..., None],
#         raw_data.instruction_indices[..., None],
#     ],
#     axis=1,
# )

print(indice_pairs, indice_pairs.shape)
duplicates = calculate_duplicates(indice_pairs)
unique_counts = np.sort(duplicates.unique_counts)[::-1]  
duplicates_ratio = duplicates.duplicates_ratio  

print(f"Unique value count vector: {unique_counts}")
print("Number of unique values: %d" % (unique_counts.shape[0]))
print("Percentage of unique values: %.2f%%" % (duplicates_ratio * 100))

print(
    "Number of values that appear only once: %d (%.2f%% of all pairs)"
    % (
        np.sum(unique_counts == 1),
        np.sum(unique_counts == 1) / unique_counts.shape[0] * 100,
    )
)

# Print maximum and minimum counts
max_count = np.max(unique_counts)
min_count = np.min(unique_counts)
print(f"Maximum Count of Duplicates: {max_count}")
print(f"Minimum Count of Duplicates: {min_count}")
plt.figure(figsize=(10, 6))
plt.plot(unique_counts, marker=".")
plt.title("Counts of Duplicate Samples")
plt.xlabel("Unique Pair Index")
plt.ylabel("Count of Duplicates")
plt.grid()

# Annotate maximum and minimum counts on the plot
plt.annotate(
    f"Max: {max_count}",
    xy=(np.argmax(unique_counts), max_count),
    xytext=(np.argmax(unique_counts), max_count + 1),
    arrowprops=dict(facecolor="black", shrink=0.05),
    horizontalalignment="center",
)

plt.annotate(
    f"Min: {min_count}",
    xy=(np.argmin(unique_counts), min_count),
    xytext=(np.argmin(unique_counts), min_count + 1),
    arrowprops=dict(facecolor="red", shrink=0.05),
    horizontalalignment="center",
)

plt.savefig("renewed_logics_duplicates.png")
