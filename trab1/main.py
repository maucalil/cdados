
# Author: Maurício Calil Manfrim Lucera - 11813306

def main():
    num_bins = int(input("Enter the number of bins: "))
    data = get_data()

    if data is None:
        return
    
    bins, bin_width = calculate_bins(data, num_bins)
    frequencies = calculate_histogram(data, bins, num_bins)
    total_samples = len(data)
    densities = calculate_density(frequencies, total_samples)

    display_frequency(bins, frequencies, bin_width, num_bins)
    display_density(bins, densities, bin_width, num_bins)

def get_data():
    data_source = input("Enter 'manual' to enter data manually or 'file' to provide a file path: ")
    data = None
    
    if data_source == 'manual':
        data = [float(val) for val in input("Enter the data values separated by spaces: ").split()]
    elif data_source == 'file':
        file_path = input("Enter the path of the data file: ")
        with open(file_path, 'r') as file:
            data = [float(val) for val in file.read().split()]
    else:
        print("Invalid option. Please choose 'manual' or 'file'.")
    
    return data

def calculate_bins(data, num_bins):
    min_val = min(data)
    max_val = max(data)
    bin_width = (max_val - min_val) / num_bins
    
    bins = [min_val + i * bin_width for i in range(num_bins + 1)]

    return bins, bin_width

def calculate_histogram(data, bins, num_bins):
    frequencies = [0] * (num_bins)
    
    for val in data:
        for i in range(num_bins):
            if bins[i] <= val < bins[i + 1] or (i == num_bins - 1 and val == bins[i + 1]):
                frequencies[i] += 1
                break
    
    return frequencies

def calculate_density(frequencies, total_samples):
    densities = [freq / total_samples for freq in frequencies]
    return densities

def display_frequency(bins, frequencies, bin_width, num_bins):
    print("\nFrequency:")
    for i in range(num_bins):
        print(f"{frequencies[i]} | {bins[i]:.6f} - {bins[i] + bin_width:.6f}")

def display_density(bins, densities, bin_width, num_bins):
    print("\nDensity:")
    for i in range(num_bins):
        print(f"{densities[i]:.3f} | {bins[i]:.6f} - {bins[i] + bin_width:.6f}")

if __name__ == "__main__":
    main()
