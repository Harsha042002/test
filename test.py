import sys
import os
import importlib.metadata
import subprocess

def get_package_size(package_name):
    try:
        # Get the package distribution
        dist = importlib.metadata.distribution(package_name)
        
        # Get the package location
        location = dist.locate_file('')
        
        # Calculate total size of the package
        total_size = 0
        for root, dirs, files in os.walk(location):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
        
        return total_size
    except Exception as e:
        print(f"Error calculating size for {package_name}: {e}")
        return 0

def main():
    # Ensure setuptools is installed
    try:
        import importlib.metadata
    except ImportError:
        print("Installing setuptools...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'setuptools'])
        import importlib.metadata

    total_size_bytes = 0
    packages = []

    # Get all installed packages
    for dist in importlib.metadata.distributions():
        package_name = dist.metadata['Name']
        package_size = get_package_size(package_name)
        total_size_bytes += package_size
        packages.append((package_name, package_size))

    # Sort packages by size in descending order
    packages.sort(key=lambda x: x[1], reverse=True)

    # Print top 10 largest packages
    print("Top 10 Largest Packages:")
    for name, size in packages[:10]:
        print(f"{name}: {size / (1024 * 1024):.2f} MB")

    # Print total size
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"\nTotal Size of All Packages: {total_size_mb:.2f} MB")

if __name__ == "__main__":
    main()