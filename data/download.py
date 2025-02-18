import kagglehub

# Download latest version
path = kagglehub.dataset_download("carlolepelaars/camvid", path="./data")

print("Path to dataset files:", path)