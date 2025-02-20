import kagglehub

# Download latest version
path = kagglehub.dataset_download("carlolepelaars/camvid")

print("Path to dataset files:", path)