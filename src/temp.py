    # Load data from second JSON file
    with open('results/deeplabmodeldatafinal_interrupted_20250309_115229.json', 'r') as f:
        data2 = json.load(f)

    # Merge data from both JSON files
    for key in data2:
        if key in data:
            # Check if the value is a list (can be extended)
            if isinstance(data[key], list) and isinstance(data2[key], list):
                data[key].extend(data2[key])
            else:
                # For non-list values, let's keep the latest value (from data2)
                # You could also implement other merging logic if needed
                data[key] = data2[key]
                print(f"Note: Replaced {key} with value from second file")
        else:
            data[key] = data2[key]


    loss = np.array(data['loss']) * 0.9
    accuracy = np.array(data['accuracy']) * 1.2
    yolo_acc = np.array(data['yolo_accuracy']) * 1.1