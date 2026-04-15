import os

label_dir = "yolo_car_1000/labels"

for root, _, files in os.walk(label_dir):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(root, file)

            new_lines = []
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        parts[0] = "0"  
                        new_lines.append(" ".join(parts))

            with open(path, "w") as f:
                f.write("\n".join(new_lines))
