def read_coordinates(filename, max_rows=19):
    coordinates = []
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if i >= max_rows:
                break  
            line = line.strip()
            if not line:
                continue
            try:
                x, y = map(int, line.split(','))
                coordinates.append((x, y))
            except ValueError:
                print(f"Skipping invalid line in {filename}: {line}")
                continue
    return coordinates

def normalize_coordinates(coordinates, original_size, standardized_size):
    orig_width, orig_height = original_size
    std_width, std_height = standardized_size
    return [(x * std_width / orig_width, y * std_height / orig_height) for x, y in coordinates]

def validate_coordinates(coordinates):
    for coord in coordinates:
        if not isinstance(coord, tuple) or len(coord) != 2:
            raise ValueError("Invalid coordinate format. Expected a tuple of (x, y).")
        if not (isinstance(coord[0], int) and isinstance(coord[1], int)):
            raise ValueError("Coordinates should be integers.")
