
from ultralytics import YOLO
import sys

with open("debug_output.txt", "w") as f:
    try:
        model = YOLO("yolo26n-seg.pt")
        f.write(f"Model type: {type(model)}\n")
        
        # Check model.names
        if hasattr(model, 'names'):
            f.write(f"Has model.names: Yes\n")
            f.write(f"Type: {type(model.names)}\n")
            f.write(f"Value: {model.names}\n")
        else:
            f.write("Has model.names: No\n")
            
        # Check model.model.names
        if hasattr(model, 'model'):
            if hasattr(model.model, 'names'):
                f.write(f"Has model.model.names: Yes\n")
                f.write(f"Value: {model.model.names}\n")
            else:
                f.write("Has model.model.names: No\n")
        else:
            f.write("Has model.model: No\n")

    except Exception as e:
        f.write(f"Error: {e}\n")
