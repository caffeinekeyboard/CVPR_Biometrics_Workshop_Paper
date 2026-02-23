import nbis
from nbis import NbisExtractorSettings
import os
import cv2
import numpy as np

# Resolve paths relative to this script
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_IMAGE_PATH = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', 'assets', 'anguli_fingerprint.png'))

# Configuration for the NbisExtractor
settings = NbisExtractorSettings(
    # Do not filter on minutiae quality (get all minutiae)
    min_quality=0.0,
    # Do not get the fingerprint center or ROI
    get_center=False,
    # Do not use SIVV to check if the image is a fingerprint
    check_fingerprint=False,
    # Compute the NFIQ2 quality score
    compute_nfiq2=True,
    # No specific PPI, use the default
    ppi=None,
)

extractor = nbis.new_nbis_extractor(settings)

# Read the image and extract minutiae
image_bytes = open(_IMAGE_PATH, "rb").read()
minutiae = extractor.extract_minutiae(image_bytes)

# Get the list of minutiae points
points = minutiae.get()
print(f"Extracted {len(points)} minutiae points")

# Load and display the image with minutiae
img = cv2.imread(_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load image at {_IMAGE_PATH}")

# Create visualization
canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Plot minutiae points
for i, minutia in enumerate(points):
    x = int(minutia.x())
    y = int(minutia.y())
    angle = np.radians(minutia.angle())  # Convert to radians
    kind = minutia.kind()
    
    # Draw minutiae point (red circle)
    cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)
    
    # Draw orientation line (yellow arrow)
    arrow_length = 15
    end_x = int(x + arrow_length * np.cos(angle))
    end_y = int(y + arrow_length * np.sin(angle))
    cv2.arrowedLine(canvas, (x, y), (end_x, end_y), (0, 255, 255), 1, tipLength=0.3)

# Add text labels
cv2.putText(canvas, f'NBIS Minutiae Detection: {len(points)} points', 
           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Save visualization
output_path = os.path.join(_SCRIPT_DIR, 'nbis_minutiae_detection.png')
cv2.imwrite(output_path, canvas)
print(f"Visualization saved to {output_path}")

# Print first few minutiae for verification
print("\nFirst 5 minutiae points:")
for i in range(min(5, len(points))):
    p = points[i]
    print(f"  {i+1}. x={p.x()}, y={p.y()}, angle={p.angle()}°, kind={p.kind()}, reliability={p.reliability():.4f}")