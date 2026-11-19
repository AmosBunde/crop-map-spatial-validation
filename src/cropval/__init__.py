"""Spatial validation for smallholder crop-type mapping (CV4A Kenya dataset)."""

CLASSES = {
    1: "Maize",
    2: "Cassava",
    3: "Common bean",
    4: "Maize + common bean",
    5: "Maize + cassava",
    6: "Maize + soybean",
    7: "Cassava + common bean",
}
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
TILES = ["0", "1", "2", "3"]
