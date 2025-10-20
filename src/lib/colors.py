COLORS = {
    "$CosmicGold": (20, 150, 255),
    "$DeepSpaceBlue": (100, 30, 20),
    "$NebulaMagenta": (200, 40, 180),
    "$CyanGas": (220, 200, 0),
    "$Starlight": (150, 223, 255),
    "$RoyalVoid": (130, 0, 75),
    "$OxidizedRust": (20, 90, 200),
    "$OxygenTeal": (160, 180, 40),
    "$PaleHotYellow": (205, 250, 255),
    "$DeepCrimson": (30, 10, 150),
    "$SunsetOrange": (0, 120, 255),
    "$ElectricViolet": (211, 0, 148),
    "$LuminousMint": (175, 255, 100),
    "$CharcoalVoid": (30, 25, 25),
    "$StellarCrimson": (0, 50, 255),
    "$DeepRuby": (60, 0, 240),
    # Pure, overwhelming red. Subtracts blue and green.
    "$AggressiveHydrogenAlpha": (-150, -150, 255),
    # Intense orange-red for sulphur emissions (SII).
    "$SulphurBurn": (-100, 50, 255),
    # A piercing cyan for oxygen (OIII), subtracting red.
    "$OxygenGlow": (255, 200, -100),
    # Darkens everything it touches, enhancing shadows.
    "$VoidCrusher": (-50, -50, -50),
    # A brighter, more intense gold that suppresses blue.
    "$StarfireGold": (-50, 200, 255),
    # A vibrant teal that removes green, useful for specific nebula gases.
    "$PlasmaTeal": (255, -100, 0),
}

PALETTES = {
    "cosmic_embers": [
        ("f140m", "high", "$StellarCrimson", 0.5),
        ("f182m", "high", "$SunsetOrange", 0.9),  # Focus Color
        ("f212n", "low", "$CosmicGold", 0.6),
        ("f277w", "low", "$PaleHotYellow", 0.4),
        ("f300m", "both", "$Starlight", 0.3),
        ("f335m", "both", "$CyanGas", 0.4),
        ("f444w", "low", "$DeepSpaceBlue", 0.7),
    ],
    "gas_and_stars": [
        ("f187n", "high", "$NebulaMagenta", 0.6),
        ("f212n", "high", "$ElectricViolet", 0.5),
        ("f470n", "high", "$DeepRuby", 0.4),
        ("f115w", "low", "$Starlight", 0.3),
        ("f150w", "low", "$PaleHotYellow", 0.5),
        ("f200w", "low", "$CosmicGold", 0.9),  # Focus Color
        ("f277w", "low", "$OxygenTeal", 0.6),
        ("f335m", "low", "$CyanGas", 0.5),
        ("f444w", "low", "$DeepSpaceBlue", 0.7),
    ],
    "golden_nebula": [
        ("f140m", "high", "$DeepCrimson", 0.5),
        ("f182m", "low", "$OxidizedRust", 0.6),
        ("f277w", "low", "$CosmicGold", 0.8),
        ("f335m", "both", "$PaleHotYellow", 0.9),  # Focus Color
        ("f444w", "both", "$Starlight", 0.4),
    ],
}