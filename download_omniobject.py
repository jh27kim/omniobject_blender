import subprocess

CATEGORY = [
    "backpack",
    "bed",
    "book",
    "bottle",
    "brush", 
    "cake",
    "candle",
    "chair",
    "clock",
    "dice",
    "cup",
    "dumbbell",
    "fan",
    "fire_extinguisher",
    "fork",
    "gloves",
    "guitar",
    "hair_dryer",
    "hamburger",
    "hammer",
    "hand_cream",
    "handbag",
    "hat",
    "helmet",
    "hot_dog",
    "keyboard",
    "knife",
    "laptop",
    "microwaveoven",
    "monitor",
    "mouse",
    "peanut",
    "pen",
    "picnic_basket",
    "pineapple",
    "red_wine_glass",
    "scissor",
    "shoe",
    "shrimp",
    "skateboard",
    "sofa",
    "sushi",
    "table",
    "umbrella",
    "walnet",
    "watermelon",
]

failed_list = []
for cat in CATEGORY:
    try:
        subprocess.run(["odl", "get", f"OpenXD-OmniObject3D-New/raw/raw_scans/{cat}.tar.gz"])
    except:
        failed_list.append(cat)

print("failed object")
print(failed_list)
