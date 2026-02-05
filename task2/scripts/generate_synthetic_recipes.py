#!/usr/bin/env python3
import json
import random
from pathlib import Path


OUT_PATH = Path("data/raw/recipes.jsonl")
TARGET_COUNT = 500

BASE_RECIPES = [
    {
        "title": "Onion Egg Omelette",
        "ingredients": ["eggs", "onion", "salt", "pepper", "oil"],
        "instructions": [
            "Beat eggs with salt and pepper.",
            "Saute chopped onion in a pan with oil until soft.",
            "Pour eggs over onions and cook until set.",
            "Fold and serve warm.",
        ],
    },
    {
        "title": "Tomato Egg Stir-Fry",
        "ingredients": ["eggs", "tomato", "garlic", "salt", "oil", "sugar"],
        "instructions": [
            "Whisk eggs with a pinch of salt.",
            "Scramble eggs in oil, remove and set aside.",
            "Saute garlic and tomato with a little sugar and salt.",
            "Return eggs, stir to coat, serve with rice.",
        ],
    },
    {
        "title": "Potato Onion Hash",
        "ingredients": ["potato", "onion", "salt", "pepper", "oil"],
        "instructions": [
            "Dice potatoes and onions.",
            "Cook potatoes in oil until golden.",
            "Add onions and cook until softened.",
            "Season and serve crispy.",
        ],
    },
    {
        "title": "Garlic Butter Pasta",
        "ingredients": ["pasta", "garlic", "butter", "salt", "pepper", "parsley"],
        "instructions": [
            "Boil pasta until al dente.",
            "Melt butter and saute garlic.",
            "Toss pasta with garlic butter.",
            "Season and garnish with parsley.",
        ],
    },
    {
        "title": "Simple Veggie Fried Rice",
        "ingredients": ["rice", "egg", "onion", "carrot", "peas", "soy sauce", "oil"],
        "instructions": [
            "Scramble egg and set aside.",
            "Saute onion, carrot, peas in oil.",
            "Add rice and soy sauce, stir-fry.",
            "Return egg and mix well.",
        ],
    },
]

INGREDIENT_POOL = [
    "egg",
    "eggs",
    "onion",
    "garlic",
    "tomato",
    "potato",
    "carrot",
    "peas",
    "spinach",
    "bell pepper",
    "rice",
    "pasta",
    "bread",
    "cheese",
    "milk",
    "butter",
    "oil",
    "salt",
    "pepper",
    "soy sauce",
    "chili",
    "cumin",
    "coriander",
    "lemon",
    "ginger",
    "mushroom",
    "corn",
    "yogurt",
    "beans",
    "chickpeas",
]

TITLE_TEMPLATES = [
    "{main} with {secondary}",
    "{secondary} {main} Skillet",
    "{main} and {secondary} Bowl",
    "Quick {main} {secondary}",
    "{main} {secondary} Stir-Fry",
]

STEP_TEMPLATES = [
    "Prepare and chop the ingredients.",
    "Heat oil in a pan and add {secondary}.",
    "Add {main} and cook until tender.",
    "Season with salt and pepper, then serve warm.",
]


def pick_main_secondary():
    main = random.choice(["eggs", "pasta", "rice", "potato", "beans"])
    secondary = random.choice([i for i in INGREDIENT_POOL if i != main])
    return main, secondary


def build_recipe():
    main, secondary = pick_main_secondary()
    title = random.choice(TITLE_TEMPLATES).format(
        main=main.title(), secondary=secondary.title()
    )
    ingredients = {main, secondary, "salt", "pepper", "oil"}
    for _ in range(random.randint(2, 5)):
        ingredients.add(random.choice(INGREDIENT_POOL))
    instructions = [
        s.format(main=main, secondary=secondary) for s in STEP_TEMPLATES
    ]
    return {
        "title": title,
        "ingredients": sorted(list(ingredients)),
        "instructions": instructions,
    }


def main():
    random.seed(7)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    items = list(BASE_RECIPES)
    while len(items) < TARGET_COUNT:
        items.append(build_recipe())

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(items)} recipes to {OUT_PATH}")


if __name__ == "__main__":
    main()
