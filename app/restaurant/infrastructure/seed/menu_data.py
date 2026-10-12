"""Deterministic seed fixtures for the restaurant menu (no randomness, stable ids).

Changing any dish content changes ``MENU_DATASET_VERSION`` expectations: bump the version string
whenever the menu fixture is edited so index fingerprints and evaluation labels stay consistent.
"""

from __future__ import annotations

from decimal import Decimal

from app.restaurant.domain.canonical import MENU_DATASET_VERSION  # noqa: F401  (re-exported)
from app.restaurant.domain.entities import Dish

RESTAURANT_ID = "rest_001"
RESTAURANT_NAME = "AI Waiter"
RESTAURANT_TAGLINE = "Fresh, fire-grilled & made to order"

CATEGORIES: list[tuple[str, str]] = [
    ("Burgers", "🍔"),
    ("Pizza", "🍕"),
    ("Pasta", "🍝"),
    ("Chicken", "🍗"),
    ("Salads", "🥗"),
    ("Healthy Meals", "🥑"),
    ("Desserts", "🍰"),
    ("Drinks", "🥤"),
]

# (name, description, ingredients, allergens, price USD, kcal, protein_g, spice 0-3, tags, available)
_Row = tuple[str, str, list[str], list[str], float, int, int, int, list[str], bool]

_MENU: dict[str, list[_Row]] = {
    "Burgers": [
        ("Classic Beef Burger", "Flame-grilled beef patty with melted cheddar, crisp lettuce, tomato and house sauce on a toasted brioche bun.",
         ["beef patty", "cheddar cheese", "lettuce", "tomato", "pickles", "brioche bun", "house sauce"], ["gluten", "dairy", "eggs", "sesame"], 13.0, 780, 38, 0, [], True),
        ("Spicy Jalapeno Burger", "Spicy beef burger with pepper jack, pickled jalapenos, crispy onion rings and smoky chipotle mayo.",
         ["beef patty", "pepper jack cheese", "jalapenos", "onion rings", "chipotle mayo", "brioche bun"], ["gluten", "dairy", "eggs", "sesame"], 14.5, 840, 40, 3, [], True),
        ("Mushroom Swiss Burger", "Juicy beef patty topped with Swiss cheese, sauteed mushrooms and garlic aioli.",
         ["beef patty", "swiss cheese", "mushrooms", "garlic aioli", "brioche bun"], ["gluten", "dairy", "eggs", "sesame"], 15.0, 810, 39, 0, [], True),
        ("Double Smash Burger", "Two smashed beef patties with American cheese, caramelized onion and burger sauce. Our biggest burger.",
         ["beef patty", "american cheese", "caramelized onion", "burger sauce", "brioche bun"], ["gluten", "dairy", "eggs", "sesame", "mustard"], 17.5, 1020, 58, 0, [], True),
        ("Crispy Chicken Burger", "Crunchy buttermilk fried chicken fillet with creamy coleslaw and mayo.",
         ["fried chicken fillet", "coleslaw", "mayo", "brioche bun", "pickles"], ["gluten", "dairy", "eggs", "sesame"], 12.5, 720, 34, 1, [], True),
        ("Veggie Falafel Burger", "Plant-based falafel patty with tahini, tomato, pickles and arugula on a whole wheat bun.",
         ["falafel patty", "tahini", "tomato", "pickles", "arugula", "whole wheat bun"], ["gluten", "sesame"], 10.5, 560, 17, 1, ["vegetarian", "vegan"], True),
        ("BBQ Bacon Burger", "Beef patty with smoked bacon, cheddar, sweet BBQ sauce and crispy onion rings.",
         ["beef patty", "bacon", "cheddar cheese", "bbq sauce", "onion rings", "brioche bun"], ["gluten", "dairy", "eggs", "sesame"], 16.5, 930, 44, 0, [], True),
        ("Peanut Satay Chicken Burger", "Grilled chicken breast with creamy peanut satay sauce, cucumber and sweet chili.",
         ["grilled chicken breast", "peanut satay sauce", "cucumber", "sweet chili sauce", "brioche bun"], ["gluten", "peanuts", "soy", "sesame"], 13.5, 690, 36, 2, [], True),
    ],
    "Pizza": [
        ("Margherita Pizza", "Classic wood-fired pizza with San Marzano tomato sauce, fresh mozzarella and basil.",
         ["pizza dough", "tomato sauce", "mozzarella", "fresh basil", "olive oil"], ["gluten", "dairy"], 11.5, 820, 34, 0, ["vegetarian"], True),
        ("Pepperoni Pizza", "Thin crust pizza loaded with spicy pepperoni slices and bubbling mozzarella.",
         ["pizza dough", "tomato sauce", "mozzarella", "pepperoni"], ["gluten", "dairy"], 13.0, 960, 40, 1, [], True),
        ("Four Cheese Pizza", "Rich pizza with mozzarella, gorgonzola, parmesan and provolone.",
         ["pizza dough", "mozzarella", "gorgonzola", "parmesan", "provolone"], ["gluten", "dairy"], 14.0, 1040, 46, 0, ["vegetarian"], True),
        ("BBQ Chicken Pizza", "Grilled chicken, sweet BBQ sauce, red onion and mozzarella on a crisp base.",
         ["pizza dough", "grilled chicken", "bbq sauce", "red onion", "mozzarella"], ["gluten", "dairy"], 14.5, 910, 48, 0, [], True),
        ("Spicy Diavola Pizza", "Fiery pizza with spicy salami, chili flakes, fresh chilies and mozzarella.",
         ["pizza dough", "tomato sauce", "spicy salami", "chili flakes", "fresh chilies", "mozzarella"], ["gluten", "dairy"], 15.0, 990, 42, 3, [], True),
        ("Veggie Garden Pizza", "Colorful pizza with bell peppers, mushrooms, olives, zucchini and mozzarella.",
         ["pizza dough", "tomato sauce", "bell peppers", "mushrooms", "olives", "zucchini", "mozzarella"], ["gluten", "dairy"], 12.5, 760, 28, 0, ["vegetarian"], True),
        ("Pesto Chicken Pizza", "Basil pesto base topped with grilled chicken, cherry tomatoes and parmesan.",
         ["pizza dough", "basil pesto", "pine nuts", "grilled chicken", "cherry tomatoes", "parmesan"], ["gluten", "dairy", "tree_nuts"], 15.5, 880, 46, 0, [], True),
        ("Tuna Sweetcorn Pizza", "Mediterranean pizza with flaked tuna, sweetcorn, olives and mozzarella.",
         ["pizza dough", "tomato sauce", "tuna", "sweetcorn", "olives", "mozzarella"], ["gluten", "dairy", "fish"], 13.5, 850, 44, 0, [], True),
    ],
    "Pasta": [
        ("Spaghetti Bolognese", "Slow-cooked beef ragu with tomato and herbs over spaghetti, finished with parmesan.",
         ["spaghetti", "ground beef", "tomato", "onion", "carrot", "parmesan"], ["gluten", "dairy"], 11.5, 710, 36, 0, [], True),
        ("Fettuccine Alfredo", "Silky fettuccine tossed in butter, cream and parmesan sauce.",
         ["fettuccine", "cream", "butter", "parmesan", "black pepper"], ["gluten", "dairy", "eggs"], 12.5, 890, 24, 0, ["vegetarian"], True),
        ("Penne Arrabbiata", "Spicy penne in a garlic, chili and tomato sauce with fresh parsley. Vegan and fiery.",
         ["penne", "tomato", "garlic", "chili", "parsley", "olive oil"], ["gluten"], 10.0, 520, 16, 3, ["vegetarian", "vegan"], True),
        ("Shrimp Linguine", "Linguine with garlic butter shrimp, white wine, chili and lemon.",
         ["linguine", "shrimp", "garlic", "white wine", "chili", "lemon", "butter"], ["gluten", "shellfish", "dairy"], 17.5, 640, 38, 1, [], True),
        ("Pesto Genovese Penne", "Penne in a fresh basil pesto with pine nuts, parmesan and olive oil.",
         ["penne", "basil pesto", "pine nuts", "parmesan", "olive oil"], ["gluten", "dairy", "tree_nuts"], 11.5, 680, 22, 0, ["vegetarian"], True),
        ("Chicken Mushroom Pasta", "Creamy rigatoni with grilled chicken, mushrooms and thyme.",
         ["rigatoni", "grilled chicken", "mushrooms", "cream", "thyme", "parmesan"], ["gluten", "dairy"], 13.0, 780, 41, 0, [], True),
        ("Lasagna al Forno", "Layers of pasta, beef ragu, bechamel and mozzarella baked until golden.",
         ["lasagna sheets", "ground beef", "bechamel", "mozzarella", "tomato"], ["gluten", "dairy", "eggs"], 13.5, 840, 38, 0, [], True),
        ("Whole Wheat Veggie Pasta", "Light whole wheat penne with zucchini, cherry tomatoes, spinach and olive oil.",
         ["whole wheat penne", "zucchini", "cherry tomatoes", "spinach", "olive oil", "garlic"], ["gluten"], 9.5, 430, 17, 0, ["vegetarian", "vegan"], True),
    ],
    "Chicken": [
        ("Grilled Lemon Herb Chicken", "Tender grilled chicken breast marinated in lemon and herbs, with roasted potatoes and steamed broccoli.",
         ["chicken breast", "lemon", "thyme", "rosemary", "potatoes", "broccoli"], [], 15.5, 480, 52, 0, ["gluten_free"], True),
        ("Crispy Fried Chicken", "Four pieces of golden buttermilk fried chicken with fries and garlic dip.",
         ["chicken pieces", "buttermilk", "flour", "fries", "garlic dip"], ["gluten", "dairy", "eggs"], 14.0, 950, 56, 1, [], True),
        ("Chicken Shawarma Plate", "Marinated shawarma chicken with garlic sauce, pickles, fries and warm pita.",
         ["chicken thigh", "garlic sauce", "pickles", "fries", "pita bread"], ["gluten", "sesame"], 12.5, 820, 44, 1, [], True),
        ("Spicy Peri-Peri Chicken", "Half chicken flame-grilled in fiery peri-peri sauce, served with rice. Very spicy and very high in protein.",
         ["chicken", "peri-peri sauce", "chili", "garlic", "rice"], [], 16.5, 710, 58, 3, ["gluten_free"], True),
        ("Chicken Parmigiana", "Breaded chicken breast baked with tomato sauce and melted mozzarella, served with spaghetti.",
         ["chicken breast", "breadcrumbs", "tomato sauce", "mozzarella", "spaghetti"], ["gluten", "dairy", "eggs"], 15.0, 860, 52, 0, [], True),
        ("Chicken Teriyaki Rice Bowl", "Glazed teriyaki chicken over steamed rice with sesame seeds and scallions.",
         ["chicken thigh", "teriyaki sauce", "rice", "sesame seeds", "scallions"], ["soy", "sesame", "gluten"], 14.5, 620, 40, 0, [], True),
        ("Butter Chicken with Rice", "Mild creamy tomato curry with tender chicken, cashew cream and basmati rice.",
         ["chicken", "tomato", "cream", "butter", "cashews", "basmati rice", "garam masala"], ["dairy", "tree_nuts"], 15.5, 790, 42, 2, [], True),
        ("Chicken Satay Skewers", "Grilled chicken skewers with warm peanut sauce and cucumber salad.",
         ["chicken breast", "peanut sauce", "cucumber", "lime"], ["peanuts", "soy"], 11.5, 540, 38, 1, [], True),
    ],
    "Salads": [
        ("Caesar Salad with Grilled Chicken", "Romaine lettuce, grilled chicken, parmesan, croutons and classic Caesar dressing.",
         ["romaine lettuce", "grilled chicken", "parmesan", "croutons", "caesar dressing", "anchovies"], ["gluten", "dairy", "eggs", "fish"], 11.0, 450, 36, 0, [], True),
        ("Greek Salad", "Crunchy cucumber, tomato, red onion, olives and feta with oregano dressing.",
         ["cucumber", "tomato", "red onion", "olives", "feta cheese", "oregano"], ["dairy"], 8.0, 320, 12, 0, ["vegetarian", "gluten_free"], True),
        ("Quinoa Avocado Salad", "Protein-rich quinoa with avocado, chickpeas, cucumber and lemon dressing.",
         ["quinoa", "avocado", "chickpeas", "cucumber", "lemon", "olive oil"], [], 10.0, 380, 14, 0, ["vegetarian", "vegan", "gluten_free"], True),
        ("Tuna Nicoise Salad", "Seared tuna with boiled egg, green beans, potatoes and olives on mixed greens.",
         ["tuna", "egg", "green beans", "potatoes", "olives", "mixed greens"], ["fish", "eggs", "mustard"], 12.5, 410, 34, 0, ["gluten_free"], True),
        ("Asian Peanut Noodle Salad", "Cold soba noodles with crunchy vegetables and a spicy peanut-sesame dressing.",
         ["soba noodles", "carrot", "red cabbage", "peanuts", "sesame oil", "soy sauce", "chili"], ["gluten", "peanuts", "soy", "sesame"], 10.0, 520, 15, 2, ["vegetarian", "vegan"], True),
        ("Roasted Beet and Goat Cheese Salad", "Roasted beets, creamy goat cheese, toasted walnuts and arugula with balsamic glaze.",
         ["beets", "goat cheese", "walnuts", "arugula", "balsamic glaze"], ["dairy", "tree_nuts"], 9.5, 360, 13, 0, ["vegetarian", "gluten_free"], True),
        ("Chicken Cobb Salad", "Grilled chicken, bacon, boiled egg, avocado, tomato and blue cheese on crisp greens.",
         ["grilled chicken", "bacon", "egg", "avocado", "tomato", "blue cheese", "romaine lettuce"], ["dairy", "eggs"], 13.0, 520, 42, 0, ["gluten_free"], True),
        ("Lentil and Roasted Veg Salad", "Warm lentils with roasted zucchini, peppers, red onion and lemon-herb vinaigrette.",
         ["lentils", "zucchini", "bell peppers", "red onion", "lemon", "parsley"], [], 8.5, 340, 16, 0, ["vegetarian", "vegan", "gluten_free"], True),
    ],
    "Healthy Meals": [
        ("Grilled Chicken Bowl", "Grilled chicken breast with rice and fresh vegetables.",
         ["chicken breast", "rice", "broccoli", "carrot", "olive oil"], [], 18.5, 390, 42, 0, ["gluten_free"], True),
        ("Salmon Quinoa Bowl", "Oven-baked salmon over quinoa with asparagus, spinach and lemon.",
         ["salmon", "quinoa", "asparagus", "spinach", "lemon"], ["fish"], 21.5, 520, 38, 0, ["gluten_free"], True),
        ("Turkey Meatballs with Zucchini Noodles", "Lean turkey meatballs in tomato sauce over spiralized zucchini. Low carb and low calorie.",
         ["ground turkey", "zucchini", "tomato sauce", "garlic", "basil"], ["eggs"], 16.0, 340, 36, 0, ["gluten_free"], True),
        ("Tofu Teriyaki Veggie Bowl", "Crispy tofu with steamed brown rice, broccoli, edamame and light teriyaki glaze.",
         ["tofu", "brown rice", "broccoli", "edamame", "teriyaki sauce", "sesame seeds"], ["soy", "sesame"], 12.5, 430, 24, 0, ["vegetarian", "vegan"], True),
        ("Lean Beef and Sweet Potato Plate", "Grilled lean beef sirloin with roasted sweet potato and green beans.",
         ["beef sirloin", "sweet potato", "green beans", "olive oil"], [], 19.5, 510, 46, 0, ["gluten_free"], True),
        ("Shrimp and Avocado Power Bowl", "Garlic shrimp with avocado, brown rice, edamame and lime. Currently sold out.",
         ["shrimp", "avocado", "brown rice", "edamame", "lime"], ["shellfish", "soy"], 20.5, 430, 34, 0, ["gluten_free"], False),
        ("Egg White Veggie Omelette", "Fluffy egg white omelette with spinach, mushrooms and tomato. A light, low-calorie option.",
         ["egg whites", "spinach", "mushrooms", "tomato"], ["eggs"], 8.0, 210, 24, 0, ["vegetarian", "gluten_free"], True),
        ("Chickpea Buddha Bowl", "Roasted chickpeas, brown rice, roasted vegetables and lemon-tahini dressing.",
         ["chickpeas", "brown rice", "sweet potato", "kale", "tahini", "lemon"], ["sesame"], 10.5, 520, 18, 0, ["vegetarian", "vegan"], True),
    ],
    "Desserts": [
        ("Chocolate Lava Cake", "Warm dark chocolate cake with a molten center, served with vanilla ice cream.",
         ["dark chocolate", "flour", "butter", "eggs", "sugar", "vanilla ice cream"], ["gluten", "dairy", "eggs"], 6.5, 480, 7, 0, ["vegetarian"], True),
        ("New York Cheesecake", "Creamy baked cheesecake on a buttery biscuit base with strawberry sauce.",
         ["cream cheese", "biscuit base", "eggs", "sugar", "strawberry sauce"], ["gluten", "dairy", "eggs"], 6.5, 520, 8, 0, ["vegetarian"], True),
        ("Tiramisu", "Espresso-soaked ladyfingers layered with mascarpone cream and cocoa. Seasonal, currently unavailable.",
         ["ladyfingers", "mascarpone", "espresso", "eggs", "cocoa"], ["gluten", "dairy", "eggs"], 7.5, 430, 7, 0, ["vegetarian"], False),
        ("Fresh Fruit Bowl", "Seasonal fruit with mint and a squeeze of lime. Light, fat-free and vegan.",
         ["watermelon", "strawberries", "kiwi", "grapes", "mint", "lime"], [], 4.5, 120, 2, 0, ["vegetarian", "vegan", "gluten_free"], True),
        ("Baklava", "Flaky phyllo layered with pistachios and walnuts, soaked in honey syrup.",
         ["phyllo dough", "pistachios", "walnuts", "honey syrup", "butter"], ["gluten", "dairy", "tree_nuts"], 5.5, 410, 6, 0, ["vegetarian"], True),
        ("Peanut Butter Brownie", "Fudgy chocolate brownie swirled with peanut butter and topped with crushed peanuts.",
         ["dark chocolate", "peanut butter", "flour", "eggs", "butter", "peanuts"], ["gluten", "dairy", "eggs", "peanuts"], 6.0, 520, 9, 0, ["vegetarian"], True),
        ("Coconut Chia Pudding", "Creamy chia seeds soaked in coconut milk with mango. Vegan and gluten-free.",
         ["chia seeds", "coconut milk", "mango", "maple syrup"], [], 5.5, 230, 6, 0, ["vegetarian", "vegan", "gluten_free"], True),
        ("Basbousa with Cream", "Traditional semolina cake soaked in syrup and served with clotted cream.",
         ["semolina", "sugar syrup", "coconut", "butter", "clotted cream"], ["gluten", "dairy"], 5.0, 390, 5, 0, ["vegetarian"], True),
    ],
    "Drinks": [
        ("Fresh Orange Juice", "Freshly squeezed orange juice, no added sugar.",
         ["oranges"], [], 4.0, 110, 2, 0, ["vegetarian", "vegan", "gluten_free"], True),
        ("Mango Smoothie", "Thick mango smoothie blended with yogurt and a hint of honey.",
         ["mango", "yogurt", "honey", "ice"], ["dairy"], 5.0, 220, 5, 0, ["vegetarian", "gluten_free"], True),
        ("Iced Coffee", "Cold-brewed coffee over ice with milk and vanilla syrup.",
         ["cold brew coffee", "milk", "vanilla syrup", "ice"], ["dairy"], 4.5, 90, 3, 0, ["vegetarian", "gluten_free"], True),
        ("Vanilla Protein Shake", "Post-workout shake with whey protein, banana, milk and vanilla. High protein drink.",
         ["whey protein", "banana", "milk", "vanilla"], ["dairy"], 7.5, 260, 30, 0, ["vegetarian", "gluten_free"], True),
        ("Mint Lemonade", "Refreshing lemonade blended with fresh mint leaves and ice.",
         ["lemon juice", "mint", "sugar", "ice"], [], 3.5, 130, 0, 0, ["vegetarian", "vegan", "gluten_free"], True),
        ("Green Detox Juice", "Cold-pressed spinach, apple, cucumber and ginger juice.",
         ["spinach", "green apple", "cucumber", "ginger"], [], 5.5, 90, 2, 0, ["vegetarian", "vegan", "gluten_free"], True),
        ("Hibiscus Tea (Karkade)", "Chilled or hot Egyptian hibiscus tea, naturally caffeine-free.",
         ["hibiscus flowers", "water", "sugar"], [], 2.5, 5, 0, 0, ["vegetarian", "vegan", "gluten_free"], True),
        ("Peanut Butter Banana Shake", "Thick milkshake with peanut butter, banana and vanilla ice cream.",
         ["peanut butter", "banana", "milk", "vanilla ice cream"], ["dairy", "peanuts"], 7.0, 410, 14, 0, ["vegetarian", "gluten_free"], True),
    ],
}


def build_menu() -> list[Dish]:
    dishes: list[Dish] = []
    n = 0
    for category, _emoji in CATEGORIES:
        for name, desc, ingredients, allergens, price, kcal, protein, spice, tags, available in _MENU[category]:
            n += 1
            dishes.append(
                Dish(
                    id=f"dish_{n:03d}",
                    name=name,
                    category=category,
                    description=desc,
                    ingredients=tuple(ingredients),
                    allergens=tuple(sorted(allergens)),
                    price=Decimal(str(price)),
                    currency="USD",
                    image_url=f"/dishes/dish_{n:03d}.jpg",
                    calories=kcal,
                    protein_grams=protein,
                    spice_level=spice,
                    tags=tuple(tags),
                    available=available,
                    sort_order=n,
                )
            )
    return dishes
