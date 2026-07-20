# training/splits.py

# ======================================================
# DOSTĘPNE SPLITY
# ======================================================

SPLITS = {

    1: [
        "FBW"
    ],

    2: [
        "Upper",
        "Lower"
    ],

    3: [
        "Push",
        "Pull",
        "Legs"
    ],

    4: [
        "Upper A",
        "Lower A",
        "Upper B",
        "Lower B"
    ],

    5: [
        "Push",
        "Pull",
        "Legs",
        "Upper",
        "Lower"
    ],

    6: [
        "Push A",
        "Pull A",
        "Legs A",
        "Push B",
        "Pull B",
        "Legs B"
    ]

}


# ======================================================
# PARTIE DLA POSZCZEGÓLNYCH TRENINGÓW
# ======================================================

SPLIT_CONTENT = {

    # ---------- FBW ----------

    "FBW": [
        "Klatka",
        "Plecy",
        "Barki",
        "Nogi",
        "Biceps",
        "Triceps",
        "Brzuch"
    ],

    # ---------- UPPER / LOWER ----------

    "Upper": [
        "Klatka",
        "Plecy",
        "Barki",
        "Biceps",
        "Triceps"
    ],

    "Lower": [
        "Nogi",
        "Brzuch"
    ],

    # ---------- PUSH / PULL / LEGS ----------

    "Push": [
        "Klatka",
        "Barki",
        "Triceps"
    ],

    "Pull": [
        "Plecy",
        "Biceps"
    ],

    "Legs": [
        "Nogi",
        "Brzuch"
    ],

    # ---------- UPPER LOWER A/B ----------

    "Upper A": [
        "Klatka",
        "Plecy",
        "Barki",
        "Biceps",
        "Triceps"
    ],

    "Upper B": [
        "Klatka",
        "Plecy",
        "Barki",
        "Biceps",
        "Triceps"
    ],

    "Lower A": [
        "Nogi",
        "Brzuch"
    ],

    "Lower B": [
        "Nogi",
        "Brzuch"
    ],

    # ---------- PUSH PULL LEGS A/B ----------

    "Push A": [
        "Klatka",
        "Barki",
        "Triceps"
    ],

    "Push B": [
        "Klatka",
        "Barki",
        "Triceps"
    ],

    "Pull A": [
        "Plecy",
        "Biceps"
    ],

    "Pull B": [
        "Plecy",
        "Biceps"
    ],

    "Legs A": [
        "Nogi",
        "Brzuch"
    ],

    "Legs B": [
        "Nogi",
        "Brzuch"
    ]

}


# ======================================================
# NAZWY DNI TYGODNIA
# ======================================================

DAYS = [
    "Poniedziałek",
    "Wtorek",
    "Środa",
    "Czwartek",
    "Piątek",
    "Sobota",
    "Niedziela"
]