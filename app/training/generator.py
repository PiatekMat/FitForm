# training/generator.py

import random

from .exercies import EXERCISES
from .rules import TRAINING_RULES
from .splits import SPLITS, SPLIT_CONTENT, DAYS

# =====================================================
# Docelowa objętość tygodniowa (liczba serii)
# =====================================================

WEEKLY_VOLUME = {

    "male": {

        "masa": {
            "Klatka": 18,
            "Plecy": 18,
            "Barki": 16,
            "Biceps": 12,
            "Triceps": 12,
            "Nogi": 18,
            "Brzuch": 10
        },

        "redukcja": {
            "Klatka": 14,
            "Plecy": 14,
            "Barki": 12,
            "Biceps": 10,
            "Triceps": 10,
            "Nogi": 16,
            "Brzuch": 10
        },

        "utrzymanie": {
            "Klatka": 16,
            "Plecy": 16,
            "Barki": 14,
            "Biceps": 10,
            "Triceps": 10,
            "Nogi": 16,
            "Brzuch": 8
        }

    },

    "female": {

        "masa": {
            "Klatka": 12,
            "Plecy": 14,
            "Barki": 14,
            "Biceps": 8,
            "Triceps": 8,
            "Nogi": 20,
            "Brzuch": 10
        },

        "redukcja": {
            "Klatka": 10,
            "Plecy": 12,
            "Barki": 12,
            "Biceps": 8,
            "Triceps": 8,
            "Nogi": 18,
            "Brzuch": 10
        },

        "utrzymanie": {
            "Klatka": 10,
            "Plecy": 12,
            "Barki": 12,
            "Biceps": 8,
            "Triceps": 8,
            "Nogi": 18,
            "Brzuch": 8
        }

    }

}
# ===============================================
# Funkcje pomocnicze
# ===============================================

def get_goal(predicted_weight_change):
    """
    predicted_weight_change = przewidywana zmiana masy po symulacji

    np.

    -2.1 -> redukcja
    +1.3 -> masa
    0.05 -> utrzymanie
    """

    if predicted_weight_change >= 0.5:
        return "masa"

    elif predicted_weight_change <= -0.5:
        return "redukcja"

    return "utrzymanie"


def get_rules(gender, goal, technical):

    return TRAINING_RULES[gender][goal][technical]


def get_split(training_days):

    return SPLITS[training_days]


# ===============================================
# Wybór ćwiczeń dla jednej partii
# ===============================================

def choose_exercises(
        muscle_group,
        exercise_count,
        fatigue_limit,
        used_exercises
):
    """
    Dobiera ćwiczenia dla jednej partii.

    Zasady:

    ✔ najpierw wielostawowe
    ✔ później średnie
    ✔ na końcu izolacje
    ✔ bez powtórzeń
    ✔ z kontrolą fatigue
    """

    available = [

        ex

        for ex in EXERCISES[muscle_group]

        if ex["name"] not in used_exercises[muscle_group]

    ]


    # ----------------------------
    # Podział na grupy
    # ----------------------------

    heavy = [

        ex

        for ex in available

        if ex["compound"] and ex["technical"] >= 3

    ]

    medium = [

        ex

        for ex in available

        if ex["compound"] and ex["technical"] <= 2

    ]

    isolation = [

        ex

        for ex in available

        if not ex["compound"]

    ]


    # losowanie wewnątrz kategorii

    random.shuffle(heavy)
    random.shuffle(medium)
    random.shuffle(isolation)

    chosen = []

    fatigue = 0


    # =====================================
    # 1 ciężkie ćwiczenie
    # =====================================

    if heavy:

        ex = heavy.pop(0)

        chosen.append(ex)

        fatigue += ex["fatigue"]

        used_exercises[muscle_group].add(ex["name"])


    # =====================================
    # 1 średnie
    # =====================================

    if len(chosen) < exercise_count:

        if medium:

            ex = medium.pop(0)

            if fatigue + ex["fatigue"] <= fatigue_limit:

                chosen.append(ex)

                fatigue += ex["fatigue"]

                used_exercises[muscle_group].add(ex["name"])


    # =====================================
    # reszta izolacji
    # =====================================

    while len(chosen) < exercise_count and isolation:

        ex = isolation.pop(0)

        if fatigue + ex["fatigue"] > fatigue_limit:

            break

        chosen.append(ex)

        fatigue += ex["fatigue"]

        used_exercises[muscle_group].add(ex["name"])


    # =====================================
    # jeżeli nadal brakuje ćwiczeń
    # =====================================

    leftovers = heavy + medium + isolation

    random.shuffle(leftovers)

    while len(chosen) < exercise_count and leftovers:

        ex = leftovers.pop()

        if ex in chosen:
            continue

        chosen.append(ex)

        used_exercises[muscle_group].add(ex["name"])

    return chosen

# ===============================================
# Budowanie pojedynczego treningu
# ===============================================

EXERCISE_COUNT = {

    "Klatka": 3,
    "Plecy": 3,
    "Barki": 3,

    "Biceps": 2,
    "Triceps": 2,

    "Nogi": 4,

    "Brzuch": 2

}


# maksymalne zmęczenie pojedynczego treningu

MAX_FATIGUE = {

    1: 22,
    2: 18,
    3: 16,
    4: 15,
    5: 14,
    6: 13

}


def build_workout(
        split_name,
        gender,
        goal,
        training_days,
        used_exercises
):

    muscles = SPLIT_CONTENT[split_name]

    workout = []

    current_fatigue = 0

    fatigue_limit = MAX_FATIGUE[training_days]

    for muscle in muscles:

        remaining = fatigue_limit - current_fatigue

        exercises = choose_exercises(

            muscle_group=muscle,

            exercise_count=EXERCISE_COUNT[muscle],

            fatigue_limit=max(remaining, 3),

            used_exercises=used_exercises

        )

        exercise_list = []

        for exercise in exercises:

            params = get_rules(

                gender,

                goal,

                exercise["technical"]

            )

            exercise_list.append({

                "name": exercise["name"],

                "compound": exercise["compound"],

                "technical": exercise["technical"],

                "fatigue": exercise["fatigue"],

                "priority": exercise["priority"],

                "sets": params["sets"],

                "reps": params["reps"],

                "rest": params["rest"],

                "tempo": params["tempo"],

                "rpe": params["rpe"]

            })

            current_fatigue += exercise["fatigue"]

        workout.append({

            "muscle": muscle,

            "exercises": exercise_list

        })

    return {

        "split": split_name,

        "fatigue": current_fatigue,

        "muscles": workout

    }

def generate_training_plan(
        gender,
        predicted_weight_change,
        training_days
):

    goal = get_goal(predicted_weight_change)

    split = get_split(training_days)

    used_exercises = {

        muscle: set()

        for muscle in EXERCISES

    }

    weekly_sets = {

        muscle: 0

        for muscle in EXERCISES

    }

    plan = []

    for day_index, split_name in enumerate(split):

        workout = build_workout(

            split_name,

            gender,

            goal,

            training_days,

            used_exercises

        )

        # ===================================
        # kontrola objętości
        # ===================================

        for muscle in workout["muscles"]:

            muscle_name = muscle["muscle"]

            target = WEEKLY_VOLUME[gender][goal][muscle_name]

            for exercise in muscle["exercises"]:

                remaining = target - weekly_sets[muscle_name]

                if remaining <= 0:

                    exercise["sets"] = 0

                    continue

                if exercise["sets"] > remaining:

                    exercise["sets"] = remaining

                weekly_sets[muscle_name] += exercise["sets"]

        workout["day"] = DAYS[day_index]

        plan.append(workout)

    return {

        "goal": goal,

        "training_days": training_days,

        "plan": plan

    }