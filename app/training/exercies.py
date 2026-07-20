# training/exercises.py

EXERCISES = {

    # ==========================
    # KLATKA PIERSIOWA
    # ==========================

    "Klatka": [

        {
            "name": "Wyciskanie sztangi leżąc",
            "technical": 3,
            "fatigue": 3,
            "priority": 5,
            "compound": True
        },

        {
            "name": "Wyciskanie hantli leżąc",
            "technical": 2,
            "fatigue": 3,
            "priority": 4,
            "compound": True
        },

        {
            "name": "Wyciskanie hantli skos dodatni",
            "technical": 3,
            "fatigue": 3,
            "priority": 5,
            "compound": True
        },

        {
            "name": "Maszyna Chest Press",
            "technical": 1,
            "fatigue": 2,
            "priority": 3,
            "compound": True
        },

        {
            "name": "Dipy",
            "technical": 4,
            "fatigue": 4,
            "priority": 5,
            "compound": True
        },

        {
            "name": "Rozpiętki z hantlami",
            "technical": 1,
            "fatigue": 1,
            "priority": 2,
            "compound": False
        },

        {
            "name": "Butterfly",
            "technical": 1,
            "fatigue": 1,
            "priority": 1,
            "compound": False
        }

    ],

    # ==========================
    # PLECY
    # ==========================

    "Plecy": [

        {
            "name": "Martwy ciąg",
            "technical": 4,
            "fatigue": 4,
            "priority": 5,
            "compound": True
        },

        {
            "name": "Podciąganie nachwytem",
            "technical": 4,
            "fatigue": 3,
            "priority": 5,
            "compound": True
        },

        {
            "name": "Ściąganie drążka",
            "technical": 2,
            "fatigue": 2,
            "priority": 4,
            "compound": True
        },

        {
            "name": "Wiosłowanie sztangą",
            "technical": 3,
            "fatigue": 3,
            "priority": 5,
            "compound": True
        },

        {
            "name": "Wiosłowanie hantlem",
            "technical": 2,
            "fatigue": 2,
            "priority": 4,
            "compound": True
        },

        {
            "name": "Pullover na wyciągu",
            "technical": 1,
            "fatigue": 1,
            "priority": 2,
            "compound": False
        },

        {
            "name": "Face Pull",
            "technical": 1,
            "fatigue": 1,
            "priority": 2,
            "compound": False
        }

    ],

    # ==========================
    # BARKI
    # ==========================

    "Barki": [

        {
            "name": "Wyciskanie żołnierskie",
            "technical": 4,
            "fatigue": 4,
            "priority": 5,
            "compound": True
        },

        {
            "name": "Arnold Press",
            "technical": 3,
            "fatigue": 3,
            "priority": 4,
            "compound": True
        },

        {
            "name": "Wyciskanie hantli siedząc",
            "technical": 2,
            "fatigue": 3,
            "priority": 4,
            "compound": True
        },

        {
            "name": "Unoszenie bokiem",
            "technical": 1,
            "fatigue": 1,
            "priority": 3,
            "compound": False
        },

        {
            "name": "Unoszenie przodem",
            "technical": 1,
            "fatigue": 1,
            "priority": 2,
            "compound": False
        },

        {
            "name": "Odwrotne rozpiętki",
            "technical": 1,
            "fatigue": 1,
            "priority": 3,
            "compound": False
        }

    ],

    # ==========================
    # BICEPS
    # ==========================

    "Biceps": [

        {
            "name": "Uginanie sztangi",
            "technical": 2,
            "fatigue": 2,
            "priority": 4,
            "compound": False
        },

        {
            "name": "Hammer Curl",
            "technical": 2,
            "fatigue": 2,
            "priority": 4,
            "compound": False
        },

        {
            "name": "Uginanie hantli siedząc",
            "technical": 2,
            "fatigue": 2,
            "priority": 3,
            "compound": False
        },

        {
            "name": "Modlitewnik",
            "technical": 2,
            "fatigue": 2,
            "priority": 3,
            "compound": False
        },

        {
            "name": "Spider Curl",
            "technical": 2,
            "fatigue": 2,
            "priority": 2,
            "compound": False
        },

        {
            "name": "Linka wyciągu",
            "technical": 1,
            "fatigue": 1,
            "priority": 2,
            "compound": False
        }

    ],

    # ==========================
    # TRICEPS
    # ==========================

    "Triceps": [

        {
            "name": "Wąskie wyciskanie",
            "technical": 3,
            "fatigue": 3,
            "priority": 5,
            "compound": True
        },

        {
            "name": "Dipy na poręczach",
            "technical": 4,
            "fatigue": 3,
            "priority": 5,
            "compound": True
        },

        {
            "name": "Francuskie wyciskanie",
            "technical": 2,
            "fatigue": 2,
            "priority": 4,
            "compound": False
        },

        {
            "name": "Prostowanie na wyciągu",
            "technical": 1,
            "fatigue": 1,
            "priority": 3,
            "compound": False
        },

        {
            "name": "Prostowanie hantla nad głową",
            "technical": 2,
            "fatigue": 2,
            "priority": 3,
            "compound": False
        },

        {
            "name": "Kickback",
            "technical": 1,
            "fatigue": 1,
            "priority": 1,
            "compound": False
        }

    ],

    # ==========================
    # NOGI
    # ==========================

    "Nogi": [

        {
            "name": "Przysiad ze sztangą",
            "technical": 4,
            "fatigue": 4,
            "priority": 5,
            "compound": True
        },

        {
            "name": "Front Squat",
            "technical": 4,
            "fatigue": 4,
            "priority": 5,
            "compound": True
        },

        {
            "name": "Suwnica",
            "technical": 2,
            "fatigue": 3,
            "priority": 4,
            "compound": True
        },

        {
            "name": "Martwy ciąg RDL",
            "technical": 4,
            "fatigue": 4,
            "priority": 5,
            "compound": True
        },

        {
            "name": "Hip Thrust",
            "technical": 3,
            "fatigue": 3,
            "priority": 4,
            "compound": True
        },

        {
            "name": "Wykroki",
            "technical": 2,
            "fatigue": 3,
            "priority": 3,
            "compound": True
        },

        {
            "name": "Prostowanie nóg",
            "technical": 1,
            "fatigue": 1,
            "priority": 2,
            "compound": False
        },

        {
            "name": "Uginanie nóg",
            "technical": 1,
            "fatigue": 1,
            "priority": 2,
            "compound": False
        },

        {
            "name": "Wspięcia na palce",
            "technical": 1,
            "fatigue": 1,
            "priority": 1,
            "compound": False
        }

    ],

    # ==========================
    # BRZUCH
    # ==========================

    "Brzuch": [

        {
            "name": "Plank",
            "technical": 1,
            "fatigue": 1,
            "priority": 4,
            "compound": False
        },

        {
            "name": "Cable Crunch",
            "technical": 2,
            "fatigue": 1,
            "priority": 4,
            "compound": False
        },

        {
            "name": "Unoszenie nóg",
            "technical": 2,
            "fatigue": 2,
            "priority": 4,
            "compound": False
        },

        {
            "name": "Ab Wheel",
            "technical": 3,
            "fatigue": 2,
            "priority": 5,
            "compound": False
        },

        {
            "name": "Russian Twist",
            "technical": 1,
            "fatigue": 1,
            "priority": 2,
            "compound": False
        },

        {
            "name": "Mountain Climbers",
            "technical": 2,
            "fatigue": 2,
            "priority": 2,
            "compound": False
        }

    ]

}