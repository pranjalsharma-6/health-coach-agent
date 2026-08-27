"""A curated exercise table, and what makes a training session coherent.

The counterpart to `ingredients.py`, and it exists for the same reason. The
nutrition side never lets the model choose a number; the training side used to
let it emit "upper body training, 45 min" and call that a plan. A category is
not a prescription — a beginner cannot act on it, and nothing downstream can
check it.

Grounding the trainer in a fixed table does three things:

1. The model picks from movements that exist, at a difficulty the user can
   actually perform, with equipment they plausibly have.
2. Every prescription carries sets, reps, rest and a form cue, so the session
   is followable without already knowing how to train.
3. A session becomes checkable. `find_problems` catches the two failures that
   make a generated workout useless: a session that trains one pattern and
   calls itself full-body, and equipment the user was never asked about.

Rep ranges are the conventional ones for the goal — roughly 8-12 for
hypertrophy, higher for endurance work, lower with longer rest for strength.
They are starting points, not prescriptions for a competitive athlete.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Sequence

from app.models.enums import TrainingStyle


class Pattern(str, Enum):
    """Movement patterns.

    Coarse on purpose. The useful check is not "did this session hit the medial
    deltoid" but "is this session six variations of the same push".
    """

    PUSH = "push"
    PULL = "pull"
    SQUAT = "squat"
    HINGE = "hinge"
    CORE = "core"
    CARRY = "carry"
    CARDIO = "cardio"
    MOBILITY = "mobility"


class Equipment(str, Enum):
    NONE = "none"                # bodyweight, works in a hostel room
    DUMBBELL = "dumbbell"
    BARBELL = "barbell"
    MACHINE = "machine"
    BAND = "band"
    PULL_UP_BAR = "pull_up_bar"


class Level(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass(frozen=True)
class Exercise:
    """One movement, with the cue a beginner needs to do it safely.

    `cue` is the single most common mistake, phrased as an instruction. One
    line, because a paragraph nobody reads is worth nothing.
    """

    name: str
    pattern: Pattern
    equipment: Equipment
    level: Level
    cue: str
    default_sets: int = 3
    default_reps: str = "8-12"
    default_rest_seconds: int = 90


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
EXERCISES: List[Exercise] = [
    # --- Push ---
    Exercise("Push-ups", Pattern.PUSH, Equipment.NONE, Level.BEGINNER,
             "Keep a straight line from head to heels; don't let the hips sag."),
    Exercise("Knee push-ups", Pattern.PUSH, Equipment.NONE, Level.BEGINNER,
             "Same straight line from knees to head — this is a full exercise, not a lesser one.",
             default_reps="10-15"),
    Exercise("Incline push-ups", Pattern.PUSH, Equipment.NONE, Level.BEGINNER,
             "Hands on a bench or table; the higher the surface, the easier it is.",
             default_reps="10-15"),
    Exercise("Dumbbell bench press", Pattern.PUSH, Equipment.DUMBBELL, Level.BEGINNER,
             "Lower until your elbows are level with your ribs, no deeper."),
    Exercise("Dumbbell shoulder press", Pattern.PUSH, Equipment.DUMBBELL, Level.BEGINNER,
             "Press overhead without arching your lower back.", default_reps="8-10"),
    Exercise("Barbell bench press", Pattern.PUSH, Equipment.BARBELL, Level.INTERMEDIATE,
             "Keep your shoulder blades pinned back; use a spotter near failure.",
             default_reps="6-8", default_rest_seconds=120),
    Exercise("Overhead press", Pattern.PUSH, Equipment.BARBELL, Level.INTERMEDIATE,
             "Squeeze your glutes to stop your lower back taking the load.",
             default_reps="6-8", default_rest_seconds=120),
    Exercise("Dips", Pattern.PUSH, Equipment.NONE, Level.ADVANCED,
             "Lean slightly forward; stop when your shoulders reach elbow height.",
             default_reps="6-10"),
    Exercise("Band chest press", Pattern.PUSH, Equipment.BAND, Level.BEGINNER,
             "Control the return — the band pulling you back is half the work.",
             default_reps="12-15"),

    # --- Pull ---
    Exercise("Dumbbell row", Pattern.PULL, Equipment.DUMBBELL, Level.BEGINNER,
             "Pull towards your hip, not your chest, and keep your back flat."),
    Exercise("Inverted row", Pattern.PULL, Equipment.NONE, Level.BEGINNER,
             "Under a sturdy table works. Body straight, chest to the edge.",
             default_reps="8-12"),
    Exercise("Band row", Pattern.PULL, Equipment.BAND, Level.BEGINNER,
             "Drive your elbows back and squeeze your shoulder blades together.",
             default_reps="12-15"),
    Exercise("Lat pulldown", Pattern.PULL, Equipment.MACHINE, Level.BEGINNER,
             "Pull to your collarbone, not behind your neck."),
    Exercise("Pull-ups", Pattern.PULL, Equipment.PULL_UP_BAR, Level.ADVANCED,
             "Start from a dead hang; chin clearly over the bar.",
             default_reps="5-8", default_rest_seconds=120),
    Exercise("Assisted pull-ups", Pattern.PULL, Equipment.PULL_UP_BAR, Level.BEGINNER,
             "Use a band or a chair under one foot. Assistance is not cheating.",
             default_reps="6-10"),
    Exercise("Barbell row", Pattern.PULL, Equipment.BARBELL, Level.INTERMEDIATE,
             "Hinge to about 45 degrees and hold that angle for every rep.",
             default_reps="6-10", default_rest_seconds=120),
    Exercise("Face pulls", Pattern.PULL, Equipment.BAND, Level.BEGINNER,
             "Pull towards your forehead, elbows high. Good for desk-bound shoulders.",
             default_reps="15-20", default_rest_seconds=60),

    # --- Squat ---
    Exercise("Bodyweight squat", Pattern.SQUAT, Equipment.NONE, Level.BEGINNER,
             "Knees track over your toes; go as deep as you can keep your heels down.",
             default_reps="12-20"),
    Exercise("Goblet squat", Pattern.SQUAT, Equipment.DUMBBELL, Level.BEGINNER,
             "Hold the weight at your chest; elbows inside your knees at the bottom.",
             default_reps="10-15"),
    Exercise("Split squat", Pattern.SQUAT, Equipment.NONE, Level.BEGINNER,
             "Front shin vertical; most of the weight through the front heel.",
             default_reps="8-12 each side"),
    Exercise("Walking lunges", Pattern.SQUAT, Equipment.NONE, Level.BEGINNER,
             "Step long enough that your front knee stays over your ankle.",
             default_reps="10-12 each side"),
    Exercise("Bulgarian split squat", Pattern.SQUAT, Equipment.DUMBBELL, Level.INTERMEDIATE,
             "Rear foot elevated behind you; this will humble you, start light.",
             default_reps="8-10 each side"),
    Exercise("Barbell back squat", Pattern.SQUAT, Equipment.BARBELL, Level.INTERMEDIATE,
             "Brace your core before you unrack; chest up throughout.",
             default_reps="5-8", default_rest_seconds=150),
    Exercise("Leg press", Pattern.SQUAT, Equipment.MACHINE, Level.BEGINNER,
             "Don't lock your knees out hard at the top.", default_reps="10-15"),
    Exercise("Step-ups", Pattern.SQUAT, Equipment.NONE, Level.BEGINNER,
             "Drive through the heel of the top foot; don't push off the floor.",
             default_reps="10-12 each side"),

    # --- Hinge ---
    Exercise("Glute bridge", Pattern.HINGE, Equipment.NONE, Level.BEGINNER,
             "Squeeze your glutes at the top; ribs stay down.", default_reps="12-20"),
    Exercise("Hip thrust", Pattern.HINGE, Equipment.DUMBBELL, Level.BEGINNER,
             "Shoulders on a bench, chin tucked. Pause at the top.",
             default_reps="10-15"),
    Exercise("Romanian deadlift", Pattern.HINGE, Equipment.DUMBBELL, Level.BEGINNER,
             "Push your hips back, not down. Stop when you feel your hamstrings.",
             default_reps="8-12"),
    Exercise("Barbell deadlift", Pattern.HINGE, Equipment.BARBELL, Level.INTERMEDIATE,
             "Bar stays against your legs the whole way up. Back flat, never rounded.",
             default_reps="5", default_rest_seconds=180),
    Exercise("Single-leg deadlift", Pattern.HINGE, Equipment.DUMBBELL, Level.INTERMEDIATE,
             "Hips square to the floor — resist rotating open.",
             default_reps="8-10 each side"),
    Exercise("Good mornings", Pattern.HINGE, Equipment.BARBELL, Level.ADVANCED,
             "Light weight. This is a hinge, not a squat.", default_reps="8-12"),
    Exercise("Nordic curl negatives", Pattern.HINGE, Equipment.NONE, Level.ADVANCED,
             "Lower as slowly as you can control; catch yourself with your hands.",
             default_reps="4-6"),

    # --- Core ---
    Exercise("Plank", Pattern.CORE, Equipment.NONE, Level.BEGINNER,
             "Squeeze glutes and brace your stomach. Quality beats duration.",
             default_reps="30-45 seconds", default_rest_seconds=60),
    Exercise("Side plank", Pattern.CORE, Equipment.NONE, Level.BEGINNER,
             "Stack your hips; don't let the bottom one drop.",
             default_reps="20-30 seconds each side", default_rest_seconds=60),
    Exercise("Dead bug", Pattern.CORE, Equipment.NONE, Level.BEGINNER,
             "Keep your lower back flat on the floor the entire time.",
             default_reps="8-10 each side", default_rest_seconds=60),
    Exercise("Bird dog", Pattern.CORE, Equipment.NONE, Level.BEGINNER,
             "Move slowly; don't let your hips rock.",
             default_reps="8-10 each side", default_rest_seconds=60),
    Exercise("Hollow body hold", Pattern.CORE, Equipment.NONE, Level.INTERMEDIATE,
             "Lower back pressed down. Bend your knees if it lifts.",
             default_reps="20-30 seconds", default_rest_seconds=60),
    Exercise("Hanging knee raises", Pattern.CORE, Equipment.PULL_UP_BAR, Level.INTERMEDIATE,
             "Control the way down; don't swing.", default_reps="8-12"),
    Exercise("Russian twists", Pattern.CORE, Equipment.NONE, Level.BEGINNER,
             "Rotate from your ribs, not your arms.", default_reps="12-16 each side",
             default_rest_seconds=60),
    Exercise("Ab wheel rollout", Pattern.CORE, Equipment.NONE, Level.ADVANCED,
             "Only go as far as you can keep your back from arching.",
             default_reps="6-10"),

    # --- Carry ---
    Exercise("Farmer's carry", Pattern.CARRY, Equipment.DUMBBELL, Level.BEGINNER,
             "Stand tall, shoulders back, walk normally.",
             default_reps="30-40 metres", default_rest_seconds=60),
    Exercise("Suitcase carry", Pattern.CARRY, Equipment.DUMBBELL, Level.INTERMEDIATE,
             "Weight in one hand; resist leaning towards it.",
             default_reps="20-30 metres each side", default_rest_seconds=60),

    # --- Cardio ---
    Exercise("Brisk walk", Pattern.CARDIO, Equipment.NONE, Level.BEGINNER,
             "Fast enough to breathe harder, slow enough to hold a conversation.",
             default_sets=1, default_reps="30-45 minutes", default_rest_seconds=0),
    Exercise("Easy jog", Pattern.CARDIO, Equipment.NONE, Level.BEGINNER,
             "Conversational pace. If you can't talk, slow down.",
             default_sets=1, default_reps="20-30 minutes", default_rest_seconds=0),
    Exercise("Cycling", Pattern.CARDIO, Equipment.NONE, Level.BEGINNER,
             "Steady effort; saddle high enough for a slight knee bend at the bottom.",
             default_sets=1, default_reps="30-45 minutes", default_rest_seconds=0),
    Exercise("Skipping", Pattern.CARDIO, Equipment.NONE, Level.BEGINNER,
             "Small jumps, land softly through the balls of your feet.",
             default_sets=4, default_reps="60 seconds", default_rest_seconds=60),
    Exercise("Stair climbing", Pattern.CARDIO, Equipment.NONE, Level.BEGINNER,
             "Whole foot on each step; use the rail on the way down.",
             default_sets=1, default_reps="15-20 minutes", default_rest_seconds=0),
    Exercise("Interval sprints", Pattern.CARDIO, Equipment.NONE, Level.ADVANCED,
             "Warm up properly first. Walk the recovery, don't jog it.",
             default_sets=6, default_reps="30 seconds hard", default_rest_seconds=90),

    # --- Mobility ---
    Exercise("Cat-cow", Pattern.MOBILITY, Equipment.NONE, Level.BEGINNER,
             "Move with your breath; no forcing.", default_sets=1,
             default_reps="8-10", default_rest_seconds=0),
    Exercise("World's greatest stretch", Pattern.MOBILITY, Equipment.NONE, Level.BEGINNER,
             "Lunge, hand inside the front foot, rotate open.", default_sets=1,
             default_reps="5 each side", default_rest_seconds=0),
    Exercise("Hip flexor stretch", Pattern.MOBILITY, Equipment.NONE, Level.BEGINNER,
             "Tuck your pelvis under before you lean forward.", default_sets=1,
             default_reps="30 seconds each side", default_rest_seconds=0),
    Exercise("Thoracic rotations", Pattern.MOBILITY, Equipment.NONE, Level.BEGINNER,
             "Rotate from the upper back, keeping your hips still.", default_sets=1,
             default_reps="8 each side", default_rest_seconds=0),
    Exercise("Sun salutations", Pattern.MOBILITY, Equipment.NONE, Level.BEGINNER,
             "One breath per movement; don't rush the transitions.", default_sets=3,
             default_reps="5 rounds", default_rest_seconds=30),
    Exercise("Downward dog hold", Pattern.MOBILITY, Equipment.NONE, Level.BEGINNER,
             "Bend your knees freely — a long spine matters more than straight legs.",
             default_sets=3, default_reps="30 seconds", default_rest_seconds=30),
    Exercise("Pigeon pose", Pattern.MOBILITY, Equipment.NONE, Level.INTERMEDIATE,
             "Keep your hips square; prop the near hip on a cushion if it lifts.",
             default_sets=1, default_reps="60 seconds each side", default_rest_seconds=0),

    # --- Swimming ---
    Exercise("Easy swim — freestyle", Pattern.CARDIO, Equipment.NONE, Level.BEGINNER,
             "Breathe every third stroke to keep your stroke even on both sides.",
             default_sets=1, default_reps="20-30 minutes", default_rest_seconds=0),
    Exercise("Swim intervals", Pattern.CARDIO, Equipment.NONE, Level.INTERMEDIATE,
             "Hold the same pace across every rep; start slower than feels right.",
             default_sets=8, default_reps="50 metres", default_rest_seconds=45),
    Exercise("Kickboard drill", Pattern.CARDIO, Equipment.NONE, Level.BEGINNER,
             "Kick from the hip, not the knee, with a small steady flutter.",
             default_sets=4, default_reps="50 metres", default_rest_seconds=45),
    Exercise("Pull buoy drill", Pattern.CARDIO, Equipment.NONE, Level.INTERMEDIATE,
             "Buoy between the thighs so your legs rest and your arms do the work.",
             default_sets=4, default_reps="50 metres", default_rest_seconds=45),
]

BY_NAME: Dict[str, Exercise] = {e.name.lower(): e for e in EXERCISES}

# Exercises whose style cannot be read off equipment alone. A jog and a swim
# both need no equipment, but someone who chose "swimming" did not ask to run.
_STYLE_OVERRIDES: Dict[str, FrozenSet[TrainingStyle]] = {
    "easy jog": frozenset({TrainingStyle.RUNNING_CYCLING}),
    "cycling": frozenset({TrainingStyle.RUNNING_CYCLING}),
    "interval sprints": frozenset({TrainingStyle.RUNNING_CYCLING}),
    "easy swim — freestyle": frozenset({TrainingStyle.SWIMMING}),
    "swim intervals": frozenset({TrainingStyle.SWIMMING}),
    "kickboard drill": frozenset({TrainingStyle.SWIMMING}),
    "pull buoy drill": frozenset({TrainingStyle.SWIMMING}),
}

# What a gym gives you that a bedroom does not.
_GYM_ONLY = {Equipment.BARBELL, Equipment.MACHINE, Equipment.PULL_UP_BAR}

_HOME_STRENGTH = frozenset(
    {TrainingStyle.BODYWEIGHT, TrainingStyle.DUMBBELLS, TrainingStyle.FULL_GYM}
)


def styles_of(exercise: Exercise) -> FrozenSet[TrainingStyle]:
    """Which training styles this exercise belongs to.

    Derived from pattern and equipment, with overrides where the two do not
    settle it. A gym membership is a superset of a bedroom: somebody with a
    rack can still do push-ups, so bodyweight work stays available to them.
    """
    override = _STYLE_OVERRIDES.get(exercise.name.lower())
    if override is not None:
        return override

    if exercise.pattern is Pattern.MOBILITY:
        # Mobility is universal — everyone benefits, nobody needs anything.
        return frozenset(TrainingStyle)

    if exercise.equipment in _GYM_ONLY:
        return frozenset({TrainingStyle.FULL_GYM})

    if exercise.equipment is Equipment.DUMBBELL:
        return frozenset({TrainingStyle.DUMBBELLS, TrainingStyle.FULL_GYM})

    if exercise.equipment is Equipment.BAND:
        return _HOME_STRENGTH

    if exercise.pattern is Pattern.CARDIO:
        # Walking, skipping, stairs: no facility required, so nobody is
        # excluded from them whatever else they picked.
        return frozenset(TrainingStyle)

    return _HOME_STRENGTH


DEFAULT_STYLES: List[TrainingStyle] = [TrainingStyle.BODYWEIGHT]

STRENGTH_PATTERNS = {
    Pattern.PUSH,
    Pattern.PULL,
    Pattern.SQUAT,
    Pattern.HINGE,
    Pattern.CORE,
    Pattern.CARRY,
}


def find(name: str) -> Optional[Exercise]:
    """Look up by name, case-insensitively. None when it is not in the table."""
    return BY_NAME.get(name.strip().lower())


def allowed_for(
    level: Level, styles: Optional[Sequence[TrainingStyle]] = None
) -> List[Exercise]:
    """Exercises this user can do, at this level, in the styles they chose.

    Levels are cumulative: an intermediate keeps every beginner movement. The
    basics do not stop working because someone got stronger.
    """
    ranks = {Level.BEGINNER: 0, Level.INTERMEDIATE: 1, Level.ADVANCED: 2}
    ceiling = ranks[level]
    chosen = {TrainingStyle(s) for s in (styles or DEFAULT_STYLES)}

    return [
        e for e in EXERCISES
        if ranks[e.level] <= ceiling and (styles_of(e) & chosen)
    ]


def find_problems(
    exercise_names: Sequence[str],
    level: Level,
    styles: Optional[Sequence[TrainingStyle]] = None,
) -> List[str]:
    """What is wrong with this session, if anything.

    Deliberately narrow. It checks the things that make a generated workout
    unusable rather than merely suboptimal — programming taste is the model's
    to argue about, safety and feasibility are not.
    """
    problems: List[str] = []
    chosen = {TrainingStyle(s) for s in (styles or DEFAULT_STYLES)}
    ranks = {Level.BEGINNER: 0, Level.INTERMEDIATE: 1, Level.ADVANCED: 2}

    known: List[Exercise] = []
    for name in exercise_names:
        exercise = find(name)
        if exercise is None:
            problems.append(
                f"'{name}' is not in the exercise table — use one that is, so "
                "the form cue and the difficulty are known."
            )
            continue
        known.append(exercise)

        exercise_styles = styles_of(exercise)
        if not (exercise_styles & chosen):
            wanted = ", ".join(sorted(s.value for s in exercise_styles))
            problems.append(
                f"'{exercise.name}' belongs to {wanted}, which the user did not "
                "choose."
            )

        if ranks[exercise.level] > ranks[level]:
            problems.append(
                f"'{exercise.name}' is {exercise.level.value}; the user is "
                f"{level.value}."
            )

    strength = [e for e in known if e.pattern in STRENGTH_PATTERNS]
    if len(strength) >= 3:
        patterns = {e.pattern for e in strength}
        if len(patterns) == 1:
            only = next(iter(patterns)).value
            problems.append(
                f"Every movement in this session is a {only}. A session of "
                "three or more strength exercises should train more than one "
                "pattern."
            )

    return problems


def cue_for(name: str) -> Optional[str]:
    """The table's form cue for an exercise, if it is one we know."""
    exercise = find(name)
    return exercise.cue if exercise else None
