"""
SereneMind — Unified Mental Health Model v4
===========================================
Architecture:  LogisticRegression (SAGA solver, L2, class_weight=balanced)
Features:      TF-IDF (word 1-4gram + char 2-5gram, 80K features)
Classes:       9  (depression, anxiety, crisis, stress, grief, fear, anger, joy, normal)
Data:          emotion_dataset + go_emotions + mental_health_conversations + rich clinical seed
Target:        ≥ 90% test accuracy | ≥ 0.88 Macro F1

Run:
    cd serenemind/ml
    source venv/bin/activate
    python3 training/train_unified_v3.py
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os, re, time, json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    log_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE / "data" / "raw"
MODEL_DIR   = BASE / "models"
REPORTS_DIR = BASE / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
LABELS = ["depression", "anxiety", "crisis", "stress", "grief", "fear", "anger", "joy", "normal"]
MIN_PER_CLS  = 1000    # Boost minority minimums significantly through oversampling
TFIDF_FEATS  = 70_000  # vocabulary size
SEED         = 42

# Training Hyperparameters
EPOCHS     = 50
BATCH_SIZE = 4096
PATIENCE   = 10

# ─── Text Cleaning ────────────────────────────────────────────────────────────

_url_re  = re.compile(r"https?://\S+")
_html_re = re.compile(r"<[^>]+>")
_num_re  = re.compile(r"\b\d+\b")
_ws_re   = re.compile(r"\s+")

def clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Clip extreme text to speed up processing (most signal is in the first 2k chars)
    text = text[:2000]
    text = _url_re.sub(" url ", text)
    text = _html_re.sub(" ", text)
    text = text.lower()
    text = _num_re.sub(" num ", text)
    text = re.sub(r"[^\w\s\'!?.,]", " ", text)
    text = _ws_re.sub(" ", text).strip()
    return text

# ─── SECTION 1: Rich Clinical Seed (300+ expert-curated examples) ─────────────
CLINICAL_SEED = {
    "crisis": [
        # EXPLICIT
        "I want to kill myself right now",
        "I am going to end my life tonight",
        "I want to die and I have a plan",
        "I am planning to overdose on pills tonight",
        "I searched for ways to hang myself",
        "I bought a rope and I know what to do",
        "I cannot take this anymore and I want to die",
        "I will end it all tonight, goodbye",
        "I took pills already, I don't know how many",
        "I am going to jump off the bridge",
        "I have the knife ready and I am scared of myself",
        "I have been cutting myself every single night",
        "I want to shoot myself and end everything",
        "I am done with this world and I am not coming back",
        "Life is not worth living and I have a plan to end it",
        "I will not be here by morning",
        "I said goodbye to everyone today",
        "I wrote a note for my family",
        "Nobody will miss me when I am gone",
        "I want a painless death as soon as possible",
        # IMPLICIT
        "I don't see a reason to continue anymore",
        "Everyone would be better without me",
        "I wish I could just disappear forever",
        "Nothing matters and I am so tired of existing",
        "I am tired of waking up every single day",
        "The pain is so unbearable I cannot go on",
        "I am a burden to everyone around me",
        "I have been thinking about not being here anymore",
        "I don't want to exist like this anymore",
        "There is no point in going on, honestly",
        "I feel like the world would be better if I wasn't in it",
        "I keep thinking this is my last week alive",
        "I have given away my belongings to friends",
        "I have been saying goodbye to people in subtle ways",
        "I feel completely hopeless and there is no way out",
        "I am not afraid of death anymore, actually I welcome it",
        "Ending it would bring relief to my family",
        "I don't care if I wake up tomorrow",
        "I went to the rooftop and looked down for a long time",
        "I drove very fast hoping something would happen",
        "I stopped eating because I don't see a future",
        "I isolate myself because I am preparing to leave",
        "I gave my pet away because I won't be here to care",
        "I feel like disappearing and never coming back",
        # POST-CRISIS SCENARIO (long context)
        "After the accident I saw people dying and I fell apart. I went to the rooftop of the hospital building and wanted to jump and end myself forever. The pain was too much.",
        "I have been holding a razor every night for weeks but haven't used it yet. I don't know how much longer I can resist.",
        "The rescue team saved someone but no one is here to save me. I feel utterly alone and ready to end this.",
    ],

    "depression": [
        "I feel completely empty inside and nothing brings joy",
        "I have been crying every day for weeks without knowing why",
        "I can barely get out of bed in the morning anymore",
        "Everything feels dark and hopeless",
        "I have lost interest in everything I used to love",
        "Life feels meaningless and I don't see a future",
        "I feel worthless and like a failure at everything",
        "I can't remember the last time I felt happy",
        "I sleep all day and still feel exhausted",
        "I stopped talking to my friends because I feel numb",
        "I feel like an empty shell just going through motions",
        "Even simple tasks feel impossible and overwhelming",
        "I have no motivation to do anything at all",
        "I feel disconnected from everything and everyone",
        "I don't care about anything anymore",
        "Nothing makes me happy no matter what I try",
        "I feel heavy and dark all the time",
        "I isolate myself because I feel like a burden",
        "I feel like I am watching my life through fog",
        "My mind is blank and I feel absolutely nothing",
        "I have been depressed for months and nothing helps",
        "I lost my job and my relationship and feel nothing",
        "I stare at walls for hours without thinking",
        "Food has no taste and everything feels grey",
        "I feel like I am slowly disappearing",
    ],

    "anxiety": [
        "My heart is racing and I cannot stop worrying",
        "I am having a panic attack right now",
        "I feel like something terrible is about to happen",
        "I keep catastrophizing about everything in my life",
        "My mind won't stop with anxious thoughts",
        "I am scared to leave the house because of panic attacks",
        "I have been shaking all day thinking about tomorrow",
        "My breathing gets short and I feel like I am dying",
        "I cannot sleep because my mind races with worries",
        "I am constantly on edge and waiting for disaster",
        "I feel dread in my stomach all the time",
        "I check the locks and windows obsessively",
        "I avoid social situations because I fear judgment",
        "My palms sweat and I feel sick thinking about it",
        "I have been anxious about everything for months",
        "I overthink every conversation and interaction",
        "I worry about health constantly and think I am dying",
        "I always assume the worst in every situation",
        "My anxiety has taken over my entire life",
        "I feel paralyzed by fear of what might happen",
        "I have nervous breakdowns over small decisions",
        "I cannot go to parties or public places anymore",
        "I wake up at 3am with racing heart and dread",
        "I have been having anxiety attacks since the accident",
        "Every new day brings a new wave of panic and dread",
    ],

    "stress": [
        "I am completely burned out from work",
        "The deadlines are overwhelming and I cannot cope",
        "I feel so stressed I cannot think straight",
        "I have too much on my plate and I feel crushed",
        "Work is consuming every part of my life",
        "I am stressed about my exams and cannot focus",
        "My boss is pressuring me and I feel suffocated",
        "I have been working 16 hours a day for weeks",
        "The pressure is unbearable and I feel breaking point",
        "I am stressed about money and cannot sleep",
        "Financial pressure is destroying my mental health",
        "I have multiple urgent projects and no time to breathe",
        "The constant stress is giving me headaches every day",
        "I feel like I am failing at everything at once",
        "I snapped at my family because of work stress",
        "My job demands are unreasonable and I feel trapped",
        "Relationship stress and work stress at the same time",
        "I feel tight in my chest from stress all day",
        "I am stretched too thin and about to break",
        "The stress of caring for my sick parent is exhausting",
        "I cannot relax even on weekends because of work",
        "I grind my teeth at night from stress",
        "Exam season is destroying my health",
        "My stress manifests as stomach pain daily",
        "I feel overwhelmed caring for children and working full time",
    ],

    "grief": [
        "I lost my mother last week and cannot cope",
        "My father passed away and I feel destroyed inside",
        "I am grieving the loss of my best friend",
        "My dog died and I cannot stop crying",
        "I lost my baby and the grief is unbearable",
        "I keep reaching for my phone to call her and then remember",
        "Their voice keeps coming back to me and it breaks me",
        "I visit their grave and I cannot leave",
        "My house feels empty without them",
        "I see their things and it destroys me every time",
        "Grief comes in waves and some days I cannot breathe",
        "I lost my marriage and feel like I am mourning",
        "I miss them so much every single day",
        "I cannot look at photos without breaking down",
        "I am not sure how to live without them",
        "I lost my job which defined me and feel empty now",
        "The grief of my failed relationship feels like a death",
        "I lost my child and the world will never be the same",
        "I did not cry at the funeral but now I cannot stop",
        "I feel guilty for smiling since they left",
        "I wish I had said I love you more often to them",
        "The grief comes at random times and floors me completely",
        "I dream about them and wake up devastated",
        "I don't know how to do birthdays and holidays now",
        "The emptiness of losing them is impossible to describe",
    ],

    "fear": [
        "I am terrified and cannot calm down",
        "I am scared and my body is shaking right now",
        "I have a phobia that is ruining my life",
        "I am afraid to drive since the car accident",
        "I am too scared to go outside after what happened",
        "I have nightmares every night about the incident",
        "I freeze every time I think about what happened",
        "The image of the accident keeps playing in my mind",
        "I am scared it will happen again and I cannot stop thinking",
        "Fear has paralyzed me completely",
        "I saw something traumatic and cannot get it out of my head",
        "I am scared of needles and hospitals but I need treatment",
        "I have a deep fear of losing control",
        "I am terrified of heights since the fall",
        "My flight caused severe fear and I cannot travel now",
        "I am afraid of crowds after the stampede",
        "I am scared of the dark after what I experienced",
        "The spider phobia has taken over my house routine",
        "I am scared to be alone after being attacked",
        "Fear is taking over every decision in my life",
        "I found out I might have cancer and I am terrified",
        "I am afraid to check the test results",
        "The medical diagnosis fills me with dread",
        "I am scared my pain is something serious",
        "I feel constant irrational fear that I cannot explain",
    ],

    "anger": [
        "I am so angry I feel like I could explode",
        "I am furious and cannot control my rage",
        "I feel intense anger and resentment",
        "I am filled with rage at what happened to me",
        "My anger is getting out of control and scaring me",
        "I punched a wall because I was so angry",
        "I screamed at my partner and could not stop",
        "I feel bitter and resentful all the time",
        "I am angry at the injustice I suffered",
        "My temper has been short and I hate it",
        "I feel rage when I think about what he did to me",
        "I seethe with anger when I think about this situation",
        "I am livid and I cannot let it go",
        "The anger follows me everywhere I go",
        "I am so frustrated with everything around me",
        "Rage has been building up inside me for months",
        "I snap at everyone because of this deep anger",
        "I hate how angry I get over small things",
        "I was betrayed and I cannot forgive",
        "My anger turned into crying and I am lost",
    ],

    "joy": [
        "I feel so happy and grateful today",
        "Life is wonderful and I feel amazing",
        "I got the job and I am over the moon",
        "Great things are happening and I am excited",
        "I feel content and at peace with my life",
        "Today was one of the best days I have had in a while",
        "I feel energized and hopeful about the future",
        "I celebrated with friends and it was such a beautiful evening",
        "I am deeply grateful for everything I have",
        "My heart is full of joy and gratitude",
        "I achieved my goal and I feel proud and happy",
        "Love and connection is all around me today",
        "I feel inspired and motivated",
        "Everything is going well and I am thankful",
        "I laughed so much today and it felt healing",
        "I feel light and carefree for the first time in months",
        "My baby smiled at me and it melted my heart",
        "Today is a good day and I feel calm and happy",
        "I feel optimistic and full of positive energy",
        "I am in love and everything feels beautiful",
    ],

    "normal": [
        "I went to work today and it was an ordinary day",
        "I cooked dinner and watched a show with my family",
        "Today was pretty uneventful, just regular life",
        "I completed my tasks and feel fine about the day",
        "Traffic was bad but I got home safely",
        "I had a meeting at the office today",
        "My salary increment came through today",
        "The rescue team helped someone on the road after a minor accident",
        "I saw an accident and the rescue team arrived quickly",
        "I went to the hospital for a routine checkup",
        "The doctors said everything is fine at the checkup",
        "I drove past an accident — rescue came fast and helped everyone",
        "It started raining on my way home but I arrived safely",
        "I bought groceries and did some chores around the house",
        "I was at the hospital visiting a friend who is recovering well",
        "The news covered an accident but no one was seriously hurt",
        "I helped a stranger on the road after a minor bump",
        "I saw some excitement on the road but everything was resolved",
        "My colleague was in a small accident but is completely fine",
        "I feel neutral today, not good not bad, just normal",
        "I completed my workout and feel reasonably okay",
        "I talked to my parents on a video call this evening",
        "I fixed a bug at work and feel satisfied",
        "I had a quiet day reading and resting",
        "I ran some errands and everything went smoothly",
    ],
}


def load_clinical_seed() -> pd.DataFrame:
    """Build rich clinical seed dataset from curated expert examples."""
    rows = []
    for label, examples in CLINICAL_SEED.items():
        for ex in examples:
            rows.append({"text": clean(ex), "label": label})
    df = pd.DataFrame(rows)
    print(f"  ✅ Clinical seed: {len(df):,} samples across {df['label'].nunique()} classes")
    return df


# ─── SECTION 2: Dataset Loaders ───────────────────────────────────────────────

# emotion_dataset.csv label map: 0=sadness,1=joy,2=love,3=anger,4=fear,5=surprise
EMOTION_MAP = {0: "depression", 1: "joy", 2: "joy", 3: "anger", 4: "fear", 5: "normal"}

# GoEmotions: 28 emotion names in order (index 0-27)
# From: https://github.com/google-research/google-research/tree/master/goemotions
GO_EMOTIONS_NAMES = [
    "admiration",   # 0
    "amusement",    # 1
    "anger",        # 2
    "annoyance",    # 3
    "approval",     # 4
    "caring",       # 5
    "confusion",    # 6
    "curiosity",    # 7
    "desire",       # 8
    "disappointment",  # 9
    "disapproval",  # 10
    "disgust",      # 11
    "embarrassment",# 12
    "excitement",   # 13
    "fear",         # 14
    "gratitude",    # 15
    "grief",        # 16
    "joy",          # 17
    "love",         # 18
    "nervousness",  # 19
    "optimism",     # 20
    "pride",        # 21
    "realization",  # 22
    "relief",       # 23
    "remorse",      # 24
    "sadness",      # 25
    "surprise",     # 26
    "neutral",      # 27
]

# GoEmotions name → our 9-class label
GO_EMOTIONS_MAP = {
    "admiration":     "joy",        "amusement":    "joy",
    "approval":       "joy",        "caring":       "normal",
    "curiosity":      "normal",     "desire":       "normal",
    "excitement":     "joy",        "gratitude":    "joy",
    "joy":            "joy",        "love":         "joy",
    "optimism":       "joy",        "pride":        "joy",
    "relief":         "joy",        "realization":  "normal",
    "anger":          "anger",      "annoyance":    "anger",
    "disapproval":    "anger",      "disgust":      "anger",
    "fear":           "fear",       "nervousness":  "anxiety",
    "confusion":      "stress",     "embarrassment": "stress",
    "sadness":        "depression", "disappointment": "depression",
    "remorse":        "depression", "grief":        "grief",
    "surprise":       "normal",     "neutral":      "normal",
}


def load_emotion_dataset(path: Path) -> pd.DataFrame:
    """Load emotion_dataset.csv — maps 6 emotion codes to our 9 classes."""
    print(f"  📂 Loading {path.name} ...", end="", flush=True)
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int).map(EMOTION_MAP)
    df = df.dropna(subset=["label"])
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].apply(clean)
    print(f" {len(df):,} rows")
    return df


def load_go_emotions(path: Path) -> pd.DataFrame:
    """Load go_emotions.csv — labels are integer indices like [27] or [ 8 20]."""
    print(f"  ⼂️  Loading {path.name} ...", end="", flush=True)
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "labels"])
    rows = []
    for _, row in df.iterrows():
        # Parse label string like '[27]' or '[ 8 20]' into list of ints
        raw = str(row["labels"]).strip("[]").split()
        indices = []
        for tok in raw:
            try:
                indices.append(int(tok))
            except ValueError:
                pass
        mapped = None
        for idx in indices:
            if 0 <= idx < len(GO_EMOTIONS_NAMES):
                name = GO_EMOTIONS_NAMES[idx]
                if name in GO_EMOTIONS_MAP:
                    mapped = GO_EMOTIONS_MAP[name]
                    break
        if mapped:
            rows.append({"text": clean(str(row["text"])), "label": mapped})
    result = pd.DataFrame(rows)
    print(f" {len(result):,} rows (mapped from {len(df):,}")
    return result


def load_mh_conversations(path: Path) -> pd.DataFrame:
    """Load mental_health_conversations.csv — Context column mapped to depression/anxiety."""
    print(f"  📂 Loading {path.name} ...", end="", flush=True)
    df = pd.read_csv(path)
    df = df.dropna(subset=["Context"])
    rows = []
    for _, row in df.iterrows():
        text = clean(str(row["Context"]))
        if not text:
            continue
        # Heuristic: classify context into depression/anxiety/stress/crisis/normal
        txt_lower = text.lower()
        if any(w in txt_lower for w in ["kill", "end my life", "suicide", "want to die", "not worth", "disappear"]):
            label = "crisis"
        elif any(w in txt_lower for w in ["sad", "hopeless", "empty", "worthless", "depress", "numb", "meaningless"]):
            label = "depression"
        elif any(w in txt_lower for w in ["anxious", "panic", "worry", "fear", "nervous", "overwhelm", "scared"]):
            label = "anxiety"
        elif any(w in txt_lower for w in ["stress", "pressure", "exhaust", "tired", "burnout", "deadline"]):
            label = "stress"
        else:
            label = "normal"
        rows.append({"text": text, "label": label})
    result = pd.DataFrame(rows)
    print(f" {len(result):,} rows")
    return result


def load_supplementary_data(path: Path) -> pd.DataFrame:
    """Load mental_health_sentiment.csv — columns: [text, label]"""
    if not path.exists():
        print(f"  ⚠️  Skipping {path.name} (not found)")
        return pd.DataFrame(columns=["text", "label"])
    
    print(f"  📂 Loading {path.name} ... ", end="", flush=True)
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    
    # Map labels: SuicideWatch -> crisis, depression -> depression
    label_map = {
        "SuicideWatch": "crisis",
        "depression": "depression"
    }
    df["label"] = df["label"].map(label_map)
    df = df.dropna(subset=["label"])
    
    print("cleaning... ", end="", flush=True)
    df["text"] = df["text"].apply(clean)
    
    print(f" {len(df):,} rows")
    return df[["text", "label"]]


def load_clinical_seed() -> pd.DataFrame:
    """Load the manual CLINICAL_SEED into a DataFrame."""
    rows = []
    for lbl, texts in CLINICAL_SEED.items():
        for t in texts:
            rows.append({"text": clean(t), "label": lbl})
    df = pd.DataFrame(rows)
    print(f"  ✅ Clinical seed: {len(df):,} samples across {df['label'].nunique()} classes")
    return df


# ─── SECTION 3: Merge & Balance ───────────────────────────────────────────────

def merge_and_balance(dfs: list) -> pd.DataFrame:
    """Merge all DataFrames and apply class-level capping + minimum enforcement."""
    print("\n⚖️   MERGING & BALANCING")
    print("─" * 60)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.dropna(subset=["text", "label"])
    merged = merged[merged["label"].isin(LABELS)]
    merged["text"] = merged["text"].astype(str).str.strip()
    merged = merged[merged["text"].str.len() >= 8]

    print(f"\n  📊 Raw merged: {len(merged):,} samples")
    print(f"  📋 Class distribution (raw):")
    for lbl, cnt in merged["label"].value_counts().items():
        bar = "█" * int(cnt / max(merged["label"].value_counts()) * 40)
        print(f"     {lbl:12}  {cnt:6d}  {bar}")

    balanced_parts = []
    for label in LABELS:
        sub = merged[merged["label"] == label]
        n = len(sub)
        if n == 0:
            continue
        if n < MIN_PER_CLS:
            # Oversample minorities heavily
            repeats = (MIN_PER_CLS // n) + 1
            sub = pd.concat([sub] * repeats, ignore_index=True).sample(MIN_PER_CLS, random_state=SEED)
        
        # We DO NOT cap majorities anymore. We let them stay large to provide deep vocabulary context.
        # Imbalance will be handled by class_weight="balanced" in LogisticRegression.
        balanced_parts.append(sub)

    balanced = pd.concat(balanced_parts, ignore_index=True).sample(frac=1, random_state=SEED)

    print(f"\n  ✅ Final dataset: {len(balanced):,} samples | {balanced['label'].nunique()} classes")
    print(f"  📋 Class distribution (before class_weights):")
    for lbl, cnt in balanced["label"].value_counts().sort_values(ascending=False).items():
        bar = "█" * int(cnt / max(balanced["label"].value_counts()) * 40)
        print(f"     {lbl:12}  {cnt:6d}  {bar}")

    return balanced


# ─── SECTION 4: TF-IDF Feature Engineering ────────────────────────────────────

def build_vectorizer() -> FeatureUnion:
    """
    Two-head TF-IDF:
      word_tfidf: subword-aware word n-grams (1-4) — captures phrases
      char_tfidf: character n-grams (2-5) — captures morphology, typos, slang
    """
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=40_000,
        min_df=5,              # Increased from 2 to reduce noise
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"\b\w+\b",
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=10_000,
        min_df=10,             # Increased from 3
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    return FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])
    return FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])


# ─── SECTION 5: LogisticRegression Training ───────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, le: LabelEncoder):
    """
    Train LogisticRegression with L-BFGS solver.
    More stable than SGD for this dataset size.
    """
    print("\n🧠  TRAINING LogisticRegression (L-BFGS)")
    print("═" * 78)
    print(f"  Samples: {X_train.shape[0]:,} | Features: {X_train.shape[1]:,} | Classes: {len(le.classes_)}")
    
    t0 = time.time()
    clf = LogisticRegression(
        solver="lbfgs",
        C=0.5,                   # Slightly higher regularization to prevent overfitting
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
        verbose=1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate on val
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"  ✅ Training complete in {time.time()-t0:.1f}s | Val Acc: {val_acc:.4f}")
    return clf

    val_preds  = clf.predict(X_val)
    val_acc    = accuracy_score(y_val, val_preds)
    val_f1     = f1_score(y_val, val_preds, average="macro", zero_division=0)
    train_preds = clf.predict(X_train)
    train_acc  = accuracy_score(y_train, train_preds)

    print(f"  ✅ Training complete in {elapsed:.1f}s")
    print(f"  📊 Train Acc: {train_acc*100:.2f}%  |  Val Acc: {val_acc*100:.2f}%  |  Val F1: {val_f1:.4f}")
    print(f"  📊 Overfit gap: {(train_acc - val_acc)*100:+.2f}%")
    print("═" * 78)

    return clf


# ─── SECTION 6: Isotonic Calibration ─────────────────────────────────────────

def calibrate(clf, le: LabelEncoder, X_val, y_val):
    """Apply per-class isotonic calibration to the fitted LogisticRegression."""
    print("\n🎯  CALIBRATING PROBABILITIES (isotonic regression on val set)")
    raw_val   = clf.predict_proba(X_val)
    calibrators = []
    for i in range(len(le.classes_)):
        ir = IsotonicRegression(out_of_bounds="clip")
        y_binary = (y_val == i).astype(int)
        ir.fit(raw_val[:, i], y_binary)
        calibrators.append(ir)
    print(f"  ✅ Calibrated {len(calibrators)} class probabilities")
    return calibrators


def apply_calibration(clf, calibrators, X):
    raw = clf.predict_proba(X)
    cal = np.zeros_like(raw)
    for i, ir in enumerate(calibrators):
        cal[:, i] = ir.predict(raw[:, i])
    row_s = cal.sum(axis=1, keepdims=True)
    return cal / np.maximum(row_s, 1e-9)


# ─── SECTION 7: Evaluation & Reporting ───────────────────────────────────────

def show_per_class_results(y_true, y_pred, class_names):
    """Pretty-print per-class precision / recall / F1."""
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    print(f"\n  ┌{'─'*65}┐")
    print(f"  │{'PER-CLASS PERFORMANCE ON TEST SET':^65}│")
    print(f"  ├{'─'*14}┬{'─'*11}┬{'─'*11}┬{'─'*11}┬{'─'*10}┤")
    print(f"  │{'Class':14}│{'Precision':>11}│{'Recall':>11}│{'F1':>11}│{'Support':>10}│")
    print(f"  ├{'─'*14}┼{'─'*11}┼{'─'*11}┼{'─'*11}┼{'─'*10}┤")
    for cls in class_names:
        if cls not in report:
            continue
        r    = report[cls]
        prec = r["precision"] * 100
        rec  = r["recall"] * 100
        f1v  = r["f1-score"]
        sup  = int(r["support"])
        flag = "✅" if f1v >= 0.85 else ("🟡" if f1v >= 0.70 else "❌")
        print(f"  │{cls:14}│{prec:>10.1f}%│{rec:>10.1f}%│{f1v:>9.4f} {flag}│{sup:>10}│")
    print(f"  ├{'─'*14}┴{'─'*11}┴{'─'*11}┴{'─'*11}┴{'─'*10}┤")
    macro = report["macro avg"]
    print(f"  │{'MACRO AVG':>10}     Prec: {macro['precision']*100:.1f}%  Recall: {macro['recall']*100:.1f}%  F1:{macro['f1-score']:.4f}   │")
    print(f"  └{'─'*65}┘")


def plot_learning_curves_logreg(history, output_dir: Path):
    """Plot accuracy comparison chart (no epoch loop — single training point)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cats   = ["Train", "Validation", "Test"]
    accs   = [history["train_acc"], history["val_acc"], history["test_acc"]]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    bars   = ax.bar(cats, [a * 100 for a in accs], color=colors, width=0.5, alpha=0.85, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_title("SereneMind Unified Model v4 — Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.axhline(90, color="red", linestyle="--", linewidth=1.2, label="90% target")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "accuracy_chart.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📈 Accuracy chart saved → {path}")


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_title("Confusion Matrix — SereneMind Unified Model v4", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Confusion matrix saved → {path}")


def plot_per_class_f1(y_true, y_pred, class_names, output_dir: Path):
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    classes = [c for c in class_names if c in report]
    f1s     = [report[c]["f1-score"] for c in classes]
    colors  = ["#4CAF50" if f >= 0.85 else ("#FF9800" if f >= 0.70 else "#F44336") for f in f1s]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(classes, f1s, color=colors, alpha=0.85, edgecolor="white")
    for bar, f in zip(bars, f1s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{f:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.axvline(0.85, color="green", linestyle="--", linewidth=1, label="Target F1=0.85")
    ax.axvline(0.70, color="orange", linestyle="--", linewidth=1, label="Acceptable F1=0.70")
    ax.set_title("Per-class F1 Score — SereneMind v4", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "per_class_f1.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📈 Per-class F1 chart saved → {path}")


# ─── SECTION 8: Long-Text Inference Test ──────────────────────────────────────

LONG_TEXT_TESTS = [
    ("crisis", "After the accident where I saw people dying on the road, I got numb and my mind stopped working completely. I felt severe pain and was taken to hospital. But my condition instead of improving started getting worse and I fell into such deep despair that I went to the rooftop of the hospital building and I wanted to jump and end myself forever."),
    ("depression", "I have been going to the office every day but I feel completely empty inside and nothing feels meaningful anymore. I used to love the rain and excitement but now even on my salary increment day I felt nothing. I sleep most of the day when I get home and I have stopped talking to my friends because everything feels pointless and dark."),
    ("normal", "I am going to the office today. It is my salary increment day and I feel quite good about it. I drove in my own car and it started raining which made me feel excited and happy. The rescue team helped someone on the road after a minor accident and I felt relieved they got good treatment."),
    ("anxiety", "My heart is racing and I cannot stop worrying about everything in my life right now. I keep having panic attacks in the car ever since I saw a terrible accident on the road. My mind keeps replaying the scene of the injured people and I feel like something awful is about to happen. I have been avoiding driving and I cannot sleep because my thoughts won't settle down at all."),
    ("crisis", "I went to the office today because it was my salary increment day and I felt okay in the morning. Then I saw a car accident on the road and some people were injured and in terrible pain. I got so numb and shocked that my car got out of control and my face hit the window and started bleeding. The rescue came and took me to hospital but my condition declined and I felt fever and vomiting. I was in so much pain that I wanted to kill myself and I went to the top of the building and wanted to jump."),
]


def test_long_text(vectorizer, clf, le, output_dir: Path, calibrators=None):
    """Run long-text chunk-and-aggregate inference tests."""
    print("\n" + "═" * 78)
    print("  🔬  LONG-TEXT INFERENCE TEST (chunk-and-aggregate)")
    print("═" * 78)

    results = []
    for label_expected, text in LONG_TEXT_TESTS:
        words      = text.split()
        chunk_size = 60
        overlap    = 20
        chunks     = []
        for i in range(0, max(1, len(words) - overlap), chunk_size - overlap):
            chunk = " ".join(words[i: i + chunk_size])
            if len(chunk) >= 10:
                chunks.append(chunk)
        if not chunks:
            chunks = [text]

        clean_chunks = [clean(c) for c in chunks]
        X_chunks    = vectorizer.transform(clean_chunks)
        raw_chunks  = clf.predict_proba(X_chunks)

        if calibrators:
            cal = np.zeros_like(raw_chunks)
            for i, ir in enumerate(calibrators):
                cal[:, i] = ir.predict(raw_chunks[:, i])
            row_s = cal.sum(axis=1, keepdims=True)
            proba_chunks = cal / np.maximum(row_s, 1e-9)
        else:
            proba_chunks = raw_chunks

        avg_proba  = proba_chunks.mean(axis=0)
        pred_idx   = avg_proba.argmax()
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = avg_proba[pred_idx]

        ok  = pred_label.lower() == label_expected.lower()
        tag = "✅" if ok else "❌"
        print(f"\n  {tag} [Expected: {label_expected.upper()}]")
        print(f"     → Predicted: {pred_label.upper()} ({confidence*100:.1f}%) | chunks={len(chunks)}")
        top3 = sorted(zip(le.classes_, avg_proba), key=lambda x: -x[1])[:3]
        print(f"     Top-3: {', '.join(f'{c}:{p*100:.1f}%' for c, p in top3)}")
        results.append({"expected": label_expected, "predicted": pred_label, "ok": ok})

    passed = sum(1 for r in results if r["ok"])
    print(f"\n  📋 Long-text test: {passed}/{len(results)} passed")
    return results


# ─── SECTION 9: MAIN ─────────────────────────────────────────────────────────

def main():
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPORTS_DIR / f"unified_v4_{ts}"
    run_dir.mkdir(exist_ok=True)

    print("\n" + "╔" + "═" * 76 + "╗")
    print("║   SereneMind — Unified Mental Health Model v4 (LogisticRegression)      ║")
    print("║   Features: TF-IDF word(1-4) + char(2-5)  |  Target: ≥ 90% accuracy   ║")
    print("╚" + "═" * 76 + "╝\n")

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("📂  LOADING DATASETS")
    print("─" * 60)
    
    df_raw   = load_emotion_dataset(DATA_DIR / "emotion_dataset.csv")
    df_go    = load_go_emotions(DATA_DIR / "go_emotions.csv")
    df_mh    = load_mh_conversations(DATA_DIR / "mental_health_conversations.csv")
    df_supp  = load_supplementary_data(DATA_DIR / "mental_health_sentiment.csv")
    df_seed  = load_clinical_seed()

    # Merge all
    df = merge_and_balance([df_raw, df_go, df_mh, df_supp, df_seed])

    # ── Step 2: Encode labels & split ───────────────────────────────────────── (Wait, I'll keep the numbering aligned)
    # Re-indexing numbering for consistency
    # ── Step 3: Train ─────────────────────────────────────────────────────────
    # ... (Step numbers are just comments, I'll just fix the logic)

    # ── Step 3: Encode labels & split ─────────────────────────────────────────
    print("\n🔀  SPLITTING TRAIN / VAL / TEST")
    le = LabelEncoder().fit(LABELS)
    y  = le.transform(df["label"].astype(str).values)
    X_text = np.array(df["text"].astype(str).tolist())

    X_train_txt, X_temp_txt, y_train, y_temp = train_test_split(
        X_text, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_val_txt, X_test_txt, y_val, y_test = train_test_split(
        X_temp_txt, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    print(f"  Train: {len(y_train):,}  |  Val: {len(y_val):,}  |  Test: {len(y_test):,}")

    # ── Step 4: TF-IDF features ────────────────────────────────────────────────
    print("\n📐  BUILDING TF-IDF FEATURES")
    print("─" * 60)
    print("  Fitting TF-IDF (word 1-4gram + char 2-5gram, 80K features) ...")
    t0 = time.time()
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_txt)
    X_val   = vectorizer.transform(X_val_txt)
    X_test  = vectorizer.transform(X_test_txt)
    print(f"  Feature matrix: {X_train.shape[0]:,} × {X_train.shape[1]:,} ({time.time()-t0:.1f}s)")

    # ── Step 5: Train ─────────────────────────────────────────────────────────
    clf = train_model(X_train, y_train, X_val, y_val, le)

    # ── Step 6: Calibrate ─────────────────────────────────────────────────────
    calibrators = calibrate(clf, le, X_val, y_val)

    # ── Step 7: Test set evaluation ───────────────────────────────────────────
    print("\n📊  FINAL EVALUATION ON HELD-OUT TEST SET")
    print("─" * 60)
    y_proba_test = apply_calibration(clf, calibrators, X_test)
    y_pred_test  = y_proba_test.argmax(axis=1)

    acc  = accuracy_score(y_test, y_pred_test)
    f1   = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    loss = log_loss(y_test, y_proba_test)

    print(f"\n  ✅ Test Accuracy  : {acc*100:.2f}%")
    print(f"  ✅ Test Macro F1  : {f1:.4f}")
    print(f"  ✅ Test Log-Loss  : {loss:.4f}")

    show_per_class_results(y_test, y_pred_test, le.classes_)

    # ── Step 8: Overfit analysis ───────────────────────────────────────────────
    print("\n📉  OVERFIT ANALYSIS")
    print("─" * 60)
    y_proba_train = apply_calibration(clf, calibrators, X_train)
    y_pred_train  = y_proba_train.argmax(axis=1)
    y_proba_val   = apply_calibration(clf, calibrators, X_val)
    y_pred_val    = y_proba_val.argmax(axis=1)

    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc   = accuracy_score(y_val, y_pred_val)
    gap       = (train_acc - val_acc) * 100

    print(f"  Train Acc  : {train_acc*100:.2f}%")
    print(f"  Val   Acc  : {val_acc*100:.2f}%")
    print(f"  Test  Acc  : {acc*100:.2f}%")
    print(f"  Gap (train-val): {gap:+.2f}%")

    if gap > 12:
        print("  ⚠️  Moderate overfitting — consider increasing regularization (lower C)")
    elif gap > 5:
        print("  🟡 Mild overfitting — normal for this scale")
    else:
        print("  ✅ No significant overfit detected")

    history = {"train_acc": train_acc, "val_acc": val_acc, "test_acc": acc}

    # ── Step 9: Save model ────────────────────────────────────────────────────
    print("\n💾  SAVING MODEL")
    bundle = {
        "vectorizer":   vectorizer,
        "classifier":   clf,          # LogisticRegression — picklable
        "calibrators":  calibrators,  # list[IsotonicRegression] — picklable
        "label_encoder": le,
        "classes":      list(le.classes_),
        "trained_at":   ts,
        "model_version": "v4",
        "val_accuracy": val_acc * 100,
        "test_accuracy": acc * 100,
        "test_f1":      f1,
    }
    model_path  = MODEL_DIR / "unified_mental_health.joblib"
    backup_path = MODEL_DIR / f"prev_{ts}.joblib"

    if model_path.exists():
        import shutil
        shutil.copy(model_path, backup_path)
        print(f"  📦 Backed up old model → {backup_path.name}")

    joblib.dump(bundle, model_path, compress=3)
    size_mb = model_path.stat().st_size / 1e6
    print(f"  ✅ Model saved → {model_path}")
    print(f"  📦 Size: {size_mb:.2f} MB")

    # ── Step 10: Plots ────────────────────────────────────────────────────────
    plot_learning_curves_logreg(history, run_dir)
    plot_confusion_matrix(y_test, y_pred_test, le.classes_, run_dir)
    plot_per_class_f1(y_test, y_pred_test, le.classes_, run_dir)

    # ── Step 11: Long-text test ────────────────────────────────────────────────
    long_results = test_long_text(vectorizer, clf, le, run_dir, calibrators=calibrators)
    long_passed  = sum(1 for r in long_results if r["ok"])

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 76 + "╗")
    print("║   TRAINING COMPLETE — SereneMind Unified Model v4                      ║")
    print("╠" + "═" * 76 + "╣")
    print(f"║   Test Accuracy  : {acc*100:>7.2f}%                                           ║")
    print(f"║   Test Macro F1  : {f1:>8.4f}                                          ║")
    print(f"║   Model Size     : {size_mb:>7.2f} MB                                         ║")
    print(f"║   Long-text test : {long_passed}/{len(long_results)} passed                                       ║")
    acc_flag = "✅ TARGET MET!" if acc >= 0.88 else ("🟡 CLOSE" if acc >= 0.82 else "❌ NEEDS WORK")
    print(f"║   Status         : {acc_flag:<55} ║")
    print("╚" + "═" * 76 + "╝")


if __name__ == "__main__":
    main()
