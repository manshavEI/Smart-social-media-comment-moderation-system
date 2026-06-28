# ============================================================
#   YouTube Toxic Comment Classifier — Enhanced Edition
#   pip install google-api-python-client pandas scikit-learn
#          nltk textblob openpyxl matplotlib seaborn pillow
#   python main.py
# ============================================================
# .\.venv\Scripts\activate
# python main.py
# ============================================================


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from googleapiclient.discovery import build
from wordcloud import WordCloud
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import re
import html
import tkinter as tk
from tkinter import ttk, messagebox
import threading

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
API_KEY = "AIzaSyD2VACo0RJhikm6fWhCbdy_V43j3lywvJY"
MAX_COMMENTS = 200


# ─────────────────────────────────────────────
#  WORD LISTS
# ─────────────────────────────────────────────

# Score +2 each → REMOVED if total ≥ 2
BANNED = {
    "hate", "kill", "murder", "die", "death", "shoot", "bomb", "terrorist",
    "abuse", "harm", "hurt", "destroy", "stupid", "idiot", "moron", "imbecile",
    "dumbass", "retard", "ugly", "freak", "loser", "pathetic", "worthless",
    "garbage", "trash", "scum", "filth", "disgusting", "vile", "evil", "demon",
    "monster", "racist", "racism", "bigot", "sexist", "homophobe", "bully",
    "harass", "stalk", "threaten", "intimidate", "cancer", "parasite", "plague",
    "vermin", "slut", "whore", "bastard", "coward", "inferior", "subhuman",
    "savage", "barbarian", "braindead", "brainless", "shutup", "kys",
    "liar", "fraud", "cheat", "phony", "scammer", "psycho", "lunatic",
    "insane", "weirdo", "creep", "pervert", "hopeless", "useless", "pointless",
    "meaningless", "failure", "nobody", "disgust", "dimwit", "halfwit",
    "scumbag", "lowlife", "degenerate", "deviant", "predator", "criminal",
    "violent", "abusive", "offensive", "repulsive", "despicable", "deplorable",
    "sickening", "nauseating", "revolting",
}

# Score +1 each → FLAGGED if total == 1 (unless positive shield cancels it)
MILD = {
    "bad", "annoying", "dumb", "boring", "lame", "meh", "cringe", "overrated",
    "mediocre", "disappointing", "frustrating", "confusing", "misleading",
    "wrong", "unfair", "biased", "clickbait", "waste", "terrible", "awful",
    "poor", "sloppy", "rushed", "lazy", "weak", "rude", "harsh", "slow",
    "cheap", "broken", "inconsistent", "inaccurate", "irrelevant",
    "overdone", "repetitive", "monotone", "dull", "flat",
}

# Any single match → subtract 1 from score (positive shield)
POSITIVE = {
    "amazing", "awesome", "excellent", "fantastic", "brilliant", "outstanding",
    "wonderful", "incredible", "superb", "perfect", "love", "great", "best",
    "beautiful", "inspiring", "helpful", "informative", "educational",
    "interesting", "engaging", "creative", "innovative", "genius", "talented",
    "skilled", "motivating", "uplifting", "heartwarming", "touching", "moving",
    "funny", "hilarious", "entertaining", "enjoyable", "delightful", "honest",
    "trustworthy", "genuine", "authentic", "kind", "compassionate", "generous",
    "thoughtful", "caring", "appreciate", "grateful", "blessed", "lucky",
    "underrated", "recommend", "quality", "professional", "legendary", "iconic",
    "classic", "timeless", "masterpiece", "insightful", "wise", "smart",
    "intelligent", "clever", "supportive", "encouraging", "optimistic", "hopeful",
    "excited", "happy", "joy", "fun", "cool", "nice", "adorable", "cute", "sweet",
    "lovely", "charming", "satisfying", "relaxing", "peaceful", "serene",
    "good", "well", "thank", "thanks", "sir", "mam", "please", "respect",
    "subscribed", "subscribe", "liked", "share", "learn", "learned", "helped",
    "understood", "clear", "easy", "simple", "quick", "efficient", "accurate",
    "love", "liked", "enjoyed", "watched", "finished", "completed", "done",
}

STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "of", "and", "or", "but",
    "not", "this", "that", "was", "are", "be", "with", "as", "for", "from", "by",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "its", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "im", "ive", "youre", "dont", "cant",
    "were", "been", "being", "am", "https", "http", "www", "com", "br", "s", "t", "u", "r",
    "amp", "nbsp",
}


# ─────────────────────────────────────────────
#  ML MODEL  (balanced training set)
# ─────────────────────────────────────────────
_TRAIN = [
    # TOXIC  → 1
    ("I hate you so much", 1),
    ("You are a stupid idiot", 1),
    ("Kill yourself you loser", 1),
    ("What a disgusting piece of garbage", 1),
    ("You are worthless and pathetic", 1),
    ("Ugly freak go away", 1),
    ("This is absolute trash content", 1),
    ("You moron stop making videos", 1),
    ("Worst channel ever you are dumb", 1),
    ("Scum of the earth", 1),
    ("Racist and offensive video", 1),
    ("Threatening you right now", 1),
    ("You are a failure and a fraud", 1),
    ("Go die nobody likes you", 1),
    ("Disgusting content I hate this", 1),
    ("Terrible person you are evil", 1),
    ("Stop making videos you idiot", 1),
    ("Absolute garbage delete your channel", 1),
    # NOT TOXIC → 0
    ("This is amazing thank you", 0),
    ("Love this video so much", 0),
    ("Very helpful tutorial learned a lot", 0),
    ("Great work keep it up", 0),
    ("Beautiful and inspiring content", 0),
    ("You are brilliant and talented", 0),
    ("Thank you for sharing this", 0),
    ("Fantastic video loved every second", 0),
    ("Lots of help really useful", 0),
    ("Good explanation sir", 0),
    ("This video helped me understand", 0),
    ("Subscribed great content", 0),
    ("Nice video very informative", 0),
    ("Well explained thank you", 0),
    ("I finished the whole video today", 0),
    ("Very nice presentation", 0),
    ("Could be better but still good", 0),
    ("Not my favourite but decent", 0),
    ("I disagree with some points here", 0),
    ("This was a bit confusing", 0),
    ("The audio quality was poor", 0),
    ("Lots of help sir thank you", 0),
    ("Awesome video very educational", 0),
    ("I learned so much from this", 0),
    ("Please make more videos like this", 0),
]
_texts = [t for t, l in _TRAIN]
_labels = [l for t, l in _TRAIN]

_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
_X_train = _vectorizer.fit_transform(_texts)
_model = LogisticRegression(max_iter=500, C=0.5)
_model.fit(_X_train, _labels)


# ─────────────────────────────────────────────
#  TEXT HELPERS
# ─────────────────────────────────────────────
def strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities (YouTube uses textDisplay which contains HTML)."""
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<.*?>", "", text)
    return text.strip()


def clean_text(text: str) -> str:
    text = strip_html(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_sentiment(comment: str) -> str:
    p = TextBlob(comment).sentiment.polarity
    if p > 0.05:
        return "Positive"
    if p < -0.05:
        return "Negative"
    return "Neutral"


def rule_score(cleaned: str) -> int:
    words = set(cleaned.split())
    score = 0
    for w in BANNED:
        if w in words or re.search(rf"\b{re.escape(w)}\b", cleaned):
            score += 2
    for w in MILD:
        if w in words or re.search(rf"\b{re.escape(w)}\b", cleaned):
            score += 1
    # Positive shield — one positive word cancels one mild word
    for w in POSITIVE:
        if w in words or re.search(rf"\b{re.escape(w)}\b", cleaned):
            score = max(0, score - 1)
            break
    return score


def classify_comment(raw_text: str):
    cleaned = clean_text(raw_text)
    score = rule_score(cleaned)
    sentiment = get_sentiment(cleaned)

    # ML is a secondary signal only when rule score is 0
    ml_pred = 0
    if score == 0:
        try:
            ml_pred = _model.predict(_vectorizer.transform([cleaned]))[0]
        except Exception:
            ml_pred = 0

    if score >= 2:
        status = "REMOVED ❌"
    elif score == 1 or (ml_pred == 1 and sentiment == "Negative"):
        status = "FLAGGED ⚠️"
    else:
        status = "ALLOWED ✅"

    return status, sentiment, score


# ─────────────────────────────────────────────
#  YOUTUBE API
# ─────────────────────────────────────────────
def get_video_title(youtube, video_id: str) -> str:
    try:
        resp = youtube.videos().list(part="snippet", id=video_id).execute()
        items = resp.get("items", [])
        if items:
            return items[0]["snippet"]["title"]
    except Exception:
        pass
    return "Unknown_Video"


def get_comments(video_id: str):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    title = get_video_title(youtube, video_id)
    comments = []
    next_page_token = None

    while len(comments) < MAX_COMMENTS:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
        ).execute()

        for item in response.get("items", []):
            sn = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text":   strip_html(sn["textDisplay"]),   # HTML stripped here
                "author": sn["authorDisplayName"],
                "likes":  sn["likeCount"],
                "time":   sn["publishedAt"],
            })
            if len(comments) >= MAX_COMMENTS:
                break

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return title, comments


# ─────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────
BG = "#0f1117"
PANEL = "#1a1d27"
ACCENT = "#ff4b6e"
ACCENT2 = "#4b9eff"
TEXT = "#e8eaf0"
SUBTEXT = "#8b8fa8"
SUCCESS = "#2ecc71"
WARNING = "#f39c12"
REMOVED = "#e74c3c"

STATUS_COLOR = {"ALLOWED ✅": SUCCESS,
                "FLAGGED ⚠️": WARNING, "REMOVED ❌": REMOVED}
SENT_COLOR = {"Positive": "#2ecc71",
              "Neutral": "#3498db", "Negative": "#e74c3c"}


# ─────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────
print(df.head())
print(df.columns)
print(len(df))


def build_charts(df: pd.DataFrame, chart_frame: tk.Frame):
    # Clear any old widgets
    for w in chart_frame.winfo_children():
        w.destroy()
    chart_frame.update_idletasks()   # force layout flush before drawing
    plt.close("all")                 # free memory from previous figures

    sns.set_theme(style="dark_background")
    plt.rcParams.update({
        "figure.facecolor": PANEL,
        "axes.facecolor":   PANEL,
        "axes.edgecolor":   "#2a2d3e",
        "axes.labelcolor":  TEXT,
        "xtick.color":      SUBTEXT,
        "ytick.color":      SUBTEXT,
        "text.color":       TEXT,
        "grid.color":       "#2a2d3e",
        "grid.alpha":       0.5,
        "font.family":      "monospace",
    })

    dpi = 96
    w_px = max(chart_frame.winfo_width(),  700)   # actual pixel width
    h_px = max(chart_frame.winfo_height(), 700)
    fig = plt.figure(figsize=(w_px/dpi, h_px/dpi), dpi=dpi, facecolor=PANEL)
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.50, wspace=0.38,
                           left=0.10, right=0.97, top=0.95, bottom=0.07)

    # ── 1. Sentiment bar ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    order = ["Positive", "Neutral", "Negative"]
    scnts = df["Sentiment"].value_counts().reindex(order, fill_value=0)
    cols = [SENT_COLOR[s] for s in order]
    bars = ax1.bar(order, scnts.values, color=cols, width=0.45,
                   zorder=3, edgecolor=PANEL, linewidth=1.5)
    ax1.set_title("💬 Sentiment", fontsize=10,
                  fontweight="bold", pad=8, color=TEXT)
    ax1.set_ylabel("Comments", fontsize=8)
    ax1.grid(axis="y", zorder=0)
    ax1.set_axisbelow(True)
    for bar, val in zip(bars, scnts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.4,
                 str(val), ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color=TEXT)

    # ── 2. Moderation pie ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    mcnts = df["Status"].value_counts()
    mcols = [STATUS_COLOR.get(s, ACCENT) for s in mcnts.index]
    _, _, autotexts = ax2.pie(
        mcnts.values,
        labels=mcnts.index,
        autopct="%1.0f%%",
        colors=mcols,
        startangle=140,
        wedgeprops=dict(linewidth=2, edgecolor=PANEL),
        textprops={"fontsize": 7, "color": TEXT},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(8)
    ax2.set_title("🛡️ Moderation", fontsize=10,
                  fontweight="bold", pad=8, color=TEXT)

    # ── 3. Top toxic authors ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    toxic = df[df["Status"] == "REMOVED ❌"]
    if not toxic.empty:
        top8 = toxic["Author"].value_counts().head(8)
        ax3.barh(top8.index[::-1], top8.values[::-1],
                 color=REMOVED, edgecolor=PANEL, linewidth=1)
        ax3.set_title("🚨 Toxic Authors", fontsize=10,
                      fontweight="bold", pad=8, color=TEXT)
        ax3.set_xlabel("Removed comments", fontsize=8)
        ax3.grid(axis="x", zorder=0)
        ax3.set_axisbelow(True)
        for i, v in enumerate(top8.values[::-1]):
            ax3.text(v + 0.05, i, str(v), va="center", fontsize=8, color=TEXT)
    else:
        ax3.text(0.5, 0.5, "No removed\ncomments 🎉",
                 ha="center", va="center", fontsize=13,
                 color=SUCCESS, transform=ax3.transAxes)
        ax3.set_title("🚨 Toxic Authors", fontsize=10,
                      fontweight="bold", pad=8, color=TEXT)
        ax3.axis("off")

    # ── 4. Word cloud ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    all_words = " ".join(df["Comment"].apply(clean_text))
    filtered = " ".join(w for w in all_words.split()
                        if w not in STOPWORDS and len(w) > 2)
    if not filtered.strip():
        filtered = "youtube comments"

    wc = WordCloud(
        width=500, height=380,
        background_color=PANEL,
        colormap="RdYlGn",
        max_words=100,
        prefer_horizontal=0.85,
        collocations=False,
    ).generate(filtered)

    ax4.imshow(wc, interpolation="bilinear")
    ax4.axis("off")
    ax4.set_title("☁️ Word Cloud", fontsize=10,
                  fontweight="bold", pad=8, color=TEXT)

    # ── Embed canvas ───────────────────────────────────────────────────────
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.configure(bg=PANEL)
    widget.pack(fill=tk.BOTH, expand=True)
    chart_frame.update_idletasks()   # force render after packing


# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YouTube Comment Analyser")
        self.geometry("1400x840")
        self.minsize(1100, 700)
        self.configure(bg=BG)
        self._build_ui()

    def _build_ui(self):
        self._header()
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG,
                              sashwidth=6, sashrelief=tk.FLAT)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        left = self._left_panel(pane)
        right = self._right_panel(pane)
        pane.add(left,  minsize=380)
        pane.add(right, minsize=550)
        pane.sash_place(0, 450, 0)

    def _header(self):
        hf = tk.Frame(self, bg=ACCENT, height=52)
        hf.pack(fill=tk.X)
        hf.pack_propagate(False)
        tk.Label(hf, text="🎬  YouTube Toxic Comment Classifier",
                 bg=ACCENT, fg="white",
                 font=("Georgia", 17, "bold")).pack(side=tk.LEFT, padx=18, pady=10)
        tk.Label(hf, text="TextBlob · TF-IDF · Rule Engine  v3",
                 bg=ACCENT, fg="#ffd4db",
                 font=("Consolas", 9)).pack(side=tk.RIGHT, padx=18)

    # ── LEFT ──────────────────────────────────────────────────────────────
    def _left_panel(self, parent):
        lf = tk.Frame(parent, bg=PANEL)

        card = tk.Frame(lf, bg="#22253a", pady=12, padx=14)
        card.pack(fill=tk.X, padx=10, pady=(12, 6))
        tk.Label(card, text="📎  YouTube URL or Video ID",
                 bg="#22253a", fg=SUBTEXT, font=("Consolas", 9)).pack(anchor="w")

        url_row = tk.Frame(card, bg="#22253a")
        url_row.pack(fill=tk.X, pady=(6, 0))
        self.url_var = tk.StringVar()
        tk.Entry(url_row, textvariable=self.url_var,
                 font=("Consolas", 11), bg="#0f1117", fg=TEXT,
                 insertbackground=ACCENT, relief=tk.FLAT, bd=6
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        tk.Button(url_row, text="▶  Analyse",
                  font=("Georgia", 10, "bold"),
                  bg=ACCENT, fg="white",
                  activebackground="#c0334f", activeforeground="white",
                  relief=tk.FLAT, bd=0, padx=14, pady=6, cursor="hand2",
                  command=self._start_analysis
                  ).pack(side=tk.LEFT, padx=(8, 0))

        # Stats strip
        stats = tk.Frame(lf, bg=PANEL)
        stats.pack(fill=tk.X, padx=10, pady=2)
        self.stat_vars = {}
        for col, (key, label, color) in enumerate([
            ("total",   "Total",   ACCENT2),
            ("allowed", "Allowed", SUCCESS),
            ("flagged", "Flagged", WARNING),
            ("removed", "Removed", REMOVED),
        ]):
            sf = tk.Frame(stats, bg=color, padx=8, pady=6)
            sf.grid(row=0, column=col, sticky="nsew", padx=2, pady=2)
            stats.columnconfigure(col, weight=1)
            v = tk.StringVar(value="—")
            self.stat_vars[key] = v
            tk.Label(sf, textvariable=v, bg=color, fg="white",
                     font=("Georgia", 15, "bold")).pack()
            tk.Label(sf, text=label, bg=color, fg="white",
                     font=("Consolas", 8)).pack()

        # Progress bar
        self.progress = ttk.Progressbar(lf, mode="indeterminate",
                                        style="Accent.Horizontal.TProgressbar")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Accent.Horizontal.TProgressbar",
                        troughcolor=BG, background=ACCENT, thickness=4)
        self.progress.pack(fill=tk.X, padx=10, pady=4)

        tk.Label(lf, text="  COMMENT FEED", bg=PANEL, fg=SUBTEXT,
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=10)

        feed_outer = tk.Frame(lf, bg="#0f1117", padx=2, pady=2)
        feed_outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 10))

        self.canvas_feed = tk.Canvas(feed_outer, bg="#0f1117",
                                     highlightthickness=0)
        sb = tk.Scrollbar(feed_outer, orient="vertical",
                          command=self.canvas_feed.yview,
                          bg=BG, troughcolor=BG, relief=tk.FLAT)
        self.canvas_feed.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_feed.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.feed_inner = tk.Frame(self.canvas_feed, bg="#0f1117")
        self._feed_win = self.canvas_feed.create_window(
            (0, 0), window=self.feed_inner, anchor="nw")
        self.feed_inner.bind("<Configure>", self._on_feed_cfg)
        self.canvas_feed.bind("<Configure>", self._on_canvas_cfg)
        self.canvas_feed.bind_all("<MouseWheel>", self._on_scroll)

        return lf

    def _on_feed_cfg(self, _e):
        self.canvas_feed.configure(scrollregion=self.canvas_feed.bbox("all"))

    def _on_canvas_cfg(self, e):
        self.canvas_feed.itemconfig(self._feed_win, width=e.width)

    def _on_scroll(self, e):
        self.canvas_feed.yview_scroll(int(-1 * (e.delta / 120)), "units")

    def _add_card(self, text, author, likes, status, sentiment, score):
        color = STATUS_COLOR.get(status, SUBTEXT)
        card = tk.Frame(self.feed_inner, bg="#1e2130",
                        highlightbackground=color,
                        highlightthickness=1, pady=6, padx=8)
        card.pack(fill=tk.X, pady=3, padx=2)

        top = tk.Frame(card, bg="#1e2130")
        top.pack(fill=tk.X)
        tk.Label(top, text=status, bg="#1e2130", fg=color,
                 font=("Consolas", 8, "bold")).pack(side=tk.LEFT)
        tk.Label(top, text=f"👍 {likes}  |  {sentiment}",
                 bg="#1e2130", fg=SUBTEXT,
                 font=("Consolas", 8)).pack(side=tk.RIGHT)

        preview = text[:200] + ("…" if len(text) > 200 else "")
        tk.Label(card, text=preview,
                 bg="#1e2130", fg=TEXT,
                 font=("Consolas", 9),
                 wraplength=370, justify="left").pack(anchor="w", pady=2)
        tk.Label(card, text=f"— {author}",
                 bg="#1e2130", fg=SUBTEXT,
                 font=("Consolas", 8, "italic")).pack(anchor="e")

    # ── RIGHT ─────────────────────────────────────────────────────────────
    def _right_panel(self, parent):
        rf = tk.Frame(parent, bg=PANEL)
        tk.Label(rf, text="  ANALYTICS  DASHBOARD",
                 bg=PANEL, fg=SUBTEXT,
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=12, pady=(10, 0))

        self.chart_frame = tk.Frame(rf, bg=PANEL, width=700, height=700)
        self.chart_frame.pack_propagate(False)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        tk.Label(self.chart_frame,
                 text="📊\n\nCharts will appear here\nafter analysis completes.",
                 bg=PANEL, fg=SUBTEXT,
                 font=("Georgia", 13), justify="center"
                 ).place(relx=0.5, rely=0.5, anchor="center")
        return rf

    # ── ANALYSIS ──────────────────────────────────────────────────────────
    def _start_analysis(self):
        url = self.url_var.get().strip()
        if not url:
            messagebox.showwarning(
                "No URL", "Please enter a YouTube URL or video ID.")
            return
        for w in self.feed_inner.winfo_children():
            w.destroy()
        for k in self.stat_vars:
            self.stat_vars[k].set("…")
        self.progress.start(10)
        threading.Thread(target=self._run, args=(url,), daemon=True).start()

    @staticmethod
    def _extract_id(url: str) -> str:
        m = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]+)", url)
        return m.group(1) if m else url

    def _run(self, url: str):
        import json as _json
        video_id = self._extract_id(url)
        try:
            video_title, comments = get_comments(video_id)
        except Exception as e:
            raw = str(e)
            friendly = raw
            try:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    payload = _json.loads(m.group())
                    friendly = payload.get("error", {}).get("message", raw)
            except Exception:
                pass
            msg = (
                f"YouTube API Error:\n\n{friendly}\n\n"
                "Common fixes:\n"
                "• Enable 'YouTube Data API v3' in Google Cloud Console\n"
                "• Remove key restrictions (HTTP referrer / IP address)\n"
                "• Check the video URL is correct\n"
                "• Comments may be disabled on this video\n"
                "• Free quota may be exhausted (10,000 units/day)"
            )
            self.after(0, lambda m=msg: self._err(m))
            return

        if not comments:
            self.after(0, lambda: self._err(
                "API returned 0 comments.\n"
                "Comments may be disabled, video private, or ID is wrong."
            ))
            return

        rows = []
        for c in comments:
            status, sentiment, score = classify_comment(c["text"])
            rows.append({
                "Comment":   c["text"],
                "Author":    c["author"],
                "Likes":     c["likes"],
                "Time":      c["time"],
                "Status":    status,
                "Sentiment": sentiment,
                "Score":     score,
            })

        df = pd.DataFrame(rows)
        safe = re.sub(r'[\\/*?:"<>|]', "_", video_title)[:60]
        csv_p = f"{safe}.csv"
        xlsx_p = f"{safe}.xlsx"
        df.to_csv(csv_p,  index=False)
        df.to_excel(xlsx_p, index=False)

        self.after(0, lambda: self._finish(
            df, video_title, len(rows), csv_p, xlsx_p))

    def _finish(self, df, title, total, csv_p, xlsx_p):
        self.progress.stop()

        self.stat_vars["total"].set(str(total))
        self.stat_vars["allowed"].set(str((df["Status"] == "ALLOWED ✅").sum()))
        self.stat_vars["flagged"].set(
            str((df["Status"] == "FLAGGED ⚠️").sum()))
        self.stat_vars["removed"].set(str((df["Status"] == "REMOVED ❌").sum()))

        for _, row in df.iterrows():
            self._add_card(row["Comment"], row["Author"], row["Likes"],
                           row["Status"], row["Sentiment"], row["Score"])

        # Force full window layout before drawing charts
        self.update()
        self.update_idletasks()
        build_charts(df, self.chart_frame)

        self.after(200, lambda: messagebox.showinfo(
            "✅  Done",
            f"Video : {title}\n"
            f"Total : {total} comments\n\n"
            f"Saved:\n  {csv_p}\n  {xlsx_p}"
        ))

    def _err(self, msg: str):
        self.progress.stop()
        messagebox.showerror("Error", msg)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    App().mainloop()
