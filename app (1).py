import os
import time

import numpy as np
import pandas as pd
import requests
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
TMDB_API_KEY = os.environ["TMDB_API_KEY"]   # set this as a Space "Secret", never hardcode it
TMDB_BASE = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_POSTER = "https://placehold.co/500x750/1a1a2e/e94560?text=No+Poster"

WATCH_REGION = "IN"
NETFLIX_PROVIDER_ID = 8
PRIME_PROVIDER_ID = 9

# ---------------------------------------------------------------------------
# 2. Fetch a large, multi-genre movie catalog from TMDB
# ---------------------------------------------------------------------------
def fetch_genre_map():
    r = requests.get(f"{TMDB_BASE}/genre/movie/list", params={"api_key": TMDB_API_KEY})
    r.raise_for_status()
    return {g["id"]: g["name"] for g in r.json()["genres"]}


def fetch_listing(endpoint, pages=10):
    results = []
    for page in range(1, pages + 1):
        r = requests.get(
            f"{TMDB_BASE}/movie/{endpoint}",
            params={"api_key": TMDB_API_KEY, "page": page, "language": "en-US"},
        )
        if r.status_code != 200:
            break
        data = r.json().get("results", [])
        if not data:
            break
        results.extend(data)
        time.sleep(0.02)
    return results


def fetch_trending(window="week", pages=10):
    results = []
    for page in range(1, pages + 1):
        r = requests.get(
            f"{TMDB_BASE}/trending/movie/{window}",
            params={"api_key": TMDB_API_KEY, "page": page},
        )
        if r.status_code != 200:
            break
        data = r.json().get("results", [])
        if not data:
            break
        results.extend(data)
        time.sleep(0.02)
    return results


def fetch_by_genre(genre_id, pages=3):
    results = []
    for page in range(1, pages + 1):
        r = requests.get(
            f"{TMDB_BASE}/discover/movie",
            params={
                "api_key": TMDB_API_KEY,
                "with_genres": genre_id,
                "sort_by": "popularity.desc",
                "page": page,
            },
        )
        if r.status_code != 200:
            break
        data = r.json().get("results", [])
        if not data:
            break
        results.extend(data)
        time.sleep(0.02)
    return results


def fetch_by_provider(provider_id, region=WATCH_REGION, pages=8):
    results = []
    for page in range(1, pages + 1):
        r = requests.get(
            f"{TMDB_BASE}/discover/movie",
            params={
                "api_key": TMDB_API_KEY,
                "with_watch_providers": provider_id,
                "watch_region": region,
                "sort_by": "popularity.desc",
                "page": page,
            },
        )
        if r.status_code != 200:
            break
        data = r.json().get("results", [])
        if not data:
            break
        results.extend(data)
        time.sleep(0.02)
    return results


print("Fetching a large catalog from TMDB — this can take 1-2 minutes...")
GENRE_MAP = fetch_genre_map()

raw = []
for endpoint in ["popular", "top_rated", "now_playing", "upcoming"]:
    raw.extend(fetch_listing(endpoint, pages=10))
raw.extend(fetch_trending("day", pages=8))
raw.extend(fetch_trending("week", pages=8))
for gid in GENRE_MAP:
    raw.extend(fetch_by_genre(gid, pages=3))

netflix_raw = fetch_by_provider(NETFLIX_PROVIDER_ID, pages=8)
prime_raw = fetch_by_provider(PRIME_PROVIDER_ID, pages=8)
print("Netflix raw results:", len(netflix_raw))
print("Prime raw results:", len(prime_raw))
raw.extend(netflix_raw)
raw.extend(prime_raw)


def to_dataframe(movie_list):
    columns = ["id", "title", "overview", "genres", "poster_path", "rating", "vote_count", "release_date"]
    seen = {}
    for m in movie_list:
        if m.get("id") not in seen and m.get("title") and m.get("overview"):
            seen[m["id"]] = m
    rows = [
        {
            "id": m["id"],
            "title": m["title"],
            "overview": m.get("overview", ""),
            "genres": " ".join(GENRE_MAP.get(gid, "") for gid in m.get("genre_ids", [])),
            "poster_path": m.get("poster_path"),
            "rating": round(m.get("vote_average", 0), 1),
            "vote_count": m.get("vote_count", 0),
            "release_date": m.get("release_date", "")[:4] if m.get("release_date") else "N/A",
        }
        for m in seen.values()
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    df_ = pd.DataFrame(rows)
    return df_[df_["vote_count"] > 15].reset_index(drop=True)


df = to_dataframe(raw)
netflix_df = to_dataframe(netflix_raw).sort_values("rating", ascending=False).reset_index(drop=True)
prime_df = to_dataframe(prime_raw).sort_values("rating", ascending=False).reset_index(drop=True)

print(f"Main catalog: {len(df)} movies")
print(f"Netflix trending ({WATCH_REGION}): {len(netflix_df)} movies")
print(f"Prime Video trending ({WATCH_REGION}): {len(prime_df)} movies")

# ---------------------------------------------------------------------------
# 3. Content-based similarity engine
# ---------------------------------------------------------------------------
df["soup"] = (df["genres"] + " ") * 3 + df["overview"]

tfidf = TfidfVectorizer(stop_words="english", max_features=30000)
tfidf_matrix = tfidf.fit_transform(df["soup"])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
title_to_index = pd.Series(df.index, index=df["title"]).to_dict()

print("Similarity engine ready:", similarity_matrix.shape)

# ---------------------------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------------------------
def poster_url(path):
    return f"{IMAGE_BASE}{path}" if path else PLACEHOLDER_POSTER


def rating_color(rating):
    if rating >= 7.5:
        return "#2ecc71"
    elif rating >= 6:
        return "#f1c40f"
    return "#e74c3c"


_trailer_cache = {}


def get_trailer_html(movie_id):
    if movie_id in _trailer_cache:
        return _trailer_cache[movie_id]
    r = requests.get(f"{TMDB_BASE}/movie/{movie_id}/videos", params={"api_key": TMDB_API_KEY})
    videos = r.json().get("results", []) if r.status_code == 200 else []
    trailer = next((v for v in videos if v.get("type") == "Trailer" and v.get("site") == "YouTube"), None)
    if not trailer:
        trailer = next((v for v in videos if v.get("site") == "YouTube"), None)
    if trailer:
        html = (
            f'<iframe width="100%" height="400" '
            f'src="https://www.youtube.com/embed/{trailer["key"]}?autoplay=1&mute=1" '
            f'frameborder="0" allow="autoplay; encrypted-media" allowfullscreen '
            f'style="border-radius:12px;"></iframe>'
        )
    else:
        html = "<p style='color:#cfcfe8;'>No trailer available for this movie.</p>"
    _trailer_cache[movie_id] = html
    return html


def get_recommendations(watched_titles, num_results=10):
    watched_titles = [t for t in watched_titles if t in title_to_index]
    if not watched_titles:
        return []
    indices = [title_to_index[t] for t in watched_titles]
    avg_scores = np.mean(similarity_matrix[indices], axis=0)
    boosted = avg_scores + (df["rating"].values / 100)
    ranked = sorted(
        [(i, score) for i, score in enumerate(boosted) if df.iloc[i]["title"] not in watched_titles],
        key=lambda x: x[1],
        reverse=True,
    )[:num_results]
    return [df.iloc[i] for i, _ in ranked]


def search_catalog(query, source_df, limit=100):
    if not query:
        return source_df.head(limit)
    matches = source_df[source_df["title"].str.contains(query, case=False, na=False)]
    return matches.head(limit)


# ---------------------------------------------------------------------------
# 5. Accounts (in-memory — resets when the Space restarts)
# ---------------------------------------------------------------------------
USERS = {}


def signup(username, password):
    username = (username or "").strip()
    if not username or not password:
        return "Please enter both a username and password.", None
    if username in USERS:
        return "That username is already taken — try logging in instead.", None
    USERS[username] = {"password": password, "watched": []}
    return f"Account created! You're signed in as **{username}**.", username


def login(username, password):
    username = (username or "").strip()
    user = USERS.get(username)
    if not user or user["password"] != password:
        return "Invalid username or password.", None
    return f"Welcome back, **{username}**!", username


def add_to_watched(username, title):
    if not username:
        return "Log in first to save movies to your watched list."
    if title and title not in USERS[username]["watched"]:
        USERS[username]["watched"].append(title)
    return f"Added **{title}** to {username}'s watched list ({len(USERS[username]['watched'])} total)."


# ---------------------------------------------------------------------------
# 6. Gradio front end — Marquee
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important; }
#title-block h1 {
    text-align: center; font-size: 2.6em;
    background: linear-gradient(90deg, #e94560, #ffd460);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800; margin-bottom: 0;
}
#title-block p { text-align: center; color: #cfcfe8; margin-top: 4px; }
.status-box { color: #f5f5f5; padding: 8px 0; }
"""


def render_gallery(records):
    return [(poster_url(m["poster_path"]), f'{m["title"]} ({m["release_date"]}) ⭐{m["rating"]}') for m in records]


def build_detail_html(movie):
    return f"""
    <div style="display:flex; gap:20px; flex-wrap:wrap; color:#f5f5f5;">
        <img src="{poster_url(movie['poster_path'])}" style="width:220px; border-radius:12px;" />
        <div style="flex:1; min-width:250px;">
            <h2 style="margin-bottom:4px;">{movie['title']} ({movie['release_date']})</h2>
            <span style="background:{rating_color(movie['rating'])}; padding:3px 10px; border-radius:20px; color:#111; font-weight:700;">⭐ {movie['rating']}</span>
            <span style="margin-left:10px; color:#b3b3cc;">{movie['genres']}</span>
            <p style="margin-top:12px; line-height:1.5; color:#dcdcf0;">{movie['overview']}</p>
        </div>
    </div>
    """


with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="rose", neutral_hue="slate")) as demo:
    current_user = gr.State(None)
    browse_records = gr.State(df.to_dict("records"))
    netflix_records = gr.State(netflix_df.to_dict("records"))
    prime_records = gr.State(prime_df.to_dict("records"))
    selected_title = gr.State(None)

    with gr.Column(elem_id="title-block"):
        gr.Markdown("# 🎬 Marquee")
        gr.Markdown("Your AI-powered movie recommendation engine")

    with gr.Tabs():
        with gr.Tab("👤 Account"):
            gr.Markdown("### Sign up or log in to save movies you've watched")
            with gr.Row():
                username_box = gr.Textbox(label="Username")
                password_box = gr.Textbox(label="Password", type="password")
            with gr.Row():
                signup_btn = gr.Button("Create Account")
                login_btn = gr.Button("Log In", variant="primary")
            account_status = gr.Markdown(elem_classes="status-box")

            signup_btn.click(signup, inputs=[username_box, password_box], outputs=[account_status, current_user])
            login_btn.click(login, inputs=[username_box, password_box], outputs=[account_status, current_user])

        with gr.Tab("📚 Browse All Movies"):
            search_box = gr.Textbox(label="Search by title", placeholder="Type a movie name...")
            gallery = gr.Gallery(value=render_gallery(df.to_dict("records")), columns=6, height=520, label="Catalog")
            detail_html = gr.HTML()
            trailer_html = gr.HTML()
            watch_btn = gr.Button("➕ Add to My Watched List")
            watch_status = gr.Markdown()

            def do_search(query):
                recs = search_catalog(query, df).to_dict("records")
                return render_gallery(recs), recs

            search_box.change(do_search, inputs=search_box, outputs=[gallery, browse_records])

            def on_select(records, evt: gr.SelectData):
                movie = records[evt.index]
                return build_detail_html(movie), get_trailer_html(movie["id"]), movie["title"]

            gallery.select(on_select, inputs=browse_records, outputs=[detail_html, trailer_html, selected_title])
            watch_btn.click(add_to_watched, inputs=[current_user, selected_title], outputs=watch_status)

        with gr.Tab("🔴 Netflix Trending"):
            gr.Markdown(f"Popular on Netflix — region: **{WATCH_REGION}**")
            nf_gallery = gr.Gallery(value=render_gallery(netflix_df.to_dict("records")), columns=6, height=520)
            nf_detail = gr.HTML()
            nf_trailer = gr.HTML()

            def on_select_nf(records, evt: gr.SelectData):
                movie = records[evt.index]
                return build_detail_html(movie), get_trailer_html(movie["id"])

            nf_gallery.select(on_select_nf, inputs=netflix_records, outputs=[nf_detail, nf_trailer])

        with gr.Tab("🔵 Prime Video Trending"):
            gr.Markdown(f"Popular on Prime Video — region: **{WATCH_REGION}**")
            pv_gallery = gr.Gallery(value=render_gallery(prime_df.to_dict("records")), columns=6, height=520)
            pv_detail = gr.HTML()
            pv_trailer = gr.HTML()

            def on_select_pv(records, evt: gr.SelectData):
                movie = records[evt.index]
                return build_detail_html(movie), get_trailer_html(movie["id"])

            pv_gallery.select(on_select_pv, inputs=prime_records, outputs=[pv_detail, pv_trailer])

        with gr.Tab("✨ Get Recommendations"):
            gr.Markdown("Uses your saved **watched list** if logged in — or pick manually below.")
            manual_watched = gr.Dropdown(
                choices=sorted(df["title"].tolist()),
                multiselect=True,
                label="Or manually select movies you've watched",
            )
            num_slider = gr.Slider(5, 20, value=10, step=1, label="Number of suggestions")
            recommend_btn = gr.Button("🔮 Recommend Movies", variant="primary")
            rec_gallery = gr.Gallery(columns=6, height=460, label="Recommended for you")

            def on_recommend(username, manual_titles, num_results):
                saved = USERS[username]["watched"] if username and username in USERS else []
                combined = list(set(saved + (manual_titles or [])))
                recs = get_recommendations(combined, int(num_results))
                return render_gallery(recs)

            recommend_btn.click(on_recommend, inputs=[current_user, manual_watched, num_slider], outputs=rec_gallery)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
