
title: Marquee Movie App
emoji: 🎬
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
---

# 🎬 Marquee — AI Movie Recommendation Engine

A movie recommendation app powered by TMDB and content-based filtering.

## Features
- Browse 1000+ movies from TMDB
- Netflix & Prime Video trending sections
- AI-powered recommendations based on your watch history
- YouTube trailers
- In-memory user accounts

## Setup

### Environment Variable (Required)
You **must** set your TMDB API key as a **Space Secret** named `TMDB_API_KEY`.

1. Get a free API key from [https://www.themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)
2. In your Hugging Face Space, go to **Settings → Secrets**
3. Add a new secret: Name = `TMDB_API_KEY`, Value = your key

## Deploy on Hugging Face Spaces

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Create a new Space with **Gradio** SDK
3. Upload these files: `app.py`, `requirements.txt`, `README.md`
4. Add your `TMDB_API_KEY` secret in Settings
