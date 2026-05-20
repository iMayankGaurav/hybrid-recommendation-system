# 🎬 Movie Recommendation System

A content-based Movie Recommendation System built using **Machine Learning**, **NLP**, and **Streamlit**.  
The app recommends movies similar to a selected movie using **TF-IDF Vectorization** and **Cosine Similarity** on movie overviews.

## 🚀 Live Demo

[Click Here to Open App](https://hybrid-recommendation-system-mayank.streamlit.app/)

---

# 🚀 Features

- 🎥 Content-based movie recommendations
- 🧠 NLP-powered recommendation engine
- 🎬 Movie posters fetched using TMDB API
- 🔍 Searchable movie dropdown
- 🌐 Interactive Streamlit web app
- ☁️ Deployed on Streamlit Cloud

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Requests
- TMDB API

---

# 🧠 How It Works

The recommendation system uses:

1. Movie overviews/descriptions
2. TF-IDF Vectorization
3. Cosine Similarity

Movies with similar textual descriptions are recommended together.

### Example

```text
Interstellar → The Martian, Gravity, Ad Astra
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/iMayankGaurav/hybrid-recommendation-system
cd hybrid-recommendation-system
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Create `.env` File

Create a `.env` file in the root directory:

```env
API_KEY=your_tmdb_api_key
```

---

## 4️⃣ Run the Application

```bash
streamlit run app.py
```

---

# 🔑 TMDB API

This project uses the TMDB API for fetching movie posters.

Get your free API key from:

https://developer.themoviedb.org/docs/getting-started

---

# 📊 Dataset

Datasets used:
- TMDB 5000 Movies Dataset
- MovieLens Dataset

---

# 🎯 Future Improvements

- Hybrid Recommendation System
- Collaborative Filtering
- User Authentication
- Watchlist Feature
- Trending Movies Section
- Deep Learning-based Recommendations

---

# 👨‍💻 Author

**Mayank Gaurav**

If you liked this project, feel free to ⭐ the repository.