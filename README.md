
# 📰 Hybrid News Recommendation System

This project is a **Streamlit web application** that recommends news articles to users based on their reading history.  
It uses a **Hybrid Recommendation Approach** combining **Content-Based Filtering** with category matching to provide personalized suggestions.

---

## 📌 Features
- **User Authentication** — Simple username-based login.
- **News Browsing** — Search and filter CNN articles by headline.
- **Read History Tracking** — Keeps track of articles you have read.
- **Hybrid Recommendations** — Uses TF-IDF vectorization + cosine similarity + category filtering.
- **Evaluation Metrics** — Precision, Recall, F1-score, NDCG, and MAP.
- **Interactive UI** — Built with [Streamlit](https://streamlit.io/).

---

## 🛠️ Tech Stack
- **Python 3.x**
- **Streamlit** (Frontend + Backend)
- **Pandas / NumPy** (Data handling)
- **NLTK** (Text preprocessing & stopwords)
- **Scikit-learn** (TF-IDF, cosine similarity, evaluation metrics)

---

## 📂 Dataset
The project uses a cleaned CNN articles dataset:

CNN_Articels_clean.csv

markdown
Copy
Edit

**Required Columns:**
- `Headline` — Article title
- `Description` — Short description
- `Category` — News category (e.g., Politics, Sports)
- `Url` — Link to the article
- `Date published` — Publish date

⚠ **Note**: This dataset is not provided in the repo due to copyright or size. You will need to place your CSV file in the project root directory.

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository
`
git clone https://github.com/akramshammi/Hybrid_News_Recommendation_System.git
cd Hybrid_News_Recommendation_System
2️⃣ Install Dependencies
It’s recommended to use a virtual environment:

bash
Copy
Edit
pip install -r requirements.txt
requirements.txt example:

nginx
Copy
Edit
streamlit
pandas
numpy
nltk
scikit-learn
3️⃣ Download NLTK Resources
bash
Copy
Edit
python -m nltk.downloader stopwords wordnet
4️⃣ Add Dataset
Place CNN_Articels_clean.csv in the project folder.

5️⃣ Run the App
bash
Copy
Edit
streamlit run app.py
📊 How It Works
Data Loading — Reads the CSV file and fills missing values.

Preprocessing — Cleans text (lowercasing, punctuation removal, stopword removal, lemmatization).

TF-IDF Vectorization — Converts text into numerical form.

Cosine Similarity — Finds similar articles based on user read history.

Category Filtering — Ensures recommendations match categories user has shown interest in.

Evaluation — Calculates recommendation quality metrics.
