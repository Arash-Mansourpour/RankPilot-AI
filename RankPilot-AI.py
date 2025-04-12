import os
import json
import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from groq import Groq
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import random
import threading
import redis
import uuid

# Initial setup
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

os.environ["GROQ_API_KEY"] = "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"

client = Groq()

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Global state
global_state = {
    "messages": [],
    "user_profile": {
        "seo_level": "beginner",
        "interests": [],
        "writing_style": "standard",
        "language_preference": "formal",
        "sentiment_history": [],
        "query_history": [],
        "tools_used": [],
        "keywords_analyzed": []
    },
    "suggestions": [],
    "analyzed_keywords": pd.DataFrame(),
    "analyzed_content": None,
    "rankings_data": {
        "keywords": ["local seo", "site optimization", "link building", "technical seo", "backlinks"],
        "positions": [3, 5, 8, 12, 7]
    },
    "session_id": str(uuid.uuid4())
}

# Load or initialize messages from Redis
def load_messages():
    messages = redis_client.get(f"chat:{global_state['session_id']}:messages")
    if messages:
        global_state["messages"] = json.loads(messages)
    return global_state["messages"]

# Save messages to Redis
def save_messages():
    redis_client.set(f"chat:{global_state['session_id']}:messages", json.dumps(global_state["messages"]))

# SEO Functions
def extract_seo_interests(text):
    seo_interests = [
        "technical seo", "content seo", "link building", "local seo", "mobile seo",
        "site speed", "keyword research", "site structure", "competitor analysis",
        "google analytics", "search console", "ranking", "local business",
        "international seo", "ecommerce seo", "on-page seo", "core web vitals"
    ]
    return [interest for interest in seo_interests if interest.lower() in text.lower()]

def detect_seo_level(text):
    expert_terms = ["core web vitals", "canonical", "schema.org", "json-ld", "http headers",
                    "rel", "pagerank", "eat", "ymyl", "bert", "mum", "ctr"]
    intermediate_terms = ["meta description", "page title", "backlink", "alt tag", "search console",
                         "on-page seo", "off-page seo", "robots.txt", "sitemap"]
    expert_count = sum(1 for term in expert_terms if term.lower() in text.lower())
    intermediate_count = sum(1 for term in intermediate_terms if term.lower() in text.lower())
    if expert_count >= 2:
        return "expert"
    elif intermediate_count >= 2 or expert_count == 1:
        return "intermediate"
    return "beginner"

def analyze_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else "No title found"
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_desc = meta_desc['content'] if meta_desc else "No meta description found"
        internal_links = len([a for a in soup.find_all('a', href=True) if url in a['href']])
        external_links = len([a for a in soup.find_all('a', href=True) if url not in a['href']])
        page_size = len(response.content) / 1024
        return {
            "title": title,
            "meta_description": meta_desc,
            "internal_links": internal_links,
            "external_links": external_links,
            "page_size_kb": round(page_size, 2)
        }
    except Exception as e:
        return {"error": f"URL analysis error: {str(e)}"}

def analyze_keywords(keywords_text):
    keywords = [k.strip() for k in keywords_text.split(',')]
    data = []
    for keyword in keywords:
        data.append({
            "keyword": keyword,
            "difficulty": random.randint(10, 90),
            "volume": random.randint(50, 5000),
            "cpc": round(random.uniform(0.5, 10.0), 2),
            "competition": round(random.uniform(0.1, 0.9), 2)
        })
    df = pd.DataFrame(data)
    for keyword in keywords:
        if keyword not in global_state["user_profile"]["keywords_analyzed"]:
            global_state["user_profile"]["keywords_analyzed"].append(keyword)
    return df

def analyze_content_seo(content):
    word_count = len(content.split())
    sentences = content.split('.')
    sentence_count = len([s for s in sentences if s.strip()])
    avg_sentence_length = word_count / max(sentence_count, 1)
    readability_score = 100 - min(avg_sentence_length * 1.5, 70)
    keyword_density = 0
    primary_keyword = global_state["user_profile"]["keywords_analyzed"][0] if global_state["user_profile"]["keywords_analyzed"] else ""
    if primary_keyword:
        keyword_count = content.lower().count(primary_keyword.lower())
        keyword_density = (keyword_count * len(primary_keyword.split())) / max(word_count, 1) * 100
    analysis = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 1),
        "readability_score": round(readability_score, 1),
        "primary_keyword": primary_keyword,
        "keyword_density": round(keyword_density, 2),
        "h_tags": content.count('#'),
        "paragraphs": content.count('\n\n') + 1,
        "issues": []
    }
    if word_count < 300:
        analysis["issues"].append({"type": "warning", "message": "Content is less than 300 words. Longer content recommended."})
    if keyword_density > 3:
        analysis["issues"].append({"type": "warning", "message": f"Keyword density ({keyword_density}%) is too high."})
    if content.count('#') < 3 and word_count > 500:
        analysis["issues"].append({"type": "info", "message": "More heading tags recommended."})
    return analysis

def update_user_profile(query):
    new_interests = extract_seo_interests(query)
    for interest in new_interests:
        if interest not in global_state["user_profile"]["interests"]:
            global_state["user_profile"]["interests"].append(interest)
    current_level = detect_seo_level(query)
    if current_level == "expert":
        global_state["user_profile"]["seo_level"] = current_level
    elif current_level == "intermediate" and global_state["user_profile"]["seo_level"] == "beginner":
        global_state["user_profile"]["seo_level"] = current_level
    current_style = analyze_writing_style(query)
    if current_style != "standard":
        global_state["user_profile"]["writing_style"] = current_style
    lang_pref = analyze_language_preference(query)
    if lang_pref != "neutral":
        global_state["user_profile"]["language_preference"] = lang_pref
    sentiment = analyze_sentiment(query)
    global_state["user_profile"]["sentiment_history"].append({
        "query": query,
        "sentiment": sentiment,
        "timestamp": datetime.datetime.now().isoformat()
    })
    global_state["user_profile"]["query_history"].append({
        "query": query,
        "timestamp": datetime.datetime.now().isoformat()
    })

def analyze_writing_style(text):
    if len(text.split()) > 20:
        if "?" in text and text.count(",") > 3:
            return "analytical"
        elif "!" in text or "?" in text:
            return "expressive"
        elif len(text) > 200:
            return "detailed"
    return "standard"

def analyze_language_preference(text):
    formal_markers = ["please", "kindly", "thank you", "respectfully"]
    informal_markers = ["hi", "hey", "thanks", "how's it going"]
    formal_count = sum(1 for marker in formal_markers if marker.lower() in text.lower())
    informal_count = sum(1 for marker in informal_markers if marker.lower() in text.lower())
    if formal_count > informal_count:
        return "formal"
    elif informal_count > formal_count:
        return "informal"
    return "neutral"

def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return "positive"
    elif sentiment_score['compound'] <= -0.05:
        return "negative"
    return "neutral"

def format_message_for_llm(query):
    system_message = f"""You are an intelligent SEO assistant providing personalized responses based on user expertise and needs.

User Profile:
- SEO Expertise Level: {global_state["user_profile"]["seo_level"]}
- SEO Interests: {', '.join(global_state["user_profile"]["interests"]) if global_state["user_profile"]["interests"] else "Not yet identified"}
- Writing Style: {global_state["user_profile"]["writing_style"]}
- Language Preference: {global_state["user_profile"]["language_preference"]}
- Analyzed Keywords: {', '.join(global_state["user_profile"]["keywords_analyzed"]) if global_state["user_profile"]["keywords_analyzed"] else "None"}

Response Guidelines:
1. For beginners, explain concepts simply with examples, avoiding jargon.
2. For intermediate users, provide practical techniques and best practices.
3. For experts, offer advanced strategies and technical insights.
4. Include actionable examples and recommendations.
5. Tailor responses to the user's SEO interests.
6. If a URL is provided, include a basic SEO analysis (meta tags, links, etc.).
7. End with three SEO-related follow-up question suggestions (format: SUGGESTIONS: Question 1 | Question 2 | Question 3).

Recent Query History:
{json.dumps([item["query"] for item in global_state["user_profile"]["query_history"][-5:]], ensure_ascii=False)}
"""
    messages = [{"role": "system", "content": system_message}]
    for message in global_state["messages"][-10:]:
        messages.append({"role": message["role"], "content": message["content"]})
    messages.append({"role": "user", "content": query})
    return messages

def generate_response(query, output_widget, suggestion_buttons):
    def stream_response():
        update_user_profile(query)
        output_widget.config(state='normal')
        output_widget.insert(tk.END, f"\nYou: {query}\n\n", "user")
        messages = format_message_for_llm(query)
        url_pattern = re.compile(r'(https?://[^\s]+)')
        url_match = url_pattern.search(query)
        seo_report = None
        if url_match:
            url = url_match.group(0)
            seo_report = analyze_url(url)
        try:
            completion = client.chat.completions.create(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                top_p=1,
                stream=True,
                stop=None
            )
            collected_response = ""
            output_widget.insert(tk.END, "Assistant: ", "bot")
            for chunk in completion:
                chunk_content = chunk.choices[0].delta.content or ""
                collected_response += chunk_content
                output_widget.insert(tk.END, chunk_content, "bot")
                output_widget.see(tk.END)
                output_widget.update()
                time.sleep(0.01)
            if seo_report and "error" not in seo_report:
                collected_response += "\n\n**SEO Analysis:**\n"
                collected_response += f"- **Title:** {seo_report['title']}\n"
                collected_response += f"- **Meta Description:** {seo_report['meta_description']}\n"
                collected_response += f"- **Internal Links:** {seo_report['internal_links']}\n"
                collected_response += f"- **External Links:** {seo_report['external_links']}\n"
                collected_response += f"- **Page Size:** {seo_report['page_size_kb']} KB\n"
                output_widget.insert(tk.END, collected_response[len(collected_response)-len(collected_response):], "bot")
            suggestions = []
            suggestion_match = re.search(r"SUGGESTIONS: (.*?)(?:\n|$)", collected_response)
            if suggestion_match:
                suggestions_text = suggestion_match.group(1)
                suggestions = [s.strip() for s in suggestions_text.split('|')]
                collected_response = collected_response.replace(suggestion_match.group(0), "")
            global_state["messages"].append({"role": "user", "content": query})
            global_state["messages"].append({"role": "assistant", "content": collected_response})
            save_messages()
            global_state["suggestions"] = suggestions
            output_widget.insert(tk.END, "\n\n**Suggestions:**\n" + "\n".join(suggestions) + "\n", "suggestions")
            
            # Update suggestion buttons
            for i, btn in enumerate(suggestion_buttons):
                if i < len(suggestions):
                    btn.config(text=suggestions[i], state='normal')
                else:
                    btn.config(text="", state='disabled')
                    
        except Exception as e:
            output_widget.insert(tk.END, f"\nError: {str(e)}\n", "error")
        output_widget.config(state='disabled')
        output_widget.see(tk.END)

    threading.Thread(target=stream_response, daemon=True).start()

def create_keyword_difficulty_chart(df):
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#2b2b2b")
    ax.set_facecolor("#3c3c3c")
    df_sorted = df.sort_values('difficulty')
    bars = ax.bar(df_sorted['keyword'], df_sorted['difficulty'], color='#1e90ff')
    for i, bar in enumerate(bars):
        difficulty = df_sorted['difficulty'].iloc[i]
        if difficulty < 30:
            bar.set_color('#32cd32')
        elif difficulty < 60:
            bar.set_color('#ffa500')
        else:
            bar.set_color('#ff4500')
    ax.set_xlabel('Keywords', color="#ffffff")
    ax.set_ylabel('Difficulty (0-100)', color="#ffffff")
    ax.set_title('Keyword Difficulty Analysis', color="#ffffff")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right', color="#ffffff")
    ax.grid(axis='y', linestyle='--', alpha=0.3, color="#ffffff")
    ax.tick_params(axis='both', colors="#ffffff")
    plt.tight_layout()
    return fig

def create_volume_competition_chart(df):
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#2b2b2b")
    ax.set_facecolor("#3c3c3c")
    scatter = ax.scatter(
        df['competition'],
        df['volume'],
        s=df['cpc'] * 30,
        alpha=0.6,
        c=df['difficulty'],
        cmap='RdYlGn_r'
    )
    for i, txt in enumerate(df['keyword']):
        ax.annotate(txt, (df['competition'].iloc[i], df['volume'].iloc[i]), fontsize=9, ha='center', color="#ffffff")
    ax.set_xlabel('Competition (0-1)', color="#ffffff")
    ax.set_ylabel('Monthly Search Volume', color="#ffffff")
    ax.set_title('Volume vs Competition', color="#ffffff")
    plt.colorbar(scatter, label='Keyword Difficulty').ax.yaxis.set_tick_params(color="#ffffff")
    ax.grid(True, linestyle='--', alpha=0.3, color="#ffffff")
    ax.tick_params(axis='both', colors="#ffffff")
    plt.tight_layout()
    return fig

def create_rankings_chart():
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#2b2b2b")
    ax.set_facecolor("#3c3c3c")
    keywords = global_state["rankings_data"]["keywords"]
    positions = global_state["rankings_data"]["positions"]
    bars = ax.barh(keywords, positions, color='#1e90ff')
    for i, bar in enumerate(bars):
        position = positions[i]
        if position <= 3:
            bar.set_color('#32cd32')
        elif position <= 10:
            bar.set_color('#ffa500')
        else:
            bar.set_color('#ff4500')
    ax.invert_yaxis()
    ax.set_xlabel('Google Position', color="#ffffff")
    ax.set_title('Keyword Rankings', color="#ffffff")
    ax.grid(axis='x', linestyle='--', alpha=0.3, color="#ffffff")
    for i, position in enumerate(positions):
        ax.text(position + 0.5, i, str(position), va='center', color="#ffffff")
    ax.set_xlim(0, max(positions) + 5)
    ax.tick_params(axis='both', colors="#ffffff")
    plt.tight_layout()
    return fig

# Tkinter UI
class SEOAssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SEO Smart Assistant")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a1a")

        # Custom styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background="#1a1a1a", tabmargins=0)
        style.configure("TNotebook.Tab", background="#2b2b2b", foreground="#ffffff", padding=[12, 6], font=("Arial", 12, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#007bff")], foreground=[("selected", "#ffffff")])
        style.configure("TButton", background="#007bff", foreground="#ffffff", font=("Arial", 12), padding=8, bordercolor="#555555")
        style.map("TButton", background=[("active", "#0056b3")], foreground=[("active", "#ffffff")])
        style.configure("TEntry", fieldbackground="#2b2b2b", foreground="#ffffff", font=("Arial", 12), padding=5)
        style.configure("TLabel", background="#1a1a1a", foreground="#ffffff", font=("Arial", 12))
        style.configure("Custom.TButton", background="#343a40", foreground="#ffffff", font=("Arial", 11), padding=6)
        style.map("Custom.TButton", background=[("active", "#495057")])

        # Main frame
        self.main_frame = tk.Frame(self.root, bg="#1a1a1a")
        self.main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        # Sidebar
        self.sidebar_frame = tk.Frame(self.main_frame, bg="#2b2b2b", width=300, relief="raised", borderwidth=1)
        self.sidebar_frame.pack(side="right", fill="y", padx=(10, 0))

        # Sidebar header
        tk.Label(self.sidebar_frame, text="SEO Dashboard", font=("Arial", 18, "bold"), bg="#2b2b2b", fg="#007bff").pack(pady=(15, 10))
        
        # Profile section
        self.profile_frame = tk.Frame(self.sidebar_frame, bg="#2b2b2b")
        self.profile_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(self.profile_frame, text="Profile", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="#ffffff").pack(anchor="w")
        self.profile_label = tk.Label(self.profile_frame, text="Level: Beginner", font=("Arial", 12), bg="#2b2b2b", fg="#adb5bd")
        self.profile_label.pack(anchor="w", pady=2)
        
        tk.Label(self.profile_frame, text="Interests:", font=("Arial", 12), bg="#2b2b2b", fg="#ffffff").pack(anchor="w", pady=2)
        self.interests_label = tk.Label(self.profile_frame, text="None", font=("Arial", 12), bg="#2b2b2b", fg="#adb5bd", wraplength=260)
        self.interests_label.pack(anchor="w", pady=2)
        
        tk.Label(self.profile_frame, text="Keywords:", font=("Arial", 12), bg="#2b2b2b", fg="#ffffff").pack(anchor="w", pady=2)
        self.keywords_label = tk.Label(self.profile_frame, text="None", font=("Arial", 12), bg="#2b2b2b", fg="#adb5bd", wraplength=260)
        self.keywords_label.pack(anchor="w", pady=2)

        # Tools section
        tk.Label(self.sidebar_frame, text="Tools", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="#ffffff").pack(anchor="w", padx=10, pady=(15, 5))
        self.url_input = ttk.Entry(self.sidebar_frame, width=25)
        self.url_input.insert(0, "https://example.com")
        self.url_input.pack(padx=10, pady=5)
        
        ttk.Button(self.sidebar_frame, text="Analyze URL", command=self.analyze_url).pack(padx=10, pady=5)
        ttk.Button(self.sidebar_frame, text="Clear History", command=self.clear_history).pack(padx=10, pady=5)
        ttk.Button(self.sidebar_frame, text="Reset Profile", command=self.reset_profile).pack(padx=10, pady=5)

        # Tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill="both", expand=True)

        # Chat Tab
        self.chat_frame = tk.Frame(self.notebook, bg="#1a1a1a")
        self.notebook.add(self.chat_frame, text="Chat")
        
        # Chat header
        tk.Label(self.chat_frame, text="SEO Assistant Chat", font=("Arial", 16, "bold"), bg="#1a1a1a", fg="#ffffff").pack(pady=(10, 5))
        
        # Chat output with border
        self.chat_output_frame = tk.Frame(self.chat_frame, bg="#2b2b2b", relief="sunken", borderwidth=1)
        self.chat_output_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.chat_output = scrolledtext.ScrolledText(
            self.chat_output_frame, height=25, width=100, font=("Arial", 12), 
            bg="#2b2b2b", fg="#ffffff", insertbackground="#ffffff", wrap=tk.WORD, 
            state='disabled', borderwidth=0, highlightthickness=0
        )
        self.chat_output.pack(padx=5, pady=5, fill="both", expand=True)
        
        self.chat_output.tag_configure("user", foreground="#007bff", font=("Arial", 12, "bold"))
        self.chat_output.tag_configure("bot", foreground="#e9ecef")
        self.chat_output.tag_configure("suggestions", foreground="#28a745", font=("Arial", 12, "italic"))
        self.chat_output.tag_configure("error", foreground="#dc3545")

        # Load previous messages
        self.load_previous_messages()

        # Chat input area
        self.chat_input_frame = tk.Frame(self.chat_frame, bg="#1a1a1a")
        self.chat_input_frame.pack(fill="x", padx=10, pady=5)
        
        self.chat_input = ttk.Entry(self.chat_input_frame, width=80)
        self.chat_input.pack(side="left", padx=(0, 5))
        self.chat_input.bind("<Return>", lambda e: self.send_chat())
        
        ttk.Button(self.chat_input_frame, text="Send", command=self.send_chat).pack(side="left")
        
        # Suggestions buttons
        self.suggestions_frame = tk.Frame(self.chat_frame, bg="#1a1a1a")
        self.suggestions_frame.pack(fill="x", padx=10, pady=5)
        self.suggestion_buttons = []
        for i in range(3):
            btn = ttk.Button(
                self.suggestions_frame, 
                text="", 
                style="Custom.TButton",
                command=lambda x=i: self.use_suggestion(x),
                state='disabled'
            )
            btn.pack(fill="x", pady=2)
            self.suggestion_buttons.append(btn)

        # Keyword Analysis Tab
        self.keyword_frame = tk.Frame(self.notebook, bg="#1a1a1a")
        self.notebook.add(self.keyword_frame, text="Keyword Analysis")
        
        tk.Label(self.keyword_frame, text="Keyword Analysis", font=("Arial", 16, "bold"), bg="#1a1a1a", fg="#ffffff").pack(pady=(10, 5))
        
        self.keyword_input = ttk.Entry(self.keyword_frame, width=60)
        self.keyword_input.insert(0, "e.g., local seo, site optimization, backlinks")
        self.keyword_input.pack(padx=10, pady=5)
        
        ttk.Button(self.keyword_frame, text="Analyze Keywords", command=self.analyze_keywords).pack(pady=5)
        
        self.keyword_output = scrolledtext.ScrolledText(
            self.keyword_frame, height=10, width=100, font=("Arial", 12), 
            bg="#2b2b2b", fg="#ffffff", state='disabled'
        )
        self.keyword_output.pack(padx=10, pady=5)
        
        self.keyword_canvas = tk.Canvas(self.keyword_frame, height=350, bg="#1a1a1a", highlightthickness=0)
        self.keyword_canvas.pack(fill="both", padx=10, pady=5)

        # Content Analysis Tab
        self.content_frame = tk.Frame(self.notebook, bg="#1a1a1a")
        self.notebook.add(self.content_frame, text="Content Analysis")
        
        tk.Label(self.content_frame, text="Content Analysis", font=("Arial", 16, "bold"), bg="#1a1a1a", fg="#ffffff").pack(pady=(10, 5))
        
        self.content_input = scrolledtext.ScrolledText(
            self.content_frame, height=10, width=100, font=("Arial", 12), 
            bg="#2b2b2b", fg="#ffffff", insertbackground="#ffffff"
        )
        self.content_input.pack(padx=10, pady=5)
        
        ttk.Button(self.content_frame, text="Analyze Content", command=self.analyze_content).pack(pady=5)
        
        self.content_output = scrolledtext.ScrolledText(
            self.content_frame, height=10, width=100, font=("Arial", 12), 
            bg="#2b2b2b", fg="#ffffff", state='disabled'
        )
        self.content_output.pack(padx=10, pady=5)

        # Rankings Tab
        self.rankings_frame = tk.Frame(self.notebook, bg="#1a1a1a")
        self.notebook.add(self.rankings_frame, text="Rankings")
        
        tk.Label(self.rankings_frame, text="Keyword Rankings", font=("Arial", 16, "bold"), bg="#1a1a1a", fg="#ffffff").pack(pady=(10, 5))
        
        self.rankings_canvas = tk.Canvas(self.rankings_frame, height=350, bg="#1a1a1a", highlightthickness=0)
        self.rankings_canvas.pack(fill="both", padx=10, pady=5)
        
        self.rankings_output = scrolledtext.ScrolledText(
            self.rankings_frame, height=10, width=100, font=("Arial", 12), 
            bg="#2b2b2b", fg="#ffffff", state='disabled'
        )
        self.rankings_output.pack(padx=10, pady=5)
        
        self.update_rankings()
        self.update_profile()

    def load_previous_messages(self):
        self.chat_output.config(state='normal')
        self.chat_output.delete(1.0, tk.END)
        messages = load_messages()
        for msg in messages:
            if msg["role"] == "user":
                self.chat_output.insert(tk.END, f"You: {msg['content']}\n\n", "user")
            else:
                self.chat_output.insert(tk.END, f"Assistant: {msg['content']}\n", "bot")
                suggestion_match = re.search(r"SUGGESTIONS: (.*?)(?:\n|$)", msg['content'])
                if suggestion_match:
                    suggestions_text = suggestion_match.group(1)
                    suggestions = [s.strip() for s in suggestions_text.split('|')]
                    self.chat_output.insert(tk.END, "\n**Suggestions:**\n" + "\n".join(suggestions) + "\n", "suggestions")
        self.chat_output.config(state='disabled')
        self.chat_output.see(tk.END)

    def send_chat(self):
        query = self.chat_input.get().strip()
        if query:
            self.chat_input.delete(0, tk.END)
            generate_response(query, self.chat_output, self.suggestion_buttons)
            self.update_profile()

    def use_suggestion(self, index):
        if index < len(global_state["suggestions"]):
            query = global_state["suggestions"][index]
            self.chat_input.delete(0, tk.END)
            self.chat_input.insert(0, query)
            self.send_chat()

    def analyze_keywords(self):
        keywords = self.keyword_input.get().strip()
        if not keywords or keywords == "e.g., local seo, site optimization, backlinks":
            messagebox.showwarning("Warning", "Please enter valid keywords.")
            return
        df = analyze_keywords(keywords)
        global_state["analyzed_keywords"] = df
        self.keyword_output.config(state='normal')
        self.keyword_output.delete(1.0, tk.END)
        self.keyword_output.insert(tk.END, df.to_string())
        self.keyword_output.config(state='disabled')
        
        for widget in self.keyword_canvas.winfo_children():
            widget.destroy()
        
        fig = create_keyword_difficulty_chart(df)
        if fig:
            canvas = FigureCanvasTkAgg(fig, master=self.keyword_canvas)
            canvas.draw()
            canvas.get_tk_widget().pack()
        self.update_profile()

    def analyze_content(self):
        content = self.content_input.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "Please enter content to analyze.")
            return
        analysis = analyze_content_seo(content)
        global_state["analyzed_content"] = analysis
        self.content_output.config(state='normal')
        self.content_output.delete(1.0, tk.END)
        self.content_output.insert(tk.END, f"Word Count: {analysis['word_count']}\n")
        self.content_output.insert(tk.END, f"Sentence Count: {analysis['sentence_count']}\n")
        self.content_output.insert(tk.END, f"Average Sentence Length: {analysis['avg_sentence_length']}\n")
        self.content_output.insert(tk.END, f"Readability Score: {analysis['readability_score']}\n")
        self.content_output.insert(tk.END, f"Primary Keyword: {analysis['primary_keyword'] or 'None'}\n")
        self.content_output.insert(tk.END, f"Keyword Density: {analysis['keyword_density']}%\n")
        self.content_output.insert(tk.END, f"Heading Tags: {analysis['h_tags']}\n")
        self.content_output.insert(tk.END, f"Paragraphs: {analysis['paragraphs']}\n")
        if analysis["issues"]:
            self.content_output.insert(tk.END, "\nIssues:\n")
            for issue in analysis["issues"]:
                self.content_output.insert(tk.END, f"- {issue['message']}\n")
        self.content_output.config(state='disabled')

    def analyze_url(self):
        url = self.url_input.get().strip()
        if not url:
            messagebox.showwarning("Warning", "Please enter a valid URL.")
            return
        report = analyze_url(url)
        if "error" in report:
            messagebox.showerror("Error", report["error"])
        else:
            messagebox.showinfo("URL Analysis",
                                f"Title: {report['title']}\n"
                                f"Meta Description: {report['meta_description']}\n"
                                f"Internal Links: {report['internal_links']}\n"
                                f"External Links: {report['external_links']}\n"
                                f"Page Size: {report['page_size_kb']} KB")

    def update_profile(self):
        level_text = {"beginner": "Beginner", "intermediate": "Intermediate", "expert": "Expert"}
        self.profile_label.config(text=f"Level: {level_text[global_state['user_profile']['seo_level']]}")
        interests = ", ".join(global_state["user_profile"]["interests"]) or "None"
        self.interests_label.config(text=interests)
        keywords = ", ".join(global_state["user_profile"]["keywords_analyzed"][:5]) or "None"
        self.keywords_label.config(text=keywords)

    def update_rankings(self):
        for widget in self.rankings_canvas.winfo_children():
            widget.destroy()
        fig = create_rankings_chart()
        if fig:
            canvas = FigureCanvasTkAgg(fig, master=self.rankings_canvas)
            canvas.draw()
            canvas.get_tk_widget().pack()
        self.rankings_output.config(state='normal')
        self.rankings_output.delete(1.0, tk.END)
        df = pd.DataFrame({
            "Keyword": global_state["rankings_data"]["keywords"],
            "Position": global_state["rankings_data"]["positions"]
        })
        self.rankings_output.insert(tk.END, df.to_string())
        self.rankings_output.config(state='disabled')

    def clear_history(self):
        global_state["messages"] = []
        save_messages()
        self.chat_output.config(state='normal')
        self.chat_output.delete(1.0, tk.END)
        self.chat_output.config(state='disabled')
        for btn in self.suggestion_buttons:
            btn.config(text="", state='disabled')

    def reset_profile(self):
        global_state["user_profile"] = {
            "seo_level": "beginner",
            "interests": [],
            "writing_style": "standard",
            "language_preference": "formal",
            "sentiment_history": [],
            "query_history": [],
            "tools_used": [],
            "keywords_analyzed": []
        }
        global_state["analyzed_keywords"] = pd.DataFrame()
        self.update_profile()

if __name__ == "__main__":
    root = tk.Tk()
    app = SEOAssistantApp(root)
    root.mainloop()