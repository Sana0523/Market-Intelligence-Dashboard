import streamlit as st
import folium
from streamlit_folium import st_folium
from gnews import GNews
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yake
import pandas as pd
import altair as alt
import concurrent.futures
from branca.element import Template, MacroElement

# --- Configuration ---
st.set_page_config(
    page_title="Market Intelligence Dashboard", 
    layout="wide", 
    initial_sidebar_state="auto" # 'auto' collapses sidebar on mobile
)

# Expanded Country List
COUNTRIES = {
    "US": {"coords": [37.0902, -95.7129], "code": "US"},
    "UK": {"coords": [54.0000, -2.0000], "code": "GB"},
    "India": {"coords": [20.5937, 78.9629], "code": "IN"},
    "Japan": {"coords": [36.2048, 138.2529], "code": "JP"},
    "Germany": {"coords": [51.1657, 10.4515], "code": "DE"},
    "China": {"coords": [35.8617, 104.1954], "code": "CN"},
    "France": {"coords": [46.2276, 2.2137], "code": "FR"},
    "Canada": {"coords": [56.1304, -106.3468], "code": "CA"}
}

# --- NLP Engines ---
analyzer = SentimentIntensityAnalyzer()
kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=3, features=None)

# --- Helper Functions ---

def get_sentiment_color(score):
    if score >= 0.05:
        return '#2ecc71' # Green
    elif score <= -0.05:
        return '#e74c3c' # Red
    else:
        return '#95a5a6' # Gray

def create_map_legend(map_object):
    """
    Injects a responsive CSS/HTML legend into the Folium map.
    Includes Media Queries for Mobile.
    """
    template = """
    {% macro html(this, kwargs) %}
    <style>
        .map-legend {
            position: fixed; 
            bottom: 30px; 
            left: 30px; 
            width: 160px; 
            height: 100px; 
            z-index:9999; 
            background-color: white; 
            opacity: 0.95;
            padding: 15px; 
            border-radius: 8px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 12px;
        }
        
        /* Mobile Specific Styles */
        @media (max-width: 768px) {
            .map-legend {
                bottom: 10px;
                left: auto;
                right: 10px; /* Move to right on mobile */
                width: 120px;
                height: auto;
                padding: 10px;
                font-size: 10px; /* Smaller font */
            }
            .legend-title {
                font-size: 12px !important;
                margin-bottom: 2px !important;
            }
            .legend-item {
                margin-bottom: 2px !important;
            }
        }
    </style>
    
    <div class="map-legend">
        <strong class="legend-title" style="font-size:14px; display:block; margin-bottom:5px;">Sentiment Legend</strong>
        <div class="legend-item" style="margin-bottom: 3px;"><i class="fa fa-circle" style="color:#2ecc71"></i>&nbsp; Positive</div>
        <div class="legend-item" style="margin-bottom: 3px;"><i class="fa fa-circle" style="color:#95a5a6"></i>&nbsp; Neutral</div>
        <div class="legend-item" style="margin-bottom: 3px;"><i class="fa fa-circle" style="color:#e74c3c"></i>&nbsp; Negative</div>
    </div>
    {% endmacro %}
    """
    macro = MacroElement()
    macro._template = Template(template)
    map_object.get_root().add_child(macro)

def analyze_country(args):
    """
    Function to be run in parallel for each country.
    """
    name, info, topic = args
    try:
        google_news = GNews(language='en', country=info['code'], max_results=10)
        news = google_news.get_news(topic)
        
        if not news:
            return None

        headlines = [item.get('title', '') for item in news]
        links = [item.get('url', '') for item in news]
        
        # Batch analysis
        compound_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        subjectivity_scores = [TextBlob(h).sentiment.subjectivity for h in headlines]
        
        avg_sent = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        avg_subj = sum(subjectivity_scores) / len(subjectivity_scores) if subjectivity_scores else 0
        
        # Keywords
        full_text = " ".join(headlines)
        keywords = [kw[0] for kw in kw_extractor.extract_keywords(full_text)]
        
        return {
            "Country": name,
            "Sentiment": avg_sent,
            "Subjectivity": avg_subj,
            "Volume": len(news),
            "Keywords": ", ".join(keywords),
            "Coords": info['coords'],
            "Headlines": headlines,
            "Links": links,
            "Scores": compound_scores
        }
    except Exception as e:
        return None

# --- Main UI ---

# Custom CSS for Mobile Optimization
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #000002;
    }
    
    /* Typography */
    h1 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        color: #2c3e50;
    }
    
    /* Mobile-First Adjustments */
    @media (max-width: 640px) {
        /* Reduce top padding on mobile */
        .block-container {
            padding-top: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        /* Make fonts slightly smaller on headers to prevent wrapping */
        h1 {
            font-size: 1.5rem;
        }
        /* Adjust metric values size */
        div[data-testid="stMetricValue"] {
            font-size: 20px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("Market Intelligence Dashboard")
st.markdown("Real-time NLP sentiment analysis of global financial news.")

# Sidebar
with st.sidebar:
    st.header("Analysis Parameters")
    topic = st.text_input("Search Topic", "Economy")
    analyze_btn = st.button("Generate Report", type="primary", use_container_width=True)
    
    st.markdown("---")
    with st.expander("Methodology Note"):
        st.markdown("""
        **Sentiment Score (VADER):**
        - Positive: > 0.05
        - Negative: < -0.05
        - Neutral: -0.05 to 0.05
        """)

# State Management
if "data" not in st.session_state:
    st.session_state.data = None

# Analysis Logic
if analyze_btn:
    with st.status("Processing Market Data...", expanded=True) as status:
        st.write("Initializing data pipelines...")
        task_args = [(name, info, topic) for name, info in COUNTRIES.items()]
        results = []
        st.write(f"Fetching headlines for: {topic}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            fetched_data = list(executor.map(analyze_country, task_args))
        st.session_state.data = [d for d in fetched_data if d is not None]
        status.update(label="Analysis Complete", state="complete", expanded=False)

# --- Dashboard Display ---
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)
    
    # 1. Executive Summary (Cards)
    g_avg = df['Sentiment'].mean()
    g_sub = df['Subjectivity'].mean()
    
    with st.container(border=True):
        st.subheader("Executive Summary")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Global Sentiment", f"{g_avg:.2f}", help="VADER Score")
        col_m2.metric("Subjectivity", f"{g_sub:.2f}", help="TextBlob Score")
        col_m3.metric("Topic", topic)
    
    # 2. Map and Chart Layout
    # Note: Streamlit automatically stacks columns on mobile.
    col_map, col_chart = st.columns([3, 2])

    with col_map:
        with st.container(border=True):
            st.subheader("Geospatial Sentiment")
            
            m = folium.Map(location=[25, 10], zoom_start=2, tiles="CartoDB positron")
            
            for index, row in df.iterrows():
                color = get_sentiment_color(row['Sentiment'])
                
                # Responsive Popup Content
                html = f"""
                <div style="font-family: 'Helvetica Neue', Arial; width: 180px;">
                    <h4 style="margin-bottom:5px; color:#333; font-size:14px;">{row['Country']}</h4>
                    <div style="font-size:11px; color:#555; margin-bottom:5px;">
                        <b>Score:</b> {row['Sentiment']:.2f}
                    </div>
                    <div style="font-size:10px; color:#777; border-top:1px solid #eee; padding-top:5px;">
                        {row['Keywords']}
                    </div>
                </div>
                """
                
                folium.CircleMarker(
                    location=row['Coords'],
                    radius=8 + (row['Volume'] * 1.2),
                    popup=folium.Popup(html, max_width=200),
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    tooltip=f"{row['Country']}"
                ).add_to(m)

            create_map_legend(m)
            # height=400 is better for mobile than 500
            st_folium(m, width=None, height=400, use_container_width=True)

    with col_chart:
        with st.container(border=True):
            st.subheader("Comparative Analysis")
            
            c = alt.Chart(df).mark_bar(cornerRadius=2).encode(
                x=alt.X('Country', sort='-y', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Sentiment', scale=alt.Scale(domain=[-1, 1]), axis=alt.Axis(title='Score')),
                color=alt.Color('Sentiment', scale=alt.Scale(domain=[-1, 1], range=['#e74c3c', '#95a5a6', '#2ecc71']), legend=None),
                tooltip=['Country', 'Sentiment', 'Subjectivity']
            ).properties(height=350) # Slightly shorter for mobile scrollability
            
            st.altair_chart(c, use_container_width=True)

    # 3. Detailed Data Table
    with st.container(border=True):
        st.subheader("News Feed")
        
        all_news_items = []
        for item in st.session_state.data:
            for h, l, s in zip(item['Headlines'], item['Links'], item['Scores']):
                all_news_items.append({
                    "Region": item['Country'],
                    "Sentiment": s,
                    "Headline": h,
                    "Link": l
                })
                
        df_news = pd.DataFrame(all_news_items)
        
        st.dataframe(
            df_news,
            column_config={
                "Link": st.column_config.LinkColumn("Link", display_text="Read"),
                "Sentiment": st.column_config.NumberColumn("Scr", format="%.2f"),
                "Headline": st.column_config.TextColumn("Headline", width="medium"), 
                "Region": st.column_config.TextColumn("Reg", width="small")
            },
            use_container_width=True,
            hide_index=True,
            height=400
        )

elif analyze_btn: 
    st.error("Data retrieval failed. Please check your internet connection or try a different topic.")
else:
    with st.container(border=True):
        st.info("Please configure the analysis parameters in the sidebar to begin.")