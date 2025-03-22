import re
import pandas as pd
import streamlit as st
from datetime import datetime
import nltk
import collections
import emoji

import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from textblob import TextBlob

# Download NLTK stopwords if not already present
nltk.download('stopwords')

# ---------------------
# Helper Functions
# ---------------------

def parse_line(line):
    """
    Parse a single WhatsApp chat line.
    Expected format: "12/31/20, 9:34 PM - John Doe: Happy New Year!"
    Adjust regex if your format differs.
    """
    pattern = r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}\s[APM]{2})\s-\s([^:]+):\s(.*)$'
    match = re.match(pattern, line)
    if match:
        date_str, time_str, sender, message = match.groups()
        try:
            timestamp = datetime.strptime(f'{date_str} {time_str}', '%m/%d/%y %I:%M %p')
        except ValueError:
            # Alternative date format (if needed)
            timestamp = datetime.strptime(f'{date_str} {time_str}', '%d/%m/%y %I:%M %p')
        return timestamp, sender, message
    return None

def extract_emojis(text):
    """Return a list of emojis found in text."""
    return [c for c in text if c in emoji.EMOJI_DATA]

def analyze_sentiment(message):
    """Return sentiment polarity from TextBlob analysis."""
    analysis = TextBlob(message)
    return analysis.sentiment.polarity  # Value between -1 and 1

def get_sentiment_label(polarity):
    """Label sentiment as Positive, Negative, or Neutral using simple thresholds."""
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# ---------------------
# Streamlit App
# ---------------------

st.title("WhatsApp Chat Analyzer")

# File uploader widget
uploaded_file = st.file_uploader("Upload your WhatsApp chat file", type=["txt"])
if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    lines = content.splitlines()
    parsed_data = [parse_line(line) for line in lines if parse_line(line)]
    
    if not parsed_data:
        st.error("No valid chat data found. Please check the file format.")
    else:
        # Create DataFrame from parsed data
        df = pd.DataFrame(parsed_data, columns=['timestamp', 'sender', 'message'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['message_length'] = df['message'].apply(len)
        df['sentiment'] = df['message'].apply(analyze_sentiment)
        df['sentiment_label'] = df['sentiment'].apply(get_sentiment_label)
        
        # ---------------------
        # Sidebar Filtering Options
        # ---------------------
        st.sidebar.header("Filtering Options")
        senders = df['sender'].unique().tolist()
        selected_sender = st.sidebar.selectbox("Select sender (or All)", ["All"] + senders)
        if selected_sender != "All":
            df = df[df['sender'] == selected_sender]
        
        min_date = df['date'].min()
        max_date = df['date'].max()
        date_range = st.sidebar.date_input("Select date range", [min_date, max_date])
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        st.subheader("Overview Statistics")
        st.write(f"Total messages: {len(df)}")
        st.write(f"Unique senders: {df['sender'].nunique()}")
        
        # ---------------------
        # Interactive Conversation Timeline
        # ---------------------
        timeline_df = df.groupby('date').size().reset_index(name='count')
        fig1 = px.line(
            timeline_df, 
            x='date', 
            y='count', 
            title='Messages per Day',
            labels={'date': 'Date', 'count': 'Message Count'},
            markers=True
        )
        fig1.update_traces(line_color="#8d6e63", marker_color="#8d6e63")
        st.plotly_chart(fig1, use_container_width=True)
        
        # ---------------------
        # Interactive Message Length Analysis
        # ---------------------
        fig2 = px.histogram(
            df, 
            x='message_length', 
            nbins=30, 
            title='Message Length Distribution',
            labels={'message_length': 'Message Length (characters)'},
            color_discrete_sequence=["#A1887F"]
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # ---------------------
        # Activity by Hour of the Day (Area Chart)
        # ---------------------
        hour_counts = df.groupby('hour').size().reset_index(name='count')
        fig_hour = px.area(
            hour_counts, 
            x='hour', 
            y='count', 
            title='Activity by Hour of the Day',
            labels={'hour': 'Hour of Day', 'count': 'Message Count'}
        )
        # Update the area chart to use brown tones
        fig_hour.update_traces(line_color="#8d6e63", fillcolor="#d7ccc8")
        st.plotly_chart(fig_hour, use_container_width=True)
        
        # ---------------------
        # Frequently Used Emojis
        # ---------------------
        st.subheader("Frequently Used Emojis")
        all_emojis = []
        df['message'].apply(lambda x: all_emojis.extend(extract_emojis(x)))
        emoji_counts = collections.Counter(all_emojis)
        if emoji_counts:
            emoji_df = pd.DataFrame(emoji_counts.items(), columns=['emoji', 'count']).sort_values(by='count', ascending=False)
            st.dataframe(emoji_df)
            fig3 = px.bar(
                emoji_df.head(10), 
                x='emoji', 
                y='count', 
                title='Top 10 Emojis',
                labels={'emoji': 'Emoji', 'count': 'Count'},
                color_discrete_sequence=["#5d4037"]
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.write("No emojis found in the chat.")
        
        # ---------------------
        # Activity by Day of the Week
        # ---------------------
        st.subheader("Activity by Day of the Week")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df['day_of_week'].value_counts().reindex(day_order)
        day_df = pd.DataFrame({'day_of_week': day_counts.index, 'count': day_counts.values})
        fig4 = px.bar(
            day_df, 
            x='day_of_week', 
            y='count', 
            title='Activity by Day of the Week',
            labels={'day_of_week': 'Day of Week', 'count': 'Message Count'},
            color_discrete_sequence=["#8d6e63"]
        )
        st.plotly_chart(fig4, use_container_width=True)
        
        # ---------------------
        # Interactive Sentiment Analysis with Brown Shades
        # ---------------------
        sentiment_counts = df['sentiment_label'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']
        fig5 = px.pie(
            sentiment_counts,
            values='count',
            names='sentiment',
            title='Sentiment Distribution',
            color='sentiment',
            color_discrete_map={
                "Positive": "#D7CCC8",  # Light brown
                "Neutral": "#A1887F",   # Medium brown
                "Negative": "#5D4037"   # Dark brown
            }
        )
        st.plotly_chart(fig5, use_container_width=True)
        
        # ---------------------
        # Word Cloud with Stopword Removal (Static Image)
        # ---------------------
        st.subheader("Word Cloud")
        stop_words = set(stopwords.words('english')).union(STOPWORDS)
        additional_stopwords = {"media", "omitted", "http", "https", "www"}
        stop_words = stop_words.union(additional_stopwords)
        text = " ".join(df['message'].tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              stopwords=stop_words,
                              colormap="copper").generate(text)
        fig_wc = wordcloud.to_array()
        st.image(fig_wc, use_container_width=True)
        
        # ---------------------
        # Interactive Chat Exploration
        # ---------------------
        st.subheader("Interactive Chat Exploration")
        unique_dates = sorted(df['date'].unique())
        selected_date = st.selectbox("Select a conversation date", unique_dates)
        chat_subset = df[df['date'] == selected_date].sort_values("timestamp")
        st.write(f"Displaying {len(chat_subset)} messages from {selected_date}:")
        for idx, row in chat_subset.iterrows():
            st.markdown(f"**{row['sender']}**: {row['message']}")
