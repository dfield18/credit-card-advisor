import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import re

# Page configuration
st.set_page_config(
    page_title="Credit Card Advisor",
    page_icon="💳",
    layout="wide"
)

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please add it to Streamlit secrets.")
        st.stop()
    return OpenAI(api_key=api_key)

client = get_openai_client()

# Load credit card data
@st.cache_data
def load_card_data():
    try:
        df = pd.read_csv('credit_cards.csv')
        if 'card_name' not in df.columns or 'perks_summary' not in df.columns:
            st.error("CSV must contain 'card_name' and 'perks_summary' columns")
            st.stop()
        return df
    except FileNotFoundError:
        st.error("credit_cards.csv file not found. Please upload your credit card data.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

df = load_card_data()

# Normalize query for better cache matching
def normalize_query(query):
    """Convert query to normalized form - similar queries will match"""
    # Lowercase and remove punctuation
    normalized = query.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    # Remove common stopwords
    stopwords = {'what', 'is', 'the', 'a', 'an', 'for', 'which', 'should', 'i', 'get', 'me', 'my', 'whats', 'best', 'good', 'top'}
    words = [w for w in normalized.split() if w not in stopwords]
    # Sort words for consistency ("travel card" = "card travel")
    return ' '.join(sorted(words))

# ULTRA-FAST keyword filtering with caching
@st.cache_data(ttl=3600)
def smart_filter_cards(query_lower, _card_data):
    """Fast keyword-based filtering with caching"""
    keywords = {
        'travel': ['travel', 'flight', 'airline', 'hotel', 'vacation', 'trip'],
        'dining': ['dining', 'restaurant', 'food', 'eat', 'meal'],
        'grocery': ['grocery', 'groceries', 'supermarket'],
        'gas': ['gas', 'fuel', 'station', 'petrol'],
        'cash': ['cash back', 'cashback', 'rebate'],
        'business': ['business', 'corporate'],
        'luxury': ['luxury', 'premium', 'exclusive'],
        'fee': ['no fee', 'no annual', 'free', 'zero'],
        'rewards': ['rewards', 'points', 'miles'],
        'bonus': ['bonus', 'sign up', 'welcome']
    }
    
    matched_keywords = []
    for category, terms in keywords.items():
        if any(term in query_lower for term in terms):
            matched_keywords.extend(terms)
    
    if matched_keywords:
        pattern = '|'.join(matched_keywords)
        mask = (
            _card_data['card_name'].str.lower().str.contains(pattern, na=False, case=False, regex=True) |
            _card_data['perks_summary'].str.lower().str.contains(pattern, na=False, case=False, regex=True)
        )
        filtered = _card_data[mask].head(10)  # Reduced to 10 for speed
        if len(filtered) > 0:
            return filtered
    
    return _card_data.head(10)  # Reduced to 10 for speed

# Cache AI responses based on normalized queries
@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours
def get_ai_recommendations(normalized_query, cards_text, num_recommendations):
    """Cached AI call - similar queries return instantly"""
    
    prompt = f"""Query: {normalized_query}

Cards:
{cards_text}

JSON format:
{{"recs":[{{"name":"card name","why":"1 sentence","perks":"key benefits"}}]}}

Top {num_recommendations} only."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=400,  # Reduced for even faster response
        top_p=0.1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content

# Main recommendation function
def get_card_recommendations(user_query, card_data, num_recommendations=5):
    try:
        # Normalize query for cache matching
        normalized = normalize_query(user_query)
        query_lower = user_query.lower()
        
        # Step 1: Fast filter (cached)
        filtered_cards = smart_filter_cards(query_lower, card_data)
        
        # Step 2: Prepare minimal data
        cards_list = []
        for _, row in filtered_cards.iterrows():
            perk_short = row['perks_summary'][:150] + "..." if len(row['perks_summary']) > 150 else row['perks_summary']
            cards_list.append(f"{row['card_name']}: {perk_short}")
        
        cards_text = "\n".join(cards_list)
        
        # Step 3: Get AI recommendations (CACHED - instant for similar queries!)
        result = get_ai_recommendations(normalized, cards_text, num_recommendations)
        
        recommendations = json.loads(result)
        
        # Parse response
        if isinstance(recommendations, dict):
            if 'recs' in recommendations:
                recommendations = recommendations['recs']
            elif 'recommendations' in recommendations:
                recommendations = recommendations['recommendations']
            elif 'cards' in recommendations:
                recommendations = recommendations['cards']
            else:
                for key, value in recommendations.items():
                    if isinstance(value, list):
                        recommendations = value
                        break
        
        if not isinstance(recommendations, list):
            recommendations = [recommendations] if recommendations else []
        
        # Normalize keys
        normalized_recs = []
        for rec in recommendations[:num_recommendations]:
            normalized_recs.append({
                'card_name': rec.get('name') or rec.get('card_name', 'Unknown'),
                'why_recommended': rec.get('why') or rec.get('why_recommended', 'Good match'),
                'key_perks': rec.get('perks') or rec.get('key_perks', 'See card details')
            })
        
        return normalized_recs
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return []

# App UI
st.title("💳 Credit Card Recommendation Assistant")
st.markdown("Ask me anything about credit cards and I'll recommend the best options for you!")

# Sidebar
with st.sidebar:
    st.header("Settings")
    num_recs = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    st.markdown("---")
    st.markdown(f"**Total Cards in Database:** {len(df)}")
    st.markdown("---")
    st.markdown("### Example Questions:")
    st.markdown("- What's the best card for travel rewards?")
    st.markdown("- I spend a lot on groceries, which card should I get?")
    st.markdown("- Best card for cash back on gas?")
    st.markdown("- Which card has the best sign-up bonus?")
    st.markdown("---")
    st.markdown("💡 **Tip**: Similar questions get instant cached results!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about credit cards..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get recommendations
    with st.chat_message("assistant"):
        with st.spinner("Analyzing credit cards..."):
            recommendations = get_card_recommendations(prompt, df, num_recs)
            
            if recommendations:
                response_text = f"Based on your question, here are my top {len(recommendations)} recommendations:\n\n"
                
                for i, rec in enumerate(recommendations, 1):
                    card_name = rec.get('card_name', 'Unknown Card')
                    why = rec.get('why_recommended', 'No explanation provided')
                    perks = rec.get('key_perks', 'No perks listed')
                    
                    response_text += f"### {i}. {card_name}\n\n"
                    response_text += f"**Why this card?** {why}\n\n"
                    response_text += f"**Key Perks:** {perks}\n\n"
                    response_text += "---\n\n"
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                error_msg = "I couldn't generate recommendations. Please try rephrasing your question."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("*Powered by OpenAI GPT-4o-mini | Built with Streamlit*")
