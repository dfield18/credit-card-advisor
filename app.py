import streamlit as st
import pandas as pd
from openai import OpenAI
import json

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

# Function to get recommendations - FASTEST VERSION (Single API Call)
def get_card_recommendations(user_query, card_data, num_recommendations=5):
    try:
        # FAST keyword-based pre-filtering (no AI needed - instant)
        query_lower = user_query.lower()
        keywords = {
            'travel': ['travel', 'flight', 'airline', 'hotel', 'vacation', 'trip'],
            'dining': ['dining', 'restaurant', 'food', 'eat', 'meal'],
            'grocery': ['grocery', 'groceries', 'supermarket', 'food shopping'],
            'gas': ['gas', 'fuel', 'station', 'petrol'],
            'cash': ['cash', 'back', 'cashback', 'rebate'],
            'business': ['business', 'corporate', 'company'],
            'luxury': ['luxury', 'premium', 'exclusive', 'elite'],
            'no fee': ['no fee', 'no annual', 'free', 'zero fee'],
            'rewards': ['rewards', 'points', 'miles'],
            'bonus': ['bonus', 'sign up', 'welcome']
        }
        
        # Find relevant keywords in query
        matched_keywords = []
        for category, terms in keywords.items():
            if any(term in query_lower for term in terms):
                matched_keywords.extend(terms)
        
        # If we found keywords, use them for fast filtering
        if matched_keywords:
            pattern = '|'.join(matched_keywords)
            filtered_cards = card_data[
                card_data['card_name'].str.lower().str.contains(pattern, na=False, case=False) |
                card_data['perks_summary'].str.lower().str.contains(pattern, na=False, case=False)
            ].head(30)
        else:
            # Fallback: take first 30 cards
            filtered_cards = card_data.head(30)
        
        # If still no matches, use all data (limited)
        if len(filtered_cards) == 0:
            filtered_cards = card_data.head(30)
        
        # Prepare data for single AI call
        cards_text = "\n\n".join([
            f"{i+1}. {row['card_name']}: {row['perks_summary']}"
            for i, row in filtered_cards.iterrows()
        ])
        
        # Single AI call with concise prompt
        prompt = f"""User needs: {user_query}

Cards (showing {len(filtered_cards)} most relevant):
{cards_text}

Return JSON with top {num_recommendations} recommendations:
{{"recommendations": [{{"card_name": "name", "why_recommended": "brief reason", "key_perks": "main perks"}}]}}"""

        # Single fast API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        recommendations = json.loads(result)
        
        # Handle different response formats
        if isinstance(recommendations, dict):
            if 'recommendations' in recommendations:
                recommendations = recommendations['recommendations']
            elif 'cards' in recommendations:
                recommendations = recommendations['cards']
            else:
                # If dict but no known key, try to extract first list value
                for key, value in recommendations.items():
                    if isinstance(value, list):
                        recommendations = value
                        break
        
        # Ensure we have a list
        if not isinstance(recommendations, list):
            recommendations = [recommendations] if recommendations else []
        
        return recommendations[:num_recommendations]
    
    except json.JSONDecodeError as e:
        st.error(f"Error parsing response: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
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
