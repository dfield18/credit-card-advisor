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

# Function to get recommendations
def get_card_recommendations(user_query, card_data, num_recommendations=5):
    # Prepare card data for GPT
    cards_text = "\n\n".join([
        f"Card {i+1}: {row['card_name']}\nPerks: {row['perks_summary']}"
        for i, row in card_data.iterrows()
    ])
    
    system_prompt = f"""You are an expert credit card advisor. You have access to a database of {len(card_data)} credit cards.

Your task is to analyze the user's question and recommend the top {num_recommendations} credit cards that best match their needs.

For each recommendation:
1. Provide the exact card name
2. Explain why it's a good match for the user's specific query
3. Highlight the most relevant perks

Format your response as a JSON array with this structure:
[
  {{
    "card_name": "exact card name from database",
    "why_recommended": "explanation of why this card matches the user's needs",
    "key_perks": "most relevant perks for this user"
  }}
]

Be specific and focus on perks that directly address the user's question."""

    user_prompt = f"""User Question: {user_query}

Available Credit Cards:
{cards_text}

Please recommend the top {num_recommendations} credit cards for this user's needs. Return your response as a JSON array."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = response.choices[0].message.content
        recommendations = json.loads(result)
        
        # Handle if the response is wrapped in an object
        if isinstance(recommendations, dict):
            if 'recommendations' in recommendations:
                recommendations = recommendations['recommendations']
            elif 'cards' in recommendations:
                recommendations = recommendations['cards']
        
        return recommendations[:num_recommendations]
    
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
st.markdown("*Powered by OpenAI GPT-4 | Built with Streamlit*")