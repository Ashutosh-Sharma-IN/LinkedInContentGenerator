# linkedin_ai_app.py - Updated with secure API key handling
import streamlit as st
import pandas as pd
import openai
import json
import os
from datetime import datetime
import plotly.express as px
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="LinkedIn AI Content Generator",
    page_icon="🚀",
    layout="wide"
)

class LinkedInAIApp:
    def __init__(self):
        self.creators = [
            "Adam Biddlecombe", "Allie K Miller", "Lawrence Ng", 
            "Greg Isenberg", "Rakesh Gohel", "Vaibhav Sisinty",
            "Max Rascher", "Ruben Hassid", "Valeria Pilkevich", 
            "Isabella Bedoya", "Paul Roetzer"
        ]
        
    def setup_sidebar(self):
        """Setup sidebar with configuration"""
        st.sidebar.title("⚙️ Configuration")
        
        # Try to get API key from environment first
        env_api_key = os.getenv("OPENAI_API_KEY")
        
        if env_api_key:
            st.sidebar.success("✅ API Key loaded from environment")
            api_key = env_api_key
            # Show masked version
            masked_key = env_api_key[:8] + "..." + env_api_key[-4:] if len(env_api_key) > 12 else "***"
            st.sidebar.text(f"Key: {masked_key}")
        else:
            st.sidebar.warning("⚠️ No API key found in .env file")
            api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password", key="api_key")
        
        # Model selection
        model_options = ["gpt-4o-mini", "gpt-4", "fine-tuned-model"]
        model = st.sidebar.selectbox("Select Model", model_options, key="model_select")
        
        # Tone options
        tone_options = ["professional", "casual", "thought-leader", "technical", "startup-focused"]
        tone = st.sidebar.selectbox("Content Tone", tone_options, key="tone_select")
        
        return api_key, model, tone
    
    # ... rest of your existing methods remain the same ...
    
    def process_reverse_prompts(self, df, api_key):
        """Process CSV and generate reverse prompts"""
        if not api_key:
            st.error("❌ API key is required for processing!")
            return df
            
        client = openai.OpenAI(api_key=api_key)
        
        progress_bar = st.progress(0)
        total_rows = len(df)
        
        for index, row in df.iterrows():
            if pd.notna(row['Post_Content']) and '[Paste' not in str(row['Post_Content']):
                try:
                    prompt = f"""
                    Analyze this LinkedIn post by {row['Author']} and create a reverse-engineered prompt.
                    
                    Post: "{row['Post_Content']}"
                    Theme: {row.get('Theme', 'General')}
                    
                    Create a clear prompt that could generate similar content focusing on:
                    - Writing style and tone specific to {row['Author']}
                    - Structure and engagement elements
                    - Target audience (AI practitioners, entrepreneurs, business leaders)
                    
                    Return only the prompt:
                    """
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                        temperature=0.3
                    )
                    
                    df.at[index, 'Reverse_Prompt'] = response.choices[0].message.content.strip()
                    
                except Exception as e:
                    df.at[index, 'Reverse_Prompt'] = f"Error: {str(e)}"
            
            # Update progress
            progress_bar.progress((index + 1) / total_rows)
        
        return df
    
    def generate_content_variations(self, prompt, api_key, model, tone, creator_style, num_variations):
        """Generate multiple content variations"""
        if not api_key:
            return [{"content": "Error: API key required", "style": "Error", "variation": 1}]
            
        client = openai.OpenAI(api_key=api_key)
        variations = []
        
        base_system_prompt = f"""
        You are a LinkedIn content creator specializing in applied AI and business transformation.
        Write engaging posts that focus on practical AI applications, tools, and insights.
        Use a {tone} tone and include relevant hashtags.
        Target audience: entrepreneurs, business leaders, AI practitioners.
        """
        
        if creator_style != "Mixed":
            base_system_prompt += f" Write in the style of {creator_style}, mimicking their voice and approach."
        
        progress_bar = st.progress(0)
        
        for i in range(num_variations):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": base_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=400,
                    temperature=0.7 + (i * 0.1)
                )
                
                variations.append({
                    'content': response.choices[0].message.content.strip(),
                    'style': creator_style,
                    'variation': i + 1
                })
                
            except Exception as e:
                variations.append({
                    'content': f"Error generating variation {i+1}: {str(e)}",
                    'style': 'Error',
                    'variation': i + 1
                })
            
            progress_bar.progress((i + 1) / num_variations)
        
        return variations

    # Include all your other methods from the previous app code...
    # (template_generator_tab, data_processor_tab, content_generator_tab, etc.)

def main():
    """Main app function"""
    st.title("🚀 LinkedIn AI Content Generator")
    st.markdown("*Transform top creator insights into your unique content strategy*")
    
    app = LinkedInAIApp()
    
    # Setup sidebar
    api_key, model, tone = app.setup_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Template", 
        "🤖 Process Data", 
        "✍️ Generate Content", 
        "🎨 Infographics", 
        "📈 Analytics"
    ])
    
    with tab1:
        app.template_generator_tab()
    
    with tab2:
        app.data_processor_tab()
    
    with tab3:
        app.content_generator_tab()
    
    with tab4:
        app.infographic_generator_tab()
    
    with tab5:
        app.analytics_tab()

if __name__ == "__main__":
    main()
