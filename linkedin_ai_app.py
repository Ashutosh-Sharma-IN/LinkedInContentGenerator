# linkedin_ai_app.py - Complete LinkedIn AI Content Generator
import streamlit as st
import pandas as pd
import openai
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import io

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
        model_options = ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
        model = st.sidebar.selectbox("Select Model", model_options, key="model_select")
        
        # Tone options
        tone_options = ["professional", "casual", "thought-leader", "technical", "startup-focused"]
        tone = st.sidebar.selectbox("Content Tone", tone_options, key="tone_select")
        
        return api_key, model, tone
    
    def template_generator_tab(self):
        """Generate CSV template for data collection"""
        st.header("📊 Generate Data Collection Template")
        
        st.markdown("""
        ### Create Your Data Collection Template
        This template helps you organize LinkedIn posts from top creators for analysis.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_posts = st.number_input("Number of posts to track", min_value=10, max_value=1000, value=50)
            include_metrics = st.checkbox("Include engagement metrics columns", value=True)
            include_themes = st.checkbox("Include theme categorization", value=True)
        
        with col2:
            selected_creators = st.multiselect(
                "Select creators to include",
                self.creators,
                default=self.creators[:5]
            )
        
        if st.button("Generate Template"):
            template_data = {
                'ID': range(1, num_posts + 1),
                'Author': ['[Paste Author Name]'] * num_posts,
                'Post_Content': ['[Paste Post Content]'] * num_posts,
                'Post_URL': ['[Paste Post URL]'] * num_posts,
                'Date_Posted': ['[YYYY-MM-DD]'] * num_posts,
            }
            
            if include_themes:
                template_data['Theme'] = ['[AI Tools/Business/Leadership/etc.]'] * num_posts
                template_data['Industry_Focus'] = ['[SaaS/Healthcare/Finance/etc.]'] * num_posts
            
            if include_metrics:
                template_data['Likes'] = [0] * num_posts
                template_data['Comments'] = [0]  * num_posts
                template_data['Shares'] = [0] * num_posts
                template_data['Engagement_Rate'] = [0.0] * num_posts
            
            template_data['Reverse_Prompt'] = [''] * num_posts
            
            df_template = pd.DataFrame(template_data)
            
            # Display template preview
            st.subheader("Template Preview")
            st.dataframe(df_template.head(), use_container_width=True)
            
            # Download button
            csv_buffer = io.StringIO()
            df_template.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download CSV Template",
                data=csv_string,
                file_name=f"linkedin_posts_template_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    def data_processor_tab(self):
        """Process uploaded CSV data"""
        st.header("🤖 Process Your LinkedIn Data")
        
        st.markdown("""
        ### Upload and Process Your Collected Data
        Upload your filled CSV template to generate reverse-engineered prompts.
        """)
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Data quality check
                st.subheader("📋 Data Quality Check")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_posts = len(df)
                    st.metric("Total Posts", total_posts)
                
                with col2:
                    filled_posts = len(df[df['Post_Content'].notna() & ~df['Post_Content'].str.contains('\[Paste', na=False)])
                    st.metric("Filled Posts", filled_posts)
                
                with col3:
                    completion_rate = (filled_posts / total_posts * 100) if total_posts > 0 else 0
                    st.metric("Completion Rate", f"{completion_rate:.1f}%")
                
                # Process reverse prompts
                if st.button("🔄 Generate Reverse Prompts") and completion_rate > 0:
                    api_key, model, tone = self.setup_sidebar()
                    processed_df = self.process_reverse_prompts(df, api_key)
                    
                    st.success("✅ Processing complete!")
                    
                    # Download processed file
                    csv_buffer = io.StringIO()
                    processed_df.to_csv(csv_buffer, index=False)
                    csv_string = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Processed Data",
                        data=csv_string,
                        file_name=f"processed_linkedin_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Store in session state for use in other tabs
                    st.session_state['processed_data'] = processed_df
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    def content_generator_tab(self):
        """Generate new content variations"""
        st.header("✍️ Generate LinkedIn Content")
        
        st.markdown("""
        ### Create New Content Based on Successful Patterns
        Generate variations of content using proven frameworks from top creators.
        """)
        
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["Manual Prompt", "Use Reverse Prompt from Data"]
        )
        
        if input_method == "Manual Prompt":
            prompt = st.text_area(
                "Enter your content prompt:",
                height=100,
                placeholder="Write a LinkedIn post about AI transformation in small businesses..."
            )
        else:
            if 'processed_data' in st.session_state:
                df = st.session_state['processed_data']
                available_prompts = df[df['Reverse_Prompt'].notna() & (df['Reverse_Prompt'] != '')]
                
                if len(available_prompts) > 0:
                    selected_post = st.selectbox(
                        "Select a reverse prompt:",
                        available_prompts.index,
                        format_func=lambda x: f"{available_prompts.loc[x, 'Author']} - {available_prompts.loc[x, 'Theme']}"
                    )
                    prompt = available_prompts.loc[selected_post, 'Reverse_Prompt']
                    st.text_area("Selected prompt:", value=prompt, height=100, disabled=True)
                else:
                    st.warning("No reverse prompts available. Please process data first.")
                    prompt = ""
            else:
                st.warning("No processed data available. Please process data first.")
                prompt = ""
        
        # Generation settings
        col1, col2 = st.columns(2)
        
        with col1:
            creator_style = st.selectbox("Creator Style", ["Mixed"] + self.creators)
            num_variations = st.slider("Number of variations", 1, 5, 3)
        
        with col2:
            content_type = st.selectbox("Content Type", [
                "Standard Post", "Carousel Post", "Video Script", "Newsletter"
            ])
        
        if st.button("🚀 Generate Content") and prompt:
            api_key, model, tone = self.setup_sidebar()
            
            variations = self.generate_content_variations(
                prompt, api_key, model, tone, creator_style, num_variations
            )
            
            st.subheader("Generated Content Variations")
            
            for i, variation in enumerate(variations):
                with st.expander(f"Variation {variation['variation']} - {variation['style']}", expanded=i==0):
                    st.markdown(variation['content'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.button("👍 Like", key=f"like_{i}")
                    with col2:
                        st.button("📋 Copy", key=f"copy_{i}")
                    with col3:
                        st.button("📤 Save", key=f"save_{i}")
    
    def infographic_generator_tab(self):
        """Generate infographic concepts"""
        st.header("🎨 Infographic Generator")
        
        st.markdown("""
        ### Create Visual Content Concepts
        Generate ideas for LinkedIn carousel posts and infographics.
        """)
        
        # Infographic type selection
        infographic_type = st.selectbox("Select infographic type:", [
            "Process Flow", "Comparison Chart", "Tips & Tricks", 
            "Statistics Showcase", "Timeline", "Checklist"
        ])
        
        topic = st.text_input("Enter your topic:", placeholder="AI tools for productivity")
        
        col1, col2 = st.columns(2)
        with col1:
            num_slides = st.slider("Number of slides", 3, 10, 5)
        with col2:
            color_scheme = st.selectbox("Color scheme", [
                "Professional Blue", "Tech Green", "Corporate Gray", 
                "Startup Orange", "Creative Purple"
            ])
        
        if st.button("🎨 Generate Infographic Concept") and topic:
            api_key, model, tone = self.setup_sidebar()
            
            if api_key:
                client = openai.OpenAI(api_key=api_key)
                
                prompt = f"""
                Create a detailed infographic concept for LinkedIn about: {topic}
                
                Type: {infographic_type}
                Number of slides: {num_slides}
                
                For each slide, provide:
                1. Slide title
                2. Key message/content
                3. Visual elements description
                4. Text layout suggestions
                
                Make it engaging for LinkedIn professionals interested in AI and business.
                """
                
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=800,
                        temperature=0.7
                    )
                    
                    st.subheader("Infographic Concept")
                    st.markdown(response.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"Error generating infographic: {str(e)}")
            else:
                st.error("API key required for infographic generation")
    
    def analytics_tab(self):
        """Display analytics and insights"""
        st.header("📈 Content Analytics & Insights")
        
        if 'processed_data' not in st.session_state:
            st.warning("No processed data available. Please process your LinkedIn data first.")
            return
        
        df = st.session_state['processed_data']
        
        # Filter data with valid metrics
        if 'Likes' in df.columns and 'Comments' in df.columns:
            metrics_df = df[(df['Likes'] > 0) | (df['Comments'] > 0)].copy()
            
            if len(metrics_df) > 0:
                st.subheader("📊 Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_likes = metrics_df['Likes'].mean()
                    st.metric("Avg Likes", f"{avg_likes:.0f}")
                
                with col2:
                    avg_comments = metrics_df['Comments'].mean()
                    st.metric("Avg Comments", f"{avg_comments:.0f}")
                
                with col3:
                    if 'Shares' in metrics_df.columns:
                        avg_shares = metrics_df['Shares'].mean()
                        st.metric("Avg Shares", f"{avg_shares:.0f}")
                
                with col4:
                    if 'Engagement_Rate' in metrics_df.columns:
                        avg_engagement = metrics_df['Engagement_Rate'].mean()
                        st.metric("Avg Engagement", f"{avg_engagement:.2f}%")
                
                # Author performance
                if len(metrics_df) > 5:
                    st.subheader("👥 Author Performance")
                    author_performance = metrics_df.groupby('Author').agg({
                        'Likes': 'mean',
                        'Comments': 'mean',
                        'Post_Content': 'count'
                    }).round(1)
                    author_performance.columns = ['Avg Likes', 'Avg Comments', 'Post Count']
                    
                    fig = px.scatter(
                        author_performance.reset_index(),
                        x='Avg Likes',
                        y='Avg Comments',
                        size='Post Count',
                        hover_name='Author',
                        title="Author Performance: Likes vs Comments"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Theme analysis
                if 'Theme' in df.columns:
                    st.subheader("🏷️ Content Themes")
                    theme_counts = df['Theme'].value_counts()
                    
                    fig = px.bar(
                        x=theme_counts.values,
                        y=theme_counts.index,
                        orientation='h',
                        title="Post Count by Theme"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Content analysis
        st.subheader("📝 Content Insights")
        
        if len(df) > 0:
            # Word count analysis
            df['word_count'] = df['Post_Content'].fillna('').str.split().str.len()
            
            col1, col2 = st.columns(2)
            
            with col1:
                avg_words = df['word_count'].mean()
                st.metric("Avg Words per Post", f"{avg_words:.0f}")
            
            with col2:
                total_posts = len(df[df['Post_Content'].notna()])
                st.metric("Total Posts Analyzed", total_posts)
            
            # Word count distribution
            fig = px.histogram(
                df,
                x='word_count',
                bins=20,
                title="Distribution of Post Length (Words)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
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

