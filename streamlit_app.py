import streamlit as st
import pandas as pd
from pinecone import Pinecone
import os
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Beauty Product Recommendations",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF69B4;
        margin-bottom: 2rem;
    }
    .product-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .similarity-badge {
        background-color: #FF69B4;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .rating-star {
        color: #FFD700;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load product data from S3"""
    S3_PATH = "s3://recommendation-system-llms/processed_data/products_for_embedding.parquet"
    df = pd.read_parquet(S3_PATH)
    metadata_df = df[["parent_asin", "title", "main_category", "average_rating"]].copy()
    return metadata_df

@st.cache_resource
def get_pinecone_index():
    """Initialize Pinecone connection"""
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index("beauty-products")

def get_recommendations(product_asin, index, metadata_df, top_k=10):
    """Get product recommendations"""
    try:
        # Fetch vector
        result = index.fetch(ids=[product_asin])
        
        if product_asin not in result['vectors']:
            return None, None
        
        query_vector = result['vectors'][product_asin]['values']
        
        # Query similar products
        similar = index.query(
            vector=query_vector,
            top_k=top_k + 1,
            include_values=False
        )
        
        # Get query product details
        query_product = metadata_df[metadata_df['parent_asin'] == product_asin].iloc[0]
        
        # Build recommendations list
        recommendations = []
        for match in similar['matches']:
            if match['id'] == product_asin:
                continue
            
            rec_data = metadata_df[metadata_df['parent_asin'] == match['id']]
            if not rec_data.empty:
                rec = rec_data.iloc[0]
                recommendations.append({
                    'asin': match['id'],
                    'title': rec['title'],
                    'category': rec['main_category'],
                    'rating': rec['average_rating'],
                    'similarity': match['score']
                })
        
        return query_product, recommendations[:top_k]
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

def get_user_recommendations(purchase_history, index, metadata_df, top_k=10):
    """Get recommendations based on purchase history"""
    try:
        # Fetch embeddings
        result = index.fetch(ids=purchase_history)
        
        if not result['vectors']:
            return []
        
        # Average embeddings
        embeddings = [result['vectors'][pid]['values'] for pid in result['vectors']]
        avg_embedding = [sum(x) / len(embeddings) for x in zip(*embeddings)]
        
        # Query
        similar = index.query(
            vector=avg_embedding,
            top_k=top_k + len(purchase_history),
            include_values=False
        )
        
        # Build recommendations
        recommendations = []
        for match in similar['matches']:
            if match['id'] in purchase_history:
                continue
            
            rec_data = metadata_df[metadata_df['parent_asin'] == match['id']]
            if not rec_data.empty:
                rec = rec_data.iloc[0]
                recommendations.append({
                    'asin': match['id'],
                    'title': rec['title'],
                    'category': rec['main_category'],
                    'rating': rec['average_rating'],
                    'similarity': match['score']
                })
        
        return recommendations[:top_k]
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return []

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">💄 Beauty Product Recommendations</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading product database..."):
        metadata_df = load_data()
        index = get_pinecone_index()
    
    st.success(f"✅ Loaded {len(metadata_df):,} products")
    
    # Sidebar
    st.sidebar.header("🔍 Search Options")
    
    search_mode = st.sidebar.radio(
        "Recommendation Mode:",
        ["🛒 Similar Products", "👤 Personalized (Multi-Product)", "📊 Browse by Category"]
    )
    
    # Mode 1: Similar Products
    if search_mode == "🛒 Similar Products":
        st.header("Find Similar Products")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Search by title
            search_query = st.text_input("🔍 Search for a product by title:", "")
            
            if search_query:
                filtered_df = metadata_df[
                    metadata_df['title'].str.contains(search_query, case=False, na=False)
                ]
                
                if not filtered_df.empty:
                    st.write(f"Found {len(filtered_df)} matching products:")
                    
                    # Show search results
                    for idx, row in filtered_df.head(10).iterrows():
                        with st.expander(f"📦 {row['title'][:80]}..."):
                            st.write(f"**Category:** {row['main_category']}")
                            st.write(f"**Rating:** ⭐ {row['average_rating']}/5.0")
                            st.write(f"**ASIN:** {row['parent_asin']}")
                            
                            if st.button("Get Recommendations", key=f"btn_{row['parent_asin']}"):
                                st.session_state['selected_asin'] = row['parent_asin']
                else:
                    st.warning("No products found. Try a different search term.")
            
            # OR select from dropdown
            st.write("**OR select from dropdown:**")
            selected_product = st.selectbox(
                "Choose a product:",
                options=metadata_df['parent_asin'].tolist(),
                format_func=lambda x: metadata_df[metadata_df['parent_asin'] == x]['title'].iloc[0][:80]
            )
            
            if st.button("Get Recommendations for Selected Product", type="primary"):
                st.session_state['selected_asin'] = selected_product
        
        with col2:
            top_k = st.slider("Number of recommendations:", 1, 20, 10)
        
        # Show recommendations
        if 'selected_asin' in st.session_state:
            product_asin = st.session_state['selected_asin']
            
            with st.spinner("Finding similar products..."):
                query_product, recommendations = get_recommendations(
                    product_asin, index, metadata_df, top_k
                )
            
            if query_product is not None and recommendations:
                # Display query product
                st.markdown("---")
                st.subheader("🛒 You Selected:")
                
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**📦 {query_product['title']}**")
                with col2:
                    st.markdown(f"**⭐ {query_product['average_rating']}/5.0**")
                with col3:
                    st.markdown(f"**🏷️ {query_product['main_category']}**")
                
                # Display recommendations
                st.markdown("---")
                st.subheader("💡 Recommended Products:")
                
                # Create DataFrame for display
                recs_df = pd.DataFrame(recommendations)
                recs_df['similarity_pct'] = (recs_df['similarity'] * 100).round(1)
                
                # Visualization
                fig = px.bar(
                    recs_df.head(10),
                    x='similarity_pct',
                    y='title',
                    orientation='h',
                    title='Top 10 Similar Products',
                    labels={'similarity_pct': 'Similarity %', 'title': 'Product'},
                    color='similarity_pct',
                    color_continuous_scale='Pinkyl'
                )
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Cards view
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        col1, col2, col3, col4 = st.columns([4, 2, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{i}. {rec['title'][:80]}...**")
                        
                        with col2:
                            st.markdown(f"🏷️ {rec['category']}")
                        
                        with col3:
                            st.markdown(f"⭐ {rec['rating']}/5.0")
                        
                        with col4:
                            similarity_pct = rec['similarity'] * 100
                            color = "green" if similarity_pct > 80 else "orange" if similarity_pct > 60 else "red"
                            st.markdown(f":{color}[{similarity_pct:.1f}%]")
                        
                        st.markdown(f"*ASIN: {rec['asin']}*")
                        st.markdown("---")
                
                # Download button
                csv = recs_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Recommendations as CSV",
                    data=csv,
                    file_name=f"recommendations_{product_asin}.csv",
                    mime="text/csv"
                )
    
    # Mode 2: Personalized Recommendations
    elif search_mode == "👤 Personalized (Multi-Product)":
        st.header("Personalized Recommendations")
        st.write("Add products to your cart to get personalized recommendations")
        
        if 'cart' not in st.session_state:
            st.session_state['cart'] = []
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("🔍 Search and add products to cart:", "")
            
            if search_query:
                filtered_df = metadata_df[
                    metadata_df['title'].str.contains(search_query, case=False, na=False)
                ]
                
                if not filtered_df.empty:
                    for idx, row in filtered_df.head(5).iterrows():
                        col_a, col_b = st.columns([4, 1])
                        
                        with col_a:
                            st.write(f"📦 {row['title'][:60]}... | ⭐ {row['average_rating']}/5.0")
                        
                        with col_b:
                            if st.button("➕ Add", key=f"add_{row['parent_asin']}"):
                                if row['parent_asin'] not in st.session_state['cart']:
                                    st.session_state['cart'].append(row['parent_asin'])
                                    st.success("Added!")
                                else:
                                    st.warning("Already in cart")
        
        with col2:
            top_k = st.slider("Recommendations:", 1, 20, 10)
        
        # Show cart
        st.subheader("🛒 Your Cart")
        
        if st.session_state['cart']:
            for asin in st.session_state['cart']:
                product = metadata_df[metadata_df['parent_asin'] == asin].iloc[0]
                
                col_a, col_b = st.columns([5, 1])
                
                with col_a:
                    st.write(f"📦 {product['title'][:70]}...")
                
                with col_b:
                    if st.button("🗑️", key=f"remove_{asin}"):
                        st.session_state['cart'].remove(asin)
                        st.rerun()
            
            st.markdown("---")
            
            if st.button("💡 Get Personalized Recommendations", type="primary"):
                with st.spinner("Analyzing your preferences..."):
                    recommendations = get_user_recommendations(
                        st.session_state['cart'], index, metadata_df, top_k
                    )
                
                if recommendations:
                    st.subheader("💡 Recommended Just For You:")
                    
                    for i, rec in enumerate(recommendations, 1):
                        with st.container():
                            col1, col2, col3, col4 = st.columns([4, 2, 1, 1])
                            
                            with col1:
                                st.markdown(f"**{i}. {rec['title'][:80]}...**")
                            
                            with col2:
                                st.markdown(f"🏷️ {rec['category']}")
                            
                            with col3:
                                st.markdown(f"⭐ {rec['rating']}/5.0")
                            
                            with col4:
                                similarity_pct = rec['similarity'] * 100
                                st.markdown(f":green[{similarity_pct:.1f}%]")
                            
                            st.markdown(f"*ASIN: {rec['asin']}*")
                            st.markdown("---")
        else:
            st.info("Your cart is empty. Search and add products above!")
    
    # Mode 3: Browse by Category
    else:
        st.header("📊 Browse by Category")
        
        categories = metadata_df['main_category'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig = px.pie(
                values=categories.values,
                names=categories.index,
                title='Product Distribution by Category'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart
            fig = px.bar(
                x=categories.values,
                y=categories.index,
                orientation='h',
                title='Number of Products per Category',
                labels={'x': 'Count', 'y': 'Category'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Category selector
        selected_category = st.selectbox("Select a category:", categories.index.tolist())
        
        category_products = metadata_df[metadata_df['main_category'] == selected_category]
        
        st.write(f"**{len(category_products)} products in {selected_category}**")
        
        # Show top rated in category
        top_rated = category_products.nlargest(10, 'average_rating')
        
        st.subheader(f"⭐ Top Rated in {selected_category}")
        
        for idx, row in top_rated.iterrows():
            with st.expander(f"⭐ {row['average_rating']}/5.0 - {row['title'][:70]}..."):
                st.write(f"**ASIN:** {row['parent_asin']}")
                if st.button("Get Similar Products", key=f"cat_{row['parent_asin']}"):
                    st.session_state['selected_asin'] = row['parent_asin']
                    st.rerun()

if __name__ == "__main__":
    main()