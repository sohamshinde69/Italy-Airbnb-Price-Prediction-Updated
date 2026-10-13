import streamlit as st  # for building the web app
import pandas as pd  # for data handling
import pickle  # for loading trained model files
import plotly.express as px  # for interactive charts and map


st.set_page_config(
    page_title="Italy Airbnb Price Prediction",
    page_icon="🏠",
    layout="wide"
)

DATA_PATH = "italy_airbnb_dropped_testing.csv"
MODEL_PATH = "model_testing.pkl"

# Hardcoded test-set metrics from dropped_data.ipynb (Gradient Boosting).
# Update these three numbers if you retrain the model.
MODEL_METRICS = {"mae": 83.46, "rmse": 268.37, "r2": 0.089}

PLACEHOLDER_IMAGE = "https://placehold.co/300x200?text=No+Photo"


# -----------------------------
# Cached loaders
# -----------------------------

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model = load_model()
data = load_data()

preprocessor = model.named_steps["preprocessor"]
encoder = preprocessor.named_transformers_["cat"]

property_types = encoder.categories_[0].tolist()
room_types = encoder.categories_[1].tolist()
cities = encoder.categories_[2].tolist()

# These flip on automatically once the columns exist in the CSV — no other
# code needs to change if you add more columns like this later.
HAS_LISTING_NAME = "name" in data.columns
HAS_PICTURE_URL = "picture_url" in data.columns


# -----------------------------
# Sidebar navigation
# -----------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["💰 Price Prediction", "📊 Data Analysis", "🗺️ Map Explorer"]
)

st.sidebar.divider()
st.sidebar.caption(
    f"Model: Gradient Boosting Regressor\n\n"
    f"MAE: €{MODEL_METRICS['mae']:.0f}  |  "
    f"RMSE: €{MODEL_METRICS['rmse']:.0f}  |  "
    f"R²: {MODEL_METRICS['r2']:.3f}"
)


# =========================================================
# PAGE 1 — PRICE PREDICTION
# =========================================================

if page == "💰 Price Prediction":

    st.title("🏠 Italy Airbnb Price Prediction")
    st.write(
        "Enter the details of an Airbnb listing to estimate its nightly price, "
        "and see similar real listings from the dataset."
    )
    st.divider()

    st.subheader("🏡 Property Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.selectbox("City", cities)
    with col2:
        room_type = st.selectbox("Room Type", room_types)
    with col3:
        property_type = st.selectbox("Property Type", property_types)

    st.subheader("📋 Listing Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        accommodates = st.number_input(
            "Accommodates", min_value=1, max_value=20, value=2, step=1
        )
    with col2:
        bedrooms = st.number_input(
            "Bedrooms", min_value=0.0, max_value=20.0, value=1.0, step=1.0
        )
    with col3:
        bathrooms = st.number_input(
            "Bathrooms", min_value=0.0, max_value=20.0, value=1.0, step=0.5
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        minimum_nights = st.number_input(
            "Minimum Nights", min_value=1, max_value=365, value=1, step=1
        )
    with col2:
        number_of_reviews = st.number_input(
            "Number of Reviews", min_value=0, max_value=1000, value=10, step=1
        )
    with col3:
        review_scores_rating = st.number_input(
            "Review Score Rating", min_value=0.0, max_value=5.0, value=4.5, step=0.1
        )

    st.divider()

    if st.button("💰 Predict Price", use_container_width=True):

        input_data = pd.DataFrame({
            "accommodates": [accommodates],
            "bathrooms": [bathrooms],
            "bedrooms": [bedrooms],
            "minimum_nights": [minimum_nights],
            "number_of_reviews": [number_of_reviews],
            "review_scores_rating": [review_scores_rating],
            "property_type": [property_type],
            "room_type": [room_type],
            "city": [city]
        })

        prediction = model.predict(input_data)[0]
        prediction = max(0, prediction)  # guard against unrealistic negative output

        # remember the inputs/result so the "similar listings" section below
        # stays visible across reruns (e.g. when Streamlit reruns on interaction)
        st.session_state["last_prediction"] = prediction
        st.session_state["last_inputs"] = {
            "city": city,
            "room_type": room_type,
            "property_type": property_type,
            "accommodates": accommodates,
            "bedrooms": bedrooms,
        }

    # Show the result + similar listings if a prediction has been made this session
    if "last_prediction" in st.session_state:
        prediction = st.session_state["last_prediction"]
        inputs = st.session_state["last_inputs"]

        st.success(f"Estimated Airbnb Price: €{prediction:,.2f} per night")
        st.caption(
            f"Typical model error is about ±€{MODEL_METRICS['mae']:.0f}, so a realistic "
            f"range is roughly €{max(0, prediction - MODEL_METRICS['mae']):,.0f} – "
            f"€{prediction + MODEL_METRICS['mae']:,.0f}."
        )

        st.subheader(f"🏘️ Similar listings in {inputs['city']}")

        similar = data[
            (data["city"] == inputs["city"]) &
            (data["room_type"] == inputs["room_type"])
        ].copy()

        if similar.empty:
            st.info("No listings found matching this city and room type.")
        else:
            # rank by closeness to the entered accommodates/bedrooms, so
            # "similar" means similar-sized listings, not just same city
            similar["similarity_score"] = (
                (similar["accommodates"] - inputs["accommodates"]).abs() +
                (similar["bedrooms"] - inputs["bedrooms"]).abs()
            )
            similar = similar.sort_values("similarity_score").head(10)

            if HAS_PICTURE_URL:
                # card layout: photo + details, 5 per row
                cols_per_row = 5
                rows = [similar.iloc[i:i + cols_per_row] for i in range(0, len(similar), cols_per_row)]

                for row_chunk in rows:
                    cols = st.columns(cols_per_row)
                    for col, (_, listing) in zip(cols, row_chunk.iterrows()):
                        with col:
                            image_url = listing.get("picture_url")
                            if pd.isna(image_url) or not str(image_url).startswith("http"):
                                image_url = PLACEHOLDER_IMAGE
                            try:
                                st.image(image_url, use_container_width=True)
                            except Exception:
                                st.image(PLACEHOLDER_IMAGE, use_container_width=True)

                            title = listing["name"] if HAS_LISTING_NAME and pd.notna(listing.get("name")) else listing["property_type"]
                            st.markdown(f"**{title}**")
                            st.caption(
                                f"{listing['room_type']} · {int(listing['accommodates'])} guests · "
                                f"{listing['bedrooms']:.0f} bd"
                            )
                            st.write(f"€{listing['price']:,.0f}/night · ⭐ {listing['review_scores_rating']:.1f}")
            else:
                # fallback: plain table if picture_url isn't in the dataset yet
                display_cols = []
                if HAS_LISTING_NAME:
                    display_cols.append("name")
                display_cols += ["property_type", "room_type", "accommodates", "bedrooms",
                                  "bathrooms", "price", "review_scores_rating"]

                st.dataframe(
                    similar[display_cols].rename(columns={
                        "property_type": "Property Type",
                        "room_type": "Room Type",
                        "accommodates": "Accommodates",
                        "bedrooms": "Bedrooms",
                        "bathrooms": "Bathrooms",
                        "price": "Price (€)",
                        "review_scores_rating": "Rating",
                        "name": "Listing Name",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            # mini map of just these similar listings
            st.map(similar.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])


# =========================================================
# PAGE 2 — DATA ANALYSIS
# =========================================================

elif page == "📊 Data Analysis":

    st.title("📊 Data Analysis")
    st.write("Key patterns in the Italy Airbnb dataset (mirrors the SQL/Power BI analysis).")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Listings", f"{len(data):,}")
    col2.metric("Avg Price", f"€{data['price'].mean():,.0f}")
    col3.metric("Cities Covered", data["city"].nunique())
    col4.metric("Avg Rating", f"{data['review_scores_rating'].mean():.2f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Price by City")
        price_by_city = (
            data.groupby("city")["price"].mean().sort_values(ascending=False).reset_index()
        )
        fig = px.bar(price_by_city, x="city", y="price", labels={"price": "Avg Price (€)", "city": "City"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Listings by Room Type")
        room_counts = data["room_type"].value_counts().reset_index()
        room_counts.columns = ["room_type", "count"]
        fig = px.pie(room_counts, names="room_type", values="count")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Distribution")
        fig = px.histogram(
            data[data["price"] < data["price"].quantile(0.99)],  # trim extreme outliers for readability
            x="price", nbins=50, labels={"price": "Price (€)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Listings by Property Type (Top 10)")
        prop_counts = data["property_type"].value_counts().head(10).reset_index()
        prop_counts.columns = ["property_type", "count"]
        fig = px.bar(
            prop_counts.sort_values("count"), x="count", y="property_type",
            orientation="h", labels={"count": "Number of Listings", "property_type": "Property Type"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Price by Room Type and City")
    pivot = data.groupby(["city", "room_type"])["price"].mean().reset_index()
    fig = px.bar(
        pivot, x="city", y="price", color="room_type", barmode="group",
        labels={"price": "Avg Price (€)", "city": "City", "room_type": "Room Type"}
    )
    st.plotly_chart(fig, use_container_width=True)

    if HAS_PICTURE_URL:
        st.subheader("Sample Listings")
        sample = data.dropna(subset=["picture_url"]).sample(min(5, len(data)), random_state=1)
        cols = st.columns(len(sample))
        for col, (_, listing) in zip(cols, sample.iterrows()):
            with col:
                try:
                    st.image(listing["picture_url"], use_container_width=True)
                except Exception:
                    st.image(PLACEHOLDER_IMAGE, use_container_width=True)
                st.caption(f"{listing['city']} · €{listing['price']:,.0f}/night")

    st.subheader("Model Performance Comparison")
    st.caption(
        "Compared on the held-out test set across Random Forest, Linear Regression, "
        "and Gradient Boosting — the project used the null-dropped dataset since it "
        "outperformed the imputed version on every metric."
    )
    metrics_df = pd.DataFrame({
        "Model": ["Random Forest", "Linear Regression", "Gradient Boosting"],
        "MAE (€)": [95.03, 86.97, 83.46],
        "RMSE (€)": [292.62, 256.46, 268.37],
        "R²": [-0.083, 0.168, 0.089],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# =========================================================
# PAGE 3 — MAP EXPLORER
# =========================================================

elif page == "🗺️ Map Explorer":

    st.title("🗺️ Geographic Explorer")
    st.write("Explore listings by location, filtered by city and price range.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        map_city = st.multiselect("City", sorted(data["city"].unique()), default=list(data["city"].unique()))
    with col2:
        price_range = st.slider(
            "Price range (€)",
            min_value=0,
            max_value=int(data["price"].quantile(0.99)),
            value=(0, int(data["price"].quantile(0.9))),
        )

    filtered = data[
        (data["city"].isin(map_city)) &
        (data["price"].between(price_range[0], price_range[1]))
    ]

    st.caption(f"Showing {len(filtered):,} of {len(data):,} listings")

    
    fig = px.scatter_map(
        filtered,
        lat="latitude",
        lon="longitude",
        color="price",
        size="accommodates",
        hover_data=["city", "room_type", "property_type", "price"],
        color_continuous_scale="Viridis",
        zoom=4.5,
        height=650,
        map_style="open-street-map",
    )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    st.plotly_chart(fig, use_container_width=True)
