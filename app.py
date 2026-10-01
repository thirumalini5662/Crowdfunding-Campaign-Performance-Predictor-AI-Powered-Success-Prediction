import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Crowdfunding Campaign Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "crowdfunding_model.pkl"
    )


model = load_model()


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/crowdfunding_cleaned.csv"
    )


df = load_data()


# =========================================================
# HEADER
# =========================================================

st.sidebar.title("🚀 Crowdfunding AI")

st.sidebar.markdown(
    """
    **Campaign Performance Predictor**

    Analyze historical crowdfunding
    campaigns and estimate the potential
    success of a new campaign.
    """
)

st.sidebar.divider()


# =========================================================
# NAVIGATION
# =========================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Campaign Predictor",
        "📊 Campaign Analytics",
        "🤖 Model Performance",
        "ℹ️ About Project"
    ]
)


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.title(
        "🚀 Crowdfunding Campaign Performance Predictor"
    )

    st.markdown(
        """
        ### AI-powered decision-support tool for entrepreneurs

        This application analyzes historical crowdfunding
        campaign patterns and uses machine learning to
        estimate the probability of success of a proposed
        campaign.
        """
    )

    st.divider()


    # -----------------------------------------------------
    # KPI CALCULATIONS
    # -----------------------------------------------------

    total_campaigns = len(df)

    successful_campaigns = int(
        df["Success"].sum()
    )

    failed_campaigns = (
        total_campaigns -
        successful_campaigns
    )

    historical_success_rate = (
        successful_campaigns /
        total_campaigns
    ) * 100


    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📊 Total Campaigns",
            f"{total_campaigns:,}"
        )

    with col2:

        st.metric(
            "✅ Successful",
            f"{successful_campaigns:,}"
        )

    with col3:

        st.metric(
            "❌ Failed",
            f"{failed_campaigns:,}"
        )

    with col4:

        st.metric(
            "📈 Historical Success Rate",
            f"{historical_success_rate:.2f}%"
        )


    st.divider()


    # -----------------------------------------------------
    # INTRODUCTION
    # -----------------------------------------------------

    st.subheader(
        "🎯 What does this application do?"
    )

    st.write(
        """
        The system uses historical crowdfunding campaign
        data to identify patterns associated with campaign
        outcomes.

        A proposed campaign can then be evaluated using
        factors such as funding goal, category, campaign
        duration, title length, country and launch timing.
        """
    )


    # -----------------------------------------------------
    # WORKFLOW
    # -----------------------------------------------------

    st.subheader(
        "⚙️ Project Workflow"
    )

    workflow = pd.DataFrame({

        "Stage": [
            "1. Historical Data",
            "2. Data Cleaning",
            "3. Feature Engineering",
            "4. Machine Learning",
            "5. Prediction",
            "6. Recommendation"
        ],

        "Purpose": [
            "Collect campaign records",
            "Prepare reliable data",
            "Create predictive variables",
            "Train Logistic Regression",
            "Estimate success probability",
            "Support campaign decisions"
        ]

    })

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        "👉 Use 'Campaign Predictor' in the sidebar "
        "to test a proposed campaign."
    )


# =========================================================
# CAMPAIGN PREDICTOR
# =========================================================

elif page == "🔮 Campaign Predictor":

    st.title(
        "🔮 AI Campaign Success Predictor"
    )

    st.write(
        "Enter the proposed campaign details below."
    )

    st.divider()


    # -----------------------------------------------------
    # INPUTS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        campaign_name = st.text_input(
            "Campaign Name",
            "Innovative Technology Project"
        )


        categories = sorted(
            df["Category"]
            .dropna()
            .unique()
            .tolist()
        )


        category = st.selectbox(
            "Category",
            categories
        )


        subcategories = sorted(
            df[
                df["Category"] == category
            ]["Subcategory"]
            .dropna()
            .unique()
            .tolist()
        )


        if len(subcategories) == 0:

            subcategory = st.text_input(
                "Subcategory",
                "Other"
            )

        else:

            subcategory = st.selectbox(
                "Subcategory",
                subcategories
            )


        countries = sorted(
            df["Country"]
            .dropna()
            .unique()
            .tolist()
        )


        country = st.selectbox(
            "Country",
            countries
        )


    with col2:

        goal_amount = st.number_input(
            "Funding Goal",
            min_value=1.0,
            value=5000.0,
            step=100.0
        )


        duration = st.number_input(
            "Campaign Duration (Days)",
            min_value=1,
            max_value=100,
            value=30
        )


        launch_month = st.slider(
            "Launch Month",
            min_value=1,
            max_value=12,
            value=9
        )


        launch_year = st.number_input(
            "Launch Year",
            min_value=2009,
            max_value=2030,
            value=2017
        )


        launch_day = st.selectbox(
            "Launch Day",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )


    st.divider()


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    predict_button = st.button(
        "🔮 PREDICT CAMPAIGN SUCCESS",
        use_container_width=True
    )


    if predict_button:

        # ---------------------------------------------
        # CREATE FEATURES
        # ---------------------------------------------

        title_length = len(
            campaign_name
        )


        input_data = pd.DataFrame({

            "Goal_Amount": [goal_amount],

            "Category": [category],

            "Subcategory": [subcategory],

            "Country": [country],

            "Duration_Days": [duration],

            "Title_Length": [title_length],

            "Launch_Month": [launch_month],

            "Launch_Year": [launch_year],

            "Launch_Day_of_Week": [launch_day]

        })


        # ---------------------------------------------
        # MODEL PREDICTION
        # ---------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        probability = model.predict_proba(
            input_data
        )[0][1]


        probability_percentage = (
            probability * 100
        )


        # ---------------------------------------------
        # RISK CLASSIFICATION
        # ---------------------------------------------

        if probability_percentage >= 70:

            risk_level = "Low"

        elif probability_percentage >= 40:

            risk_level = "Moderate"

        else:

            risk_level = "High"


        # ---------------------------------------------
        # RESULT
        # ---------------------------------------------

        st.divider()

        st.header(
            "📊 Campaign Prediction"
        )


        # ---------------------------------------------
        # SCORECARD
        # ---------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "🎯 Success Probability",
                f"{probability_percentage:.2f}%"
            )


        with col2:

            if prediction == 1:

                st.success(
                    "🟢 LIKELY SUCCESSFUL"
                )

            else:

                st.error(
                    "🔴 LIKELY UNSUCCESSFUL"
                )


        with col3:

            st.metric(
                "⚠️ Risk Level",
                risk_level
            )


        # ---------------------------------------------
        # PROBABILITY BAR
        # ---------------------------------------------

        st.subheader(
            "Success Probability"
        )

        st.progress(
            float(probability)
        )


        # ---------------------------------------------
        # SCORECARD DETAILS
        # ---------------------------------------------

        st.subheader(
            "📋 Campaign Scorecard"
        )


        scorecard = pd.DataFrame({

            "Factor": [
                "Campaign Name",
                "Category",
                "Subcategory",
                "Country",
                "Funding Goal",
                "Duration",
                "Title Length",
                "Launch Month",
                "Launch Year",
                "Launch Day"
            ],

            "Value": [
                campaign_name,
                category,
                subcategory,
                country,
                f"{goal_amount:,.2f}",
                f"{duration} days",
                f"{title_length} characters",
                launch_month,
                launch_year,
                launch_day
            ]

        })


        st.dataframe(
            scorecard,
            use_container_width=True,
            hide_index=True
        )


        # ---------------------------------------------
        # MODEL INPUT FACTORS
        # ---------------------------------------------

        st.subheader(
            "🔍 Model Input Factors"
        )

        factor_cols = st.columns(5)


        factors = [
            "Funding Goal",
            "Category",
            "Duration",
            "Title Length",
            "Launch Timing"
        ]


        for i, factor in enumerate(factors):

            with factor_cols[i]:

                st.info(
                    f"✓ {factor}"
                )


        # ---------------------------------------------
        # RECOMMENDATIONS
        # ---------------------------------------------

        st.subheader(
            "💡 Entrepreneurial Recommendations"
        )


        recommendations = []


        # Goal recommendation

        median_goal = df[
            "Goal_Amount"
        ].median()


        if goal_amount > median_goal * 2:

            recommendations.append(
                "💰 The proposed funding goal is substantially "
                "above the historical median. Review the funding "
                "requirement and target market before launch."
            )

        else:

            recommendations.append(
                "💰 The proposed funding goal is within a "
                "moderate range relative to the historical dataset."
            )


        # Duration recommendation

        median_duration = df[
            "Duration_Days"
        ].median()


        if duration > median_duration * 2:

            recommendations.append(
                "⏱️ The campaign duration is considerably longer "
                "than the historical median. Review whether the "
                "extended duration is necessary."
            )

        else:

            recommendations.append(
                "⏱️ The campaign duration is within a reasonable "
                "range compared with historical campaigns."
            )


        # Title recommendation

        if title_length < 10:

            recommendations.append(
                "✏️ Consider using a clearer and more descriptive "
                "campaign title."
            )

        elif title_length > 80:

            recommendations.append(
                "✏️ Consider shortening the campaign title so "
                "the main idea is immediately clear."
            )

        else:

            recommendations.append(
                "✏️ The campaign title has a moderate length."
            )


        # Probability recommendation

        if probability_percentage >= 70:

            recommendations.append(
                "📈 The model estimates a relatively high "
                "historical-pattern probability. Continue "
                "validating your audience, funding requirement "
                "and promotional strategy."
            )

        elif probability_percentage >= 40:

            recommendations.append(
                "📊 The model estimates a moderate probability. "
                "Consider reviewing the campaign strategy before launch."
            )

        else:

            recommendations.append(
                "⚠️ The model estimates a lower probability. "
                "Consider revisiting the funding goal, campaign "
                "positioning and launch strategy."
            )


        for recommendation in recommendations:

            st.write(
                recommendation
            )


        # ---------------------------------------------
        # DISCLAIMER
        # ---------------------------------------------

        st.caption(
            "Note: The probability is an estimate based on "
            "historical campaign patterns and should not be "
            "treated as a guarantee of campaign success."
        )


        # ---------------------------------------------
        # DOWNLOAD RESULT
        # ---------------------------------------------

        result_data = pd.DataFrame({

            "Campaign_Name": [campaign_name],

            "Category": [category],

            "Subcategory": [subcategory],

            "Country": [country],

            "Goal_Amount": [goal_amount],

            "Duration_Days": [duration],

            "Launch_Month": [launch_month],

            "Launch_Year": [launch_year],

            "Launch_Day": [launch_day],

            "Success_Probability": [
                round(
                    probability_percentage,
                    2
                )
            ],

            "Risk_Level": [risk_level],

            "Prediction": [
                "Successful"
                if prediction == 1
                else "Unsuccessful"
            ]

        })


        csv_result = result_data.to_csv(
            index=False
        )


        st.download_button(
            "⬇️ Download Prediction Report",
            csv_result,
            "campaign_prediction.csv",
            "text/csv"
        )


# =========================================================
# CAMPAIGN ANALYTICS
# =========================================================

elif page == "📊 Campaign Analytics":

    st.title(
        "📊 Historical Campaign Analytics"
    )

    st.write(
        "Explore patterns across historical crowdfunding campaigns."
    )


    # -----------------------------------------------------
    # KPIs
    # -----------------------------------------------------

    total = len(df)

    successful = int(
        df["Success"].sum()
    )

    failed = (
        total -
        successful
    )

    success_rate = (
        successful /
        total
    ) * 100


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Campaigns",
            f"{total:,}"
        )


    with col2:

        st.metric(
            "Successful",
            f"{successful:,}"
        )


    with col3:

        st.metric(
            "Failed",
            f"{failed:,}"
        )


    with col4:

        st.metric(
            "Success Rate",
            f"{success_rate:.2f}%"
        )


    st.divider()


    # -----------------------------------------------------
    # CATEGORY SUCCESS
    # -----------------------------------------------------

    category_data = (
        df.groupby("Category")
        ["Success"]
        .mean()
        .reset_index()
    )


    category_data["Success Rate"] = (
        category_data["Success"] * 100
    )


    category_data = (
        category_data
        .sort_values(
            "Success Rate",
            ascending=False
        )
        .head(10)
    )


    fig_category = px.bar(
        category_data,
        x="Success Rate",
        y="Category",
        orientation="h",
        title="Top Categories by Historical Success Rate",
        labels={
            "Success Rate": "Success Rate (%)"
        }
    )


    st.plotly_chart(
        fig_category,
        use_container_width=True
    )


    # -----------------------------------------------------
    # YEARLY TREND
    # -----------------------------------------------------

    yearly_data = (
        df.groupby("Launch_Year")
        ["Success"]
        .mean()
        .reset_index()
    )


    yearly_data["Success Rate"] = (
        yearly_data["Success"] * 100
    )


    fig_year = px.line(
        yearly_data,
        x="Launch_Year",
        y="Success Rate",
        markers=True,
        title="Historical Success Rate by Launch Year",
        labels={
            "Success Rate": "Success Rate (%)",
            "Launch_Year": "Launch Year"
        }
    )


    st.plotly_chart(
        fig_year,
        use_container_width=True
    )


    # -----------------------------------------------------
    # SUCCESS DISTRIBUTION
    # -----------------------------------------------------

    status_data = pd.DataFrame({

        "Status": [
            "Failed",
            "Successful"
        ],

        "Campaigns": [
            failed,
            successful
        ]

    })


    fig_status = px.pie(
        status_data,
        names="Status",
        values="Campaigns",
        title="Campaign Success Distribution"
    )


    st.plotly_chart(
        fig_status,
        use_container_width=True
    )


    # -----------------------------------------------------
    # DURATION ANALYSIS
    # -----------------------------------------------------

    duration_data = (
        df.groupby("Success")
        ["Duration_Days"]
        .mean()
        .reset_index()
    )


    duration_data["Status"] = (
        duration_data["Success"]
        .map({
            0: "Failed",
            1: "Successful"
        })
    )


    fig_duration = px.bar(
        duration_data,
        x="Status",
        y="Duration_Days",
        title="Average Campaign Duration by Outcome",
        labels={
            "Duration_Days": "Average Duration (Days)"
        }
    )


    st.plotly_chart(
        fig_duration,
        use_container_width=True
    )


    # -----------------------------------------------------
    # GOAL ANALYSIS
    # -----------------------------------------------------

    goal_data = df[
        df["Goal_Amount"]
        <= df["Goal_Amount"].quantile(
            0.95
        )
    ]


    fig_goal = px.box(
        goal_data,
        x="Success",
        y="Goal_Amount",
        title="Funding Goal Distribution by Outcome",
        labels={
            "Success": "Outcome",
            "Goal_Amount": "Funding Goal"
        }
    )


    fig_goal.update_xaxes(
        tickvals=[0, 1],
        ticktext=[
            "Failed",
            "Successful"
        ]
    )


    st.plotly_chart(
        fig_goal,
        use_container_width=True
    )


    # -----------------------------------------------------
    # BACKER ANALYSIS
    # -----------------------------------------------------

    backer_data = (
        df.groupby("Success")
        ["Backer_Count"]
        .mean()
        .reset_index()
    )


    backer_data["Status"] = (
        backer_data["Success"]
        .map({
            0: "Failed",
            1: "Successful"
        })
    )


    fig_backers = px.bar(
        backer_data,
        x="Status",
        y="Backer_Count",
        title="Average Backers by Campaign Outcome",
        labels={
            "Backer_Count": "Average Backers"
        }
    )


    st.plotly_chart(
        fig_backers,
        use_container_width=True
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🤖 Model Performance":

    st.title(
        "🤖 Machine Learning Model Performance"
    )

    st.write(
        """
        The predictor uses Logistic Regression to classify
        campaigns as successful or unsuccessful.
        """
    )


    # -----------------------------------------------------
    # PREPARE MODEL TEST DATA
    # -----------------------------------------------------

    features = [

        "Goal_Amount",
        "Category",
        "Subcategory",
        "Country",
        "Duration_Days",
        "Title_Length",
        "Launch_Month",
        "Launch_Year",
        "Launch_Day_of_Week"

    ]


    X = df[features]

    y = df["Success"]


    # We recreate the same split used during training

    from sklearn.model_selection import train_test_split


    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y

    )


    y_pred = model.predict(
        X_test
    )


    y_probability = model.predict_proba(
        X_test
    )[:, 1]


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    # -----------------------------------------------------
    # METRIC CARDS
    # -----------------------------------------------------

    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )


    with col3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )


    with col4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )


    with col5:

        st.metric(
            "ROC-AUC",
            f"{roc_auc:.3f}"
        )


    st.divider()


    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(
            x="Predicted",
            y="Actual",
            color="Count"
        ),
        x=["Failed", "Successful"],
        y=["Failed", "Successful"],
        title="Confusion Matrix"
    )


    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )


    # -----------------------------------------------------
    # ROC CURVE
    # -----------------------------------------------------

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_probability
    )


    fig_roc = go.Figure()


    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Model (AUC = {roc_auc:.3f})"
        )
    )


    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier"
        )
    )


    fig_roc.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )


    st.plotly_chart(
        fig_roc,
        use_container_width=True
    )


    st.info(
        "These metrics evaluate the model on the same "
        "20% test split used during model development."
    )


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "ℹ️ About Project":

    st.title(
        "ℹ️ About the Project"
    )


    st.subheader(
        "Project Title"
    )

    st.write(
        "Crowdfunding Campaign Performance Predictor"
    )


    st.subheader(
        "🎯 Objective"
    )

    st.write(
        """
        To develop an AI-powered tool that analyzes historical
        crowdfunding campaigns and estimates the success
        probability of a new campaign.
        """
    )


    st.subheader(
        "🤖 Machine Learning"
    )

    st.write(
        """
        Logistic Regression is used to classify campaigns as
        Successful or Unsuccessful and to generate a probability
        estimate.
        """
    )


    st.subheader(
        "📊 Main Variables"
    )

    st.write(
        """
        • Funding Goal

        • Category

        • Subcategory

        • Country

        • Campaign Duration

        • Title Length

        • Launch Month

        • Launch Year

        • Launch Day of Week
        """
    )


    st.subheader(
        "🛠️ Technology Stack"
    )

    st.write(
        """
        Python • Pandas • NumPy • Scikit-learn • Plotly • Streamlit
        """
    )


    st.subheader(
        "⚠️ Important Note"
    )

    st.warning(
        """
        The prediction is an estimate based on historical
        crowdfunding patterns. It is not a guarantee of
        future campaign performance.
        """
    )


    st.success(
        "🚀 Crowdfunding Campaign Performance Predictor"
    )