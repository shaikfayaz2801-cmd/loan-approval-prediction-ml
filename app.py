import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
<style>s

.hero{
    padding:35px;
    border-radius:20px;
    background:linear-gradient(135deg,#0f172a,#2563eb);
    color:white;
    text-align:center;
    margin-bottom:30px;
}

.stButton > button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "form"

if "result" not in st.session_state:
    st.session_state.result = None

st.markdown("""
<div class="hero">
    <h1>🏦 Loan Approval Prediction System</h1>
    <p style="font-size:20px;">
        Decision Tree Based Intelligent Loan Assessment
    </p>
</div>
""", unsafe_allow_html=True)

df = pd.read_csv(
    "loan_approval_decision_tree_dataset.csv"
)

emp_encoder = LabelEncoder()
edu_encoder = LabelEncoder()
loan_encoder = LabelEncoder()

df["Employment_Status"] = emp_encoder.fit_transform(
    df["Employment_Status"]
)

df["Education_Level"] = edu_encoder.fit_transform(
    df["Education_Level"]
)

df["Loan_Status"] = loan_encoder.fit_transform(
    df["Loan_Status"]
)

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

if st.session_state.page == "form":

    st.subheader("👤 Applicant Information")

    left, right = st.columns(2)

    with left:

        income = st.number_input(
            "💰 Annual Income",
            min_value=0.0,
            value=None,
            placeholder="Enter annual income"
        )

        credit_score = st.number_input(
            "📊 Credit Score",
            min_value=300,
            max_value=900,
            value=None,
            placeholder="Enter credit score"
        )

        loan_amount = st.number_input(
            "🏦 Loan Amount",
            min_value=0.0,
            value=None,
            placeholder="Enter loan amount"
        )

    with right:

        employment = st.selectbox(
            "👨‍💼 Employment Status",
            ["Select Employment"] +
            list(emp_encoder.classes_)
        )
        education = st.selectbox(
            "🎓 Education Level",
            ["Select Education"] +
            list(edu_encoder.classes_)
        )
        debt = st.number_input(
            "💳 Existing Debt",
            min_value=0.0,
            value=None,
            placeholder="Enter existing debt"
        )
    st.write("")
    if st.button(
        "🔍 Predict Loan Status",
        use_container_width=True,
        type="primary"
    ):
        if (
            income is None or
            credit_score is None or
            loan_amount is None or
            debt is None or
            employment == "Select Employment" or
            education == "Select Education"
        ):
            st.error(
                "⚠ Please fill all fields before prediction."
            )

        else:
            emp = emp_encoder.transform(
                [employment]
            )[0]
            edu = edu_encoder.transform(
                [education]
            )[0]
            input_data = [[
                income,
                credit_score,
                loan_amount,
                emp,
                edu,
                debt
            ]]
            prediction = model.predict(
                input_data
            )
            result = loan_encoder.inverse_transform(
                prediction
            )[0]
            st.session_state.result = result
            st.session_state.page = "result"
            st.rerun()
elif st.session_state.page == "result":

    result = str(
        st.session_state.result
    ).lower()

    approved_values = [
        "approved",
        "approve",
        "yes",
        "1"
    ]
    if result in approved_values:
        st.markdown("""
        <div style="
        background:linear-gradient(
            135deg,
            #10b981,
            #059669
        );
        padding:50px;
        border-radius:20px;
        text-align:center;
        color:white;
        margin-top:40px;
        ">
            <h1 style="font-size:70px;">✅</h1>
            <h1>LOAN APPROVED</h1>
            <h3>
            Your application satisfies the approval criteria.
            </h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
        background:linear-gradient(
            135deg,
            #ef4444,
            #b91c1c
        );
        padding:50px;
        border-radius:20px;
        text-align:center;
        color:white;
        margin-top:40px;
        ">
            <h1 style="font-size:70px;">❌</h1>
            <h1>LOAN REJECTED</h1>
            <h3>
            Your application does not satisfy the approval criteria.
            </h3>
        </div>
        """, unsafe_allow_html=True)
    st.write("")
    if st.button(
        "Back ",
        use_container_width=True
    ):
        st.session_state.page = "form"
        st.session_state.result = None
        st.rerun()

