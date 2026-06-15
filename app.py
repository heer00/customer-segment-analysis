import streamlit as st
import requests

# Set page configuration
st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🎯",
    layout="wide",
)

BACKEND_URL = "http://127.0.0.1:8000/api/v1"

st.title("🎯 AI-Powered Customer Intelligence Platform")
st.markdown("---")

# Sidebar Navigation
app_mode = st.sidebar.selectbox("Choose Mode", ["Predict Segment", "Manage Customer Profiles"])

if app_mode == "Predict Segment":
    st.subheader("Predict Customer Segment")
    
    # Explanatory text for input ranges
    st.info(
        "💡 **What do these parameters mean?**\n"
        "- **Annual Income (₹ in Lakhs)**: The customer's total yearly income in Lakhs of Indian Rupees (e.g. entering `15.0` represents ₹15 Lakhs per year).\n"
        "- **Spending Score (1-100)**: A relative score assigned based on shopping frequency, average transaction size, and mall loyalty. (1 = lowest spending/frequency, 100 = highest spending/frequency)."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Annual Income (₹ in Lakhs)", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
    with col2:
        spending = st.number_input("Spending Score (1-100)", min_value=1.0, max_value=100.0, value=50.0, step=1.0)
        
    if st.button("Predict Segment", type="primary"):
        # Make API request to the FastAPI backend instead of local pickle loading
        try:
            response = requests.post(
                f"{BACKEND_URL}/predict/",
                json={"annual_income": income, "spending_score": spending}
            )
            if response.status_code == 200:
                result = response.json()
                st.balloons()
                st.success(f"### Predicted Segment: **{result['segment_label']}**")
                st.markdown(f"**Profile Description**: *{result['segment_description']}*")
                
                # Show API response metadata
                with st.expander("Show Technical API Response"):
                    st.json(result)
            else:
                st.error(f"Error from API: {response.text}")
        except Exception as e:
            st.error(f"Could not connect to FastAPI backend: {e}. Please make sure your FastAPI server is running on port 8000.")

elif app_mode == "Manage Customer Profiles":
    st.subheader("Manage Customer Profiles")
    
    tab1, tab2 = st.tabs(["Register New Customer", "Registered Customers & History"])
    
    with tab1:
        st.markdown("### Register a Customer Profile")
        st.write("Creating a profile registers the customer in the SQLite database and automatically runs their initial prediction.")
        
        with st.form("register_customer_form"):
            name = st.text_input("Full Name (e.g., Rajesh Kumar)")
            email = st.text_input("Email Address")
            income = st.number_input("Annual Income (₹ in Lakhs)", min_value=0.0, max_value=200.0, value=50.0)
            spending = st.number_input("Spending Score (1-100)", min_value=1.0, max_value=100.0, value=50.0)
            
            submitted = st.form_submit_button("Register Customer", type="primary")
            if submitted:
                if not name or not email:
                    st.warning("Please enter both a name and an email address.")
                else:
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/customers/",
                            json={
                                "name": name,
                                "email": email,
                                "annual_income": income,
                                "spending_score": spending
                            }
                        )
                        if response.status_code == 201:
                            st.success(f"Customer '{name}' registered successfully!")
                            st.json(response.json())
                        else:
                            st.error(f"Registration failed: {response.json().get('detail', response.text)}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
                        
    with tab2:
        st.markdown("### Registered Database Records")
        try:
            response = requests.get(f"{BACKEND_URL}/customers/")
            if response.status_code == 200:
                customers = response.json()
                if not customers:
                    st.info("No registered customers found in database.")
                else:
                    for cust in customers:
                        with st.expander(f"👤 {cust['name']} ({cust['email']}) — ID: {cust['id']}"):
                            st.write(f"**Annual Income:** ₹{cust['annual_income']} Lakhs")
                            st.write(f"**Spending Score:** {cust['spending_score']}/100")
                            st.write(f"**Created At:** {cust['created_at']}")
                            
                            # Fetch predictions history for this specific customer
                            hist_response = requests.get(f"{BACKEND_URL}/customers/{cust['id']}/predictions")
                            if hist_response.status_code == 200:
                                history = hist_response.json()
                                if history:
                                    st.write("**Prediction History:**")
                                    for h in history:
                                        st.write(
                                            f"- *{h['predicted_at'][:19].replace('T', ' ')}*: "
                                            f"**{h['segment_label']}** (Income: ₹{h['annual_income']}L, Score: {h['spending_score']})"
                                        )
                            else:
                                st.error("Failed to load history for this customer.")
            else:
                st.error("Failed to fetch customer list from backend.")
        except Exception as e:
            st.error(f"Connection error: {e}")