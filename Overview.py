import streamlit as st
from streamlit.components.v1 import html
from utils import add_title, card_container, close_card

def add_op_overview():
  st.markdown(
    "Purpose: Generate a feasible weekly rota that meets daily coverage requirements "
    "while controlling labour cost and, where specified, promoting fair workload allocation.", unsafe_allow_html=True
  )
  st.markdown(
    """
- Upload a CSV of availability and wage rates (or use the default dataset).
- Select a cost-focused or fairness-aware optimisation objective.
- Run the optimiser and review total cost, assigned hours, and operator utilisation.
""", unsafe_allow_html=True
  )

def add_aed_overview():
  st.markdown(
    "Purpose: Explore historical performance against the NHS 4-hour AED target "
    "and identify patient segments associated with higher breach rates.", unsafe_allow_html=True
  )
  st.markdown(
    """
- Filter and review patient records in the AED Patient List.
- Update historical data (breach status is derived from length of stay).
- Examine interpretable decision-tree summaries showing conditional
breach rates across patient and treatment characteristics.
""", unsafe_allow_html=True
  )

def add_howto():
  st.header("How to use this app")
  st.write(
    "Use the Scheduling module to explore staffing trade-offs under different objectives, and the AED Analytics module to support interpretation of historical breach patterns and operational performance."
  )

def main_content():

  # st.session_state
  if "experiments" not in st.session_state:
    st.session_state.experiments = {"change_log":[], }  # list of dicts

  ## Init container ##
  title_section = st.container()
  howto_section = st.container()
  st.divider()
  col1, col2 = st.columns(2, gap="large")

  ## Layout ##
  with title_section:
    add_title(
      title="Data Management System",
      subtitle_mk="**Team Members**: Ashmi Fathima, Akash Somasundaran, Chia-Te Liu, Muhammad Raahim Sohail, Qutaybah Al Owaifeer, Wei-An Chen, Yi-Rou Lu"
    )
    st.write(
      "This application combines operator scheduling optimisation and AED "
      "performance analytics to support operational planning and monitoring in a single, "
      "interpretable interface."
    )

  with howto_section:
    add_howto()

  with col1:
    st.header("Operator Scheduling")
    with st.container():
      st.markdown('<div class="card">', unsafe_allow_html=True)
      add_op_overview()

  with col2:
    st.header("AED Breach Analytics")
    with st.container():
      st.markdown('<div class="card">', unsafe_allow_html=True)
      add_aed_overview()
  

  

if __name__ == "__main__":
    main_content()
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 0.9em; color: gray;'>
            © 2025 <b>Chia-Te Liu</b>. All rights reserved.  
            Made with ❤️ using Streamlit. Backend built by:  
            <a href='https://www.linkedin.com/in/chia-te-liu/' target='_blank'>Chia-Te Liu</a>
            <a href='https://www.linkedin.com/in/ashmi-fathima/' target='_blank'>Ashmi Fathima</a>
            <a href='https://www.linkedin.com/in/%E8%96%87%E5%AE%89-%E9%99%B3-72531b29a/' target='_blank'>Wei-An Chen</a>
            <a href='https://www.linkedin.com/in/akash-somasundaran0713/' target='_blank'>Akash Somasundaran</a>
            <a href='https://www.linkedin.com/in/qutaybah-alowaifeer-88953a150/' target='_blank'>Qutaybah Alowaifeer</a>
            <a href='https://www.linkedin.com/in/raahimsohail/' target='_blank'>Muhammad Raahim Sohail</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    with open("./international_coal/app/plerdy.html") as f:
      html_string = f.read()
      html(html_string)