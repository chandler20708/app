import streamlit as st
from utils import add_title, card_container, close_card

def add_op_overview():
  st.markdown(
    "Purpose: Generate a feasible weekly rota that meets daily coverage requirements"
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
    "Purpose: Explore historical performance against the NHS 4-hour AED target"
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
      "This application combines operator scheduling optimisation and AED"
      "performance analytics to support operational planning and monitoring in a single,"
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
    st.header("AED Patient Flow & Breach Analytics")
    with st.container():
      st.markdown('<div class="card">', unsafe_allow_html=True)
      add_aed_overview()
  

  

if __name__ == "__main__":
  main_content()
